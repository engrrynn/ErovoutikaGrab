#!/usr/bin/env python3
"""Unified Web Teleoperation & Multi-System Dashboard for ErovoutikaGrab.

Accessible at:
   👉 egrabbot.local:5001
   👉 localhost:5001

Features:
1. 🖥️ Dynamic Camera Feed Destination: Robot LCD (Default) vs. Web Browser.
2. 👁️ View-Only Mode on Robot LCD: Fullscreen HUD with zero buttons/clutter.
   - If Feed on LCD: Fullscreen live camera feed with HUD overlays.
   - If Feed on Web: Centered Erovoutika Logo + Extra-Large Character Telemetry Gauge Cluster.
3. 🕹️ Interactive Web Cockpit:
   - Live Video Stream & Click-to-Position Sweet Spot Calibration
   - Motor Teleoperation (Touch D-Pad, WASD Keyboard Controls, Micro-Nudges)
   - 3-Axis Servo Arm Teleoperation (S1/S2/S3 Sliders, Alternating Motion Macros)
   - Autonomous Mission Control (Start Auto Pick, Emergency Stop, Capabilities Showcase)
4. 🔄 100% Real-time synchronization between Robot LCD and Web Cockpit.
"""

import argparse
import json
import os
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

# Add parent directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.camera.v4l2_cam import V4L2CameraAdapter
from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.adapters.vision.mock_vision import MockVisionAdapter
from src.adapters.vision.tracker_adapter import VisualServoingTracker
from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter, DEFAULT_TARGET_CLASSES
from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter, DEFAULT_PICKABLE_CLASSES
from src.core.context import RobotContext
from src.core.state_machine import AutonomousStateMachine, RobotState
from src.manipulation.arm_controller import ArmController
from src.navigation.visual_servoing import VisualServoingController
from src.activities.activity_manager import ActivityManager
from src.planning.vlm_motion_planner import VLMMotionPlanner, VLMMotionPlan
import scripts.wifi_manager as wifi_manager

VISION_CONFIG_PATH = os.path.join(BASE_DIR, "config/vision_config.yaml")
ROBOT_CONFIG_PATH = os.path.join(BASE_DIR, "config/robot_config.yaml")
ARM_STATE_PATH = os.path.join(BASE_DIR, "data/arm_state.json")
LOGO_PATH = os.path.join(BASE_DIR, "company_asset/Erovoutika-Light-Logo-1.webp")
SSL_CERT_PATH = os.path.join(BASE_DIR, "config/ssl/cert.pem")
SSL_KEY_PATH = os.path.join(BASE_DIR, "config/ssl/key.pem")

_init_motors = {}
_init_servos = {}
if os.path.exists(ROBOT_CONFIG_PATH):
    try:
        with open(ROBOT_CONFIG_PATH, "r") as _f:
            _rcfg = yaml.safe_load(_f) or {}
            _init_motors = _rcfg.get("motors", {})
            _init_servos = _rcfg.get("servos", {})
    except Exception:
        pass

vlm_auto_advisory = True
vlm_planner = VLMMotionPlanner(
    min_overcoming_pwm=int(_init_motors.get("min_overcoming_pwm", 240)),
    base_speed=int(_init_motors.get("base_speed", 235)),
    turn_speed=int(_init_motors.get("turn_speed", 235)),
    nudge_pwm=int(_init_motors.get("nudge_pwm", 250)),
    nudge_default_ms=int(_init_motors.get("nudge_default_ms", 250)),
    max_speed=int(_init_motors.get("max_speed", 255)),
    align_tol_x=20,
    servos_cfg=_init_servos
)

_SERVER_START_TIME = time.time()

# Shared global state
camera = None
comm = None
arm = None
state_lock = threading.Lock()
is_mock_comm = False

# Feed Destination: "lcd" (default on robot LCD) or "web" (stream to browser)
feed_destination = "lcd"

# Autonomous State Machine & Vision
state_machine = None
robot_context = None
sm_running = False
sm_thread = None
showcase_running = False

vision = None
last_detection = None
vision_running = False
vision_thread = None

# Autonomous Activities Subsystem
activity_manager = ActivityManager()

vlm_info = {
    "object": "NONE",
    "category": "NONE",
    "color": "NONE",
    "color_hex": "#888888",
    "backend": "PyTorch",
    "confidence": 0.0,
    "pickable": False,
    "aligned": False,
    "dx": 0,
    "dy": 0,
    "cx": 0,
    "cy": 0,
    "bbox": [],
    "latency_ms": 0.0,
    "fps": 0.0,
    "details": "Initializing YOLOE / LiteRT..."
}

yoloe_state = {
    "model_path": "models/yolo11s.tflite",
    "imgsz": 320,
    "confidence_threshold": 0.25,
    "enforce_limitations": False,
    "num_threads": 4,
    "target_classes": "object, item, bottle, can, cup, box, package, plastic, paper, metal, trash, toy, tool, phone, ball, container",
    "pickable_classes": "bottle, cup, bowl, banana, apple, orange, sports ball, book, cell phone, remote, scissors, teddy bear, toothbrush, fork, knife, spoon, vase"
}

# Calibration & Teleop State
current_spot = {"center_x": 324, "center_y": 247, "box_width": 256, "box_height": 231}
motor_state = {
    "base_speed": int(_init_motors.get("base_speed", 235)),
    "turn_speed": int(_init_motors.get("turn_speed", 235)),
    "min_overcoming_pwm": int(_init_motors.get("min_overcoming_pwm", 240)),
    "nudge_pwm": int(_init_motors.get("nudge_pwm", 250)),
    "nudge_default_ms": int(_init_motors.get("nudge_default_ms", 250)),
    "trim_offset": int(_init_motors.get("trim_offset", 6)),
    "max_speed": int(_init_motors.get("max_speed", 255)),
    "swap_left_right": bool(_init_motors.get("swap_left_right", True))
}
servo_state = {
    "live_s1": 93,
    "live_s2": 45,
    "live_s3": 110,
    "s1_min": 60, "s1_max": 170, "s1_stow": 93, "s1_down": 170, "s1_center": 93,
    "s2_min": 0, "s2_max": 45, "s2_stow": 45, "s2_down": 0, "s2_center": 45,
    "s3_close": 70, "s3_open": 110, "s3_center": 110,
    "s3_min": 70, "s3_max": 110
}

activities_state = {
    "motion_enabled": True,
    # Person Follower
    "pf_dist": 45,
    "pf_speed": 210,
    # Color Tracking
    "ct_color": "Red",
    "ct_area": 8,
    "ct_speed": 210,
    # Generic Object Tracking
    "ot_target": "bottle",
    "ot_dist": 35,
    "ot_speed": 210,
    # Color Track + Classify
    "ctc_color": "Red",
    "ctc_area": 8,
    "ctc_speed": 210,
    # Object Sizing
    "os_target": "any",
    "os_dist": 35,
    "os_speed": 210,
    # Obstacle Avoidance
    "oa_thresh": 22,
    "oa_cruise": 205,
    "oa_turn": 225,
}


def load_all_configs():
    global current_spot, motor_state, servo_state, yoloe_state
    global current_spot, motor_state, servo_state, yoloe_state, activities_state
    # 1. Vision config
    if os.path.exists(VISION_CONFIG_PATH):
        try:
            with open(VISION_CONFIG_PATH, "r") as f:
                vcfg = yaml.safe_load(f) or {}
                spot = vcfg.get("grasp_sweet_spot", {})
                current_spot["center_x"] = spot.get("center_x", current_spot["center_x"])
                current_spot["center_y"] = spot.get("center_y", current_spot["center_y"])
                current_spot["box_width"] = spot.get("box_width", current_spot["box_width"])
                current_spot["box_height"] = spot.get("box_height", current_spot["box_height"])

                y = vcfg.get("yoloe", {})
                if "model_path" in y:
                    yoloe_state["model_path"] = y["model_path"]
                if "imgsz" in y:
                    yoloe_state["imgsz"] = int(y["imgsz"])
                if "confidence_threshold" in y:
                    yoloe_state["confidence_threshold"] = float(y["confidence_threshold"])
                if "enforce_limitations" in y:
                    yoloe_state["enforce_limitations"] = bool(y["enforce_limitations"])
                if "target_classes" in y:
                    tc = y["target_classes"]
                if "target_classes" in y or "pickable_classes" in y:
                    tc = y.get("pickable_classes", y.get("target_classes"))
                    if isinstance(tc, list):
                        yoloe_state["target_classes"] = ", ".join(tc)
                        yoloe_state["pickable_classes"] = ", ".join(tc)
                    else:
                        yoloe_state["target_classes"] = str(tc)
                        yoloe_state["pickable_classes"] = str(tc)

                # Activities settings
                act = vcfg.get("activities", {})
                if "person_follower" in act:
                    pf = act["person_follower"]
                    activities_state["pf_dist"] = int(pf.get("target_dist", activities_state["pf_dist"]))
                    activities_state["pf_speed"] = int(pf.get("follow_speed", activities_state["pf_speed"]))
                if "color_tracking" in act:
                    ct = act["color_tracking"]
                    activities_state["ct_color"] = str(ct.get("color", activities_state["ct_color"]))
                    activities_state["ct_area"] = int(ct.get("target_area", activities_state["ct_area"]))
                    activities_state["ct_speed"] = int(ct.get("follow_speed", activities_state["ct_speed"]))
                if "object_tracking" in act:
                    ot = act["object_tracking"]
                    activities_state["ot_target"] = str(ot.get("target_object", activities_state["ot_target"]))
                    activities_state["ot_dist"] = int(ot.get("target_dist", activities_state["ot_dist"]))
                    activities_state["ot_speed"] = int(ot.get("follow_speed", activities_state["ot_speed"]))
                if "color_track_and_classify" in act:
                    ctc = act["color_track_and_classify"]
                    activities_state["ctc_color"] = str(ctc.get("color", activities_state["ctc_color"]))
                    activities_state["ctc_area"] = int(ctc.get("target_area", activities_state["ctc_area"]))
                    activities_state["ctc_speed"] = int(ctc.get("follow_speed", activities_state["ctc_speed"]))
                if "object_sizing" in act:
                    os_act = act["object_sizing"]
                    activities_state["os_target"] = str(os_act.get("filter_object", activities_state["os_target"]))
                    activities_state["os_dist"] = int(os_act.get("inspection_dist", activities_state["os_dist"]))
                    activities_state["os_speed"] = int(os_act.get("align_speed", activities_state["os_speed"]))
                if "obstacle_avoidance" in act:
                    oa = act["obstacle_avoidance"]
                    activities_state["oa_thresh"] = int(oa.get("score_threshold", activities_state["oa_thresh"]))
                    activities_state["oa_cruise"] = int(oa.get("cruise_speed", activities_state["oa_cruise"]))
                    activities_state["oa_turn"] = int(oa.get("turn_speed", activities_state["oa_turn"]))
                activities_state["motion_enabled"] = bool(act.get("motion_enabled", activities_state.get("motion_enabled", True)))

                if activity_manager:
                    with activity_manager.lock:
                        activity_manager.config["motion_enabled"] = bool(activities_state.get("motion_enabled", True))
                        activity_manager.config["target_color"] = str(activities_state["ct_color"])
                        activity_manager.config["target_object"] = str(activities_state["ot_target"])
                        activity_manager.config["sizing_target_object"] = str(activities_state["os_target"])
                        activity_manager.config["follow_speed"] = int(activities_state["pf_speed"])
                        activity_manager.config["obstacle_cruise_speed"] = int(activities_state["oa_cruise"])
                        activity_manager.config["obstacle_turn_speed"] = int(activities_state["oa_turn"])
                        activity_manager.config["obstacle_threshold"] = float(activities_state["oa_thresh"]) / 100.0
                        activity_manager.config["person_target_height_ratio"] = float(activities_state["pf_dist"]) / 100.0
                        activity_manager.config["object_target_height_ratio"] = float(activities_state["ot_dist"]) / 100.0
                        activity_manager.config["sizing_target_height_ratio"] = float(activities_state["os_dist"]) / 100.0
                        activity_manager.config["color_target_area_ratio"] = float(activities_state["ct_area"]) / 100.0
        except Exception as e:
            print(f"[Web] Error loading vision config: {e}")

    # 2. Robot config
    if os.path.exists(ROBOT_CONFIG_PATH):
        try:
            with open(ROBOT_CONFIG_PATH, "r") as f:
                rcfg = yaml.safe_load(f) or {}
                m = rcfg.get("motors", {})
                for k in motor_state:
                    if k in m:
                        motor_state[k] = m[k]

                servos = rcfg.get("servos", {})
                s1 = servos.get("servo1_shoulder", {})
                s2 = servos.get("servo2_elbow", {})
                s3 = servos.get("servo3_gripper", {})

                servo_state["s1_min"] = s1.get("min_angle", 60)
                servo_state["s1_max"] = s1.get("max_angle", 170)
                servo_state["s1_stow"] = s1.get("stow_angle", 93)
                servo_state["s1_down"] = s1.get("down_angle", 170)
                servo_state["s1_center"] = s1.get("center_angle", 93)

                servo_state["s2_min"] = s2.get("min_angle", 0)
                servo_state["s2_max"] = s2.get("max_angle", 45)
                servo_state["s2_stow"] = s2.get("stow_angle", 45)
                servo_state["s2_down"] = s2.get("down_angle", 0)
                servo_state["s2_center"] = s2.get("center_angle", 45)

                s3_close_val = s3.get("close_angle", 40)
                s3_open_val = s3.get("open_angle", 170)
                servo_state["s3_close"] = int(s3_close_val)
                servo_state["s3_open"] = int(s3_open_val)
                servo_state["s3_center"] = int(s3.get("center_angle", 90))
                servo_state["s3_min"] = int(s3.get("min_angle", min(servo_state["s3_close"], servo_state["s3_open"])))
                servo_state["s3_max"] = int(s3.get("max_angle", max(servo_state["s3_close"], servo_state["s3_open"])))

                if arm:
                    servo_state["live_s1"] = arm.cur_s1
                    servo_state["live_s2"] = arm.cur_s2
                    servo_state["live_s3"] = arm.cur_s3
                elif os.path.exists(ARM_STATE_PATH):
                    try:
                        with open(ARM_STATE_PATH, "r") as f:
                            st = json.load(f)
                            servo_state["live_s1"] = int(st.get("s1", servo_state["s1_center"]))
                            servo_state["live_s2"] = int(st.get("s2", servo_state["s2_center"]))
                            servo_state["live_s3"] = int(st.get("s3", servo_state["s3_close"]))
                    except Exception:
                        servo_state["live_s1"] = servo_state["s1_center"]
                        servo_state["live_s2"] = servo_state["s2_center"]
                        servo_state["live_s3"] = servo_state["s3_close"]
                else:
                    servo_state["live_s1"] = servo_state["s1_center"]
                    servo_state["live_s2"] = servo_state["s2_center"]
                    servo_state["live_s3"] = servo_state["s3_close"]

                s3_live_min = min(servo_state["s3_close"], servo_state["s3_open"])
                s3_live_max = max(servo_state["s3_close"], servo_state["s3_open"], servo_state["s3_max"])
                servo_state["live_s1"] = max(servo_state["s1_min"], min(servo_state["s1_max"], servo_state["live_s1"]))
                servo_state["live_s2"] = max(servo_state["s2_min"], min(servo_state["s2_max"], servo_state["live_s2"]))
                servo_state["live_s3"] = max(s3_live_min, min(s3_live_max, servo_state["live_s3"]))
        except Exception as e:
            print(f"[Web] Error loading robot config: {e}")

    if vlm_planner:
        vlm_planner.reload_config(motor_state, {"grasp_sweet_spot": current_spot}, servo_state)

    if os.path.exists(ROBOT_CONFIG_PATH):
        try:
            _robot_config_mtime = os.path.getmtime(ROBOT_CONFIG_PATH)
        except Exception:
            pass


_robot_config_mtime = 0.0


def get_servo_cal_dict() -> dict:
    """Returns standardized dictionary of current servo calibration limits and angles."""
    s3_close = int(servo_state.get("s3_close", 40))
    s3_open = int(servo_state.get("s3_open", 170))
    s3_max = int(servo_state.get("s3_max", 170))
    s3_live_min = min(s3_close, s3_open)
    s3_live_max = max(s3_close, s3_open, s3_max)
    return {
        "s1_min": int(servo_state.get("s1_min", 60)),
        "s1_max": int(servo_state.get("s1_max", 170)),
        "s1_stow": int(servo_state.get("s1_stow", 93)),
        "s1_down": int(servo_state.get("s1_down", 170)),
        "s1_center": int(servo_state.get("s1_center", 93)),
        "s2_min": int(servo_state.get("s2_min", 0)),
        "s2_max": int(servo_state.get("s2_max", 45)),
        "s2_stow": int(servo_state.get("s2_stow", 45)),
        "s2_down": int(servo_state.get("s2_down", 0)),
        "s2_center": int(servo_state.get("s2_center", 45)),
        "s3_min": int(servo_state.get("s3_min", s3_live_min)),
        "s3_max": s3_max,
        "s3_open": s3_open,
        "s3_close": s3_close,
        "s3_center": int(servo_state.get("s3_center", 90)),
        "s3_live_min": s3_live_min,
        "s3_live_max": s3_live_max,
    }


def check_reload_robot_config(force: bool = False) -> bool:
    """Checks mtime of robot_config.yaml and reloads servo/motor parameters if changed or force=True."""
    global _robot_config_mtime, servo_state, motor_state, arm
    if not os.path.exists(ROBOT_CONFIG_PATH):
        return False
    try:
        mtime = os.path.getmtime(ROBOT_CONFIG_PATH)
        if not force and mtime <= _robot_config_mtime:
            return False

        with open(ROBOT_CONFIG_PATH, "r") as f:
            rcfg = yaml.safe_load(f) or {}

        servos = rcfg.get("servos", {})
        s1 = servos.get("servo1_shoulder", {})
        s2 = servos.get("servo2_elbow", {})
        s3 = servos.get("servo3_gripper", {})

        s1_down = s1.get("down_angle", 170)
        s1_stow = s1.get("stow_angle", s1.get("up_angle", 93))
        servo_state["s1_stow"] = int(s1_stow)
        servo_state["s1_down"] = int(s1_down)
        servo_state["s1_center"] = int(s1.get("center_angle", 93))
        servo_state["s1_min"] = int(s1.get("min_angle", min(60, s1_down, s1_stow)))
        servo_state["s1_max"] = int(s1.get("max_angle", max(170, s1_down, s1_stow)))

        s2_down = s2.get("down_angle", 0)
        s2_stow = s2.get("stow_angle", s2.get("up_angle", 45))
        servo_state["s2_stow"] = int(s2_stow)
        servo_state["s2_down"] = int(s2_down)
        servo_state["s2_center"] = int(s2.get("center_angle", 45))
        servo_state["s2_min"] = int(s2.get("min_angle", min(0, s2_down, s2_stow)))
        servo_state["s2_max"] = int(s2.get("max_angle", max(45, s2_down, s2_stow)))

        s3_close = s3.get("close_angle", 40)
        s3_open = s3.get("open_angle", 170)
        s3_cfg_max = s3.get("max_angle", max(s3_close, s3_open))
        s3_cfg_min = s3.get("min_angle", min(s3_close, s3_open))
        servo_state["s3_close"] = int(s3_close)
        servo_state["s3_open"] = int(s3_open)
        servo_state["s3_center"] = int(s3.get("center_angle", 90))
        servo_state["s3_min"] = int(min(s3_close, s3_open, s3_cfg_min))
        servo_state["s3_max"] = int(max(s3_close, s3_open, s3_cfg_max))

        s3_live_min = min(servo_state["s3_close"], servo_state["s3_open"])
        s3_live_max = max(servo_state["s3_close"], servo_state["s3_open"], servo_state["s3_max"])
        if "live_s1" in servo_state:
            servo_state["live_s1"] = max(servo_state["s1_min"], min(servo_state["s1_max"], int(servo_state["live_s1"])))
        if "live_s2" in servo_state:
            servo_state["live_s2"] = max(servo_state["s2_min"], min(servo_state["s2_max"], int(servo_state["live_s2"])))
        if "live_s3" in servo_state:
            servo_state["live_s3"] = max(s3_live_min, min(s3_live_max, int(servo_state["live_s3"])))

        motors = rcfg.get("motors", {})
        for k in motor_state:
            if k in motors:
                motor_state[k] = motors[k]

        if arm:
            arm.reload_config(force_disk=True)

        if vlm_planner:
            vlm_planner.reload_config(motor_state, {"grasp_sweet_spot": current_spot}, servo_state)

        _robot_config_mtime = mtime
        return True
    except Exception as e:
        print(f"[Web] Error checking/reloading robot config: {e}")
        return False


def save_vision_config():
    with open(VISION_CONFIG_PATH, "r") as f:
        vcfg = yaml.safe_load(f) or {}
    vcfg.setdefault("grasp_sweet_spot", {})
    vcfg["grasp_sweet_spot"]["center_x"] = int(current_spot["center_x"])
    vcfg["grasp_sweet_spot"]["center_y"] = int(current_spot["center_y"])
    vcfg["grasp_sweet_spot"]["box_width"] = int(current_spot["box_width"])
    vcfg["grasp_sweet_spot"]["box_height"] = int(current_spot["box_height"])
    with open(VISION_CONFIG_PATH, "w") as f:
        yaml.dump(vcfg, f, default_flow_style=False)
    print(f"[Web] Saved camera sweet spot to {VISION_CONFIG_PATH}")


def save_yoloe_config(settings: dict):
    with open(VISION_CONFIG_PATH, "r") as f:
        vcfg = yaml.safe_load(f) or {}
    vcfg.setdefault("yoloe", {})
    if "model_path" in settings and settings["model_path"]:
        vcfg["yoloe"]["model_path"] = str(settings["model_path"])
        yoloe_state["model_path"] = str(settings["model_path"])
    if "confidence_threshold" in settings:
        conf = float(settings["confidence_threshold"])
        vcfg["yoloe"]["confidence_threshold"] = conf
        yoloe_state["confidence_threshold"] = conf
    if "imgsz" in settings:
        sz = int(settings["imgsz"])
        vcfg["yoloe"]["imgsz"] = sz
        yoloe_state["imgsz"] = sz
    if "enforce_limitations" in settings:
        enf = bool(settings["enforce_limitations"])
        vcfg["yoloe"]["enforce_limitations"] = enf
        yoloe_state["enforce_limitations"] = enf
    if "target_classes" in settings:
        tc = settings["target_classes"]
    # Accept both pickable_classes (new) and target_classes (legacy)
    tc_key = "pickable_classes" if "pickable_classes" in settings else "target_classes" if "target_classes" in settings else None
    if tc_key:
        tc = settings[tc_key]
        if isinstance(tc, str):
            classes = [c.strip() for c in tc.split(",") if c.strip()]
        else:
            classes = list(tc)
        vcfg["yoloe"]["target_classes"] = classes
        yoloe_state["target_classes"] = ", ".join(classes)
        vcfg["yoloe"]["pickable_classes"] = classes
        # Remove legacy key if present
        vcfg["yoloe"].pop("target_classes", None)
        yoloe_state["pickable_classes"] = ", ".join(classes)
    with open(VISION_CONFIG_PATH, "w") as f:
        yaml.dump(vcfg, f, default_flow_style=False)
    print(f"[Web] Saved YOLOE configuration to {VISION_CONFIG_PATH}")


def save_activities_config(settings: dict = None):
    global activities_state
    if settings:
        for k, v in settings.items():
            if k in activities_state:
                if k in ("ct_color", "ot_target", "ctc_color", "os_target"):
                    activities_state[k] = str(v)
                elif k == "motion_enabled":
                    activities_state[k] = bool(v)
                else:
                    try:
                        activities_state[k] = int(v)
                    except (ValueError, TypeError):
                        pass
    with open(VISION_CONFIG_PATH, "r") as f:
        vcfg = yaml.safe_load(f) or {}
    vcfg["activities"] = {
        "motion_enabled": bool(activities_state.get("motion_enabled", True)),
        "person_follower": {
            "target_dist": int(activities_state["pf_dist"]),
            "follow_speed": int(activities_state["pf_speed"]),
        },
        "color_tracking": {
            "color": str(activities_state["ct_color"]),
            "target_area": int(activities_state["ct_area"]),
            "follow_speed": int(activities_state["ct_speed"]),
        },
        "object_tracking": {
            "target_object": str(activities_state["ot_target"]),
            "target_dist": int(activities_state["ot_dist"]),
            "follow_speed": int(activities_state["ot_speed"]),
        },
        "color_track_and_classify": {
            "color": str(activities_state["ctc_color"]),
            "target_area": int(activities_state["ctc_area"]),
            "follow_speed": int(activities_state["ctc_speed"]),
        },
        "object_sizing": {
            "filter_object": str(activities_state["os_target"]),
            "inspection_dist": int(activities_state["os_dist"]),
            "align_speed": int(activities_state["os_speed"]),
        },
        "obstacle_avoidance": {
            "score_threshold": int(activities_state["oa_thresh"]),
            "cruise_speed": int(activities_state["oa_cruise"]),
            "turn_speed": int(activities_state["oa_turn"]),
        },
    }
    with open(VISION_CONFIG_PATH, "w") as f:
        yaml.dump(vcfg, f, default_flow_style=False)
    print(f"[Web] Saved activity settings to {VISION_CONFIG_PATH}")

    if activity_manager:
        with activity_manager.lock:
            activity_manager.config["motion_enabled"] = bool(activities_state.get("motion_enabled", True))
            activity_manager.config["target_color"] = str(activities_state["ct_color"])
            activity_manager.config["target_object"] = str(activities_state["ot_target"])
            activity_manager.config["sizing_target_object"] = str(activities_state["os_target"])
            activity_manager.config["follow_speed"] = int(activities_state["pf_speed"])
            activity_manager.config["obstacle_cruise_speed"] = int(activities_state["oa_cruise"])
            activity_manager.config["obstacle_turn_speed"] = int(activities_state["oa_turn"])
            activity_manager.config["obstacle_threshold"] = float(activities_state["oa_thresh"]) / 100.0
            activity_manager.config["person_target_height_ratio"] = float(activities_state["pf_dist"]) / 100.0
            activity_manager.config["object_target_height_ratio"] = float(activities_state["ot_dist"]) / 100.0
            activity_manager.config["sizing_target_height_ratio"] = float(activities_state["os_dist"]) / 100.0
            activity_manager.config["color_target_area_ratio"] = float(activities_state["ct_area"]) / 100.0


def apply_yoloe_config(settings: dict, persist: bool = False):
    global vision
    with state_lock:
        if "confidence_threshold" in settings and vision:
            vision.confidence_threshold = float(settings["confidence_threshold"])
            yoloe_state["confidence_threshold"] = vision.confidence_threshold
        if "imgsz" in settings and vision:
            vision.imgsz = int(settings["imgsz"])
            yoloe_state["imgsz"] = vision.imgsz
        if "enforce_limitations" in settings and vision:
            enf = bool(settings["enforce_limitations"])
            vision.enforce_limitations = enf
            yoloe_state["enforce_limitations"] = enf
        if "target_classes" in settings and vision:
            tc = settings["target_classes"]
            if isinstance(tc, str):
                classes = [c.strip() for c in tc.split(",") if c.strip()]
            else:
                classes = list(tc)
            vision.set_target_classes(classes)
            yoloe_state["target_classes"] = ", ".join(classes)
        if "target_classes" in settings or "pickable_classes" in settings:
            if vision:
                tc_key = "pickable_classes" if "pickable_classes" in settings else "target_classes"
                tc = settings[tc_key]
                if isinstance(tc, str):
                    classes = [c.strip() for c in tc.split(",") if c.strip()]
                else:
                    classes = list(tc)
                vision.set_pickable_classes(classes)
                yoloe_state["pickable_classes"] = ", ".join(classes)
        if "model_path" in settings and vision:
            new_path = str(settings["model_path"])
            abs_new_path = os.path.abspath(new_path) if os.path.isabs(new_path) else os.path.abspath(os.path.join(BASE_DIR, new_path))
            cur_path = getattr(vision, "model_path", "")
            abs_cur_path = os.path.abspath(cur_path) if cur_path else ""
            if abs_new_path != abs_cur_path:
                try:
                    conf = getattr(vision, "confidence_threshold", 0.25)
                    sz = getattr(vision, "imgsz", 320)
                    classes = getattr(vision, "target_classes", DEFAULT_TARGET_CLASSES)
                    classes = getattr(vision, "pickable_classes", DEFAULT_PICKABLE_CLASSES)
                    enf = yoloe_state.get("enforce_limitations", False)
                    nth = yoloe_state.get("num_threads", 4)
                    new_vision = YOLOEVisionAdapter(
                        model_path=abs_new_path,
                        confidence_threshold=conf,
                        imgsz=sz,
                        target_classes=classes,
                        pickable_classes=classes,
                        enforce_limitations=enf,
                        num_threads=nth
                    )
                    if new_vision.model_loaded:
                        vision = new_vision
                        yoloe_state["model_path"] = new_path
                        if state_machine:
                            state_machine.vision = vision
                except Exception as e:
                    print(f"[Web] Error reloading model {new_path}: {e}")
        if persist:
            save_yoloe_config(settings)


def save_robot_motors():
    with open(ROBOT_CONFIG_PATH, "r") as f:
        rcfg = yaml.safe_load(f) or {}
    rcfg.setdefault("motors", {})
    for k, v in motor_state.items():
        rcfg["motors"][k] = bool(v) if k == "swap_left_right" else int(v)
    with open(ROBOT_CONFIG_PATH, "w") as f:
        yaml.dump(rcfg, f, default_flow_style=False)
    print(f"[Web] Saved motor parameters to {ROBOT_CONFIG_PATH}")
    if vlm_planner:
        vlm_planner.reload_config(motor_state, {"grasp_sweet_spot": current_spot}, servo_state)


def save_robot_servos():
    global _robot_config_mtime
    rcfg = {}
    if os.path.exists(ROBOT_CONFIG_PATH):
        try:
            with open(ROBOT_CONFIG_PATH, "r") as f:
                rcfg = yaml.safe_load(f) or {}
        except Exception:
            rcfg = {}
    rcfg.setdefault("servos", {})
    s1 = rcfg["servos"].setdefault("servo1_shoulder", {})
    s2 = rcfg["servos"].setdefault("servo2_elbow", {})
    s3 = rcfg["servos"].setdefault("servo3_gripper", {})

    s1_stow = int(servo_state["s1_stow"])
    s1_down = int(servo_state["s1_down"])
    s1_center = int(servo_state.get("s1_center", s1_stow))
    s1_min = min(int(servo_state.get("s1_min", 60)), s1_stow, s1_down)
    s1_max = max(int(servo_state.get("s1_max", 170)), s1_stow, s1_down)

    s1["min_angle"] = s1_min
    s1["max_angle"] = s1_max
    s1["stow_angle"] = s1_stow
    s1["up_angle"] = s1_stow
    s1["down_angle"] = s1_down
    s1["center_angle"] = s1_center

    s2_stow = int(servo_state["s2_stow"])
    s2_down = int(servo_state["s2_down"])
    s2_center = int(servo_state.get("s2_center", s2_stow))
    s2_min = min(int(servo_state.get("s2_min", 0)), s2_stow, s2_down)
    s2_max = max(int(servo_state.get("s2_max", 45)), s2_stow, s2_down)

    s2["min_angle"] = s2_min
    s2["max_angle"] = s2_max
    s2["stow_angle"] = s2_stow
    s2["up_angle"] = s2_stow
    s2["down_angle"] = s2_down
    s2["center_angle"] = s2_center

    s3_open = int(servo_state["s3_open"])
    s3_close = int(servo_state["s3_close"])
    s3_center = int(servo_state.get("s3_center", s3_open))
    s3_min = min(s3_open, s3_close)
    s3_max = max(s3_open, s3_close, int(servo_state.get("s3_max", max(s3_open, s3_close))))

    s3["min_angle"] = s3_min
    s3["max_angle"] = s3_max
    s3["close_angle"] = s3_close
    s3["open_angle"] = s3_open
    s3["center_angle"] = s3_center

    tmp_path = ROBOT_CONFIG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        yaml.dump(rcfg, f, default_flow_style=False)
    os.replace(tmp_path, ROBOT_CONFIG_PATH)

    try:
        _robot_config_mtime = os.path.getmtime(ROBOT_CONFIG_PATH)
    except Exception:
        pass

    if arm:
        arm.reload_config(force_disk=True)

    print(f"[Web] Saved servo calibration to {ROBOT_CONFIG_PATH}")
    if vlm_planner:
        vlm_planner.reload_config(motor_state, {"grasp_sweet_spot": current_spot}, servo_state)


def init_autonomous_system():
    global state_machine, robot_context, vision
    with state_lock:
        rcfg = {}
        vcfg = {}
        if os.path.exists(ROBOT_CONFIG_PATH):
            with open(ROBOT_CONFIG_PATH, "r") as f:
                rcfg = yaml.safe_load(f) or {}
        if os.path.exists(VISION_CONFIG_PATH):
            with open(VISION_CONFIG_PATH, "r") as f:
                vcfg = yaml.safe_load(f) or {}

        robot_context = RobotContext(config={"robot": rcfg, "vision": vcfg})
        spot = vcfg.get("grasp_sweet_spot", {})
        robot_context.set_sweet_spot(
            x=spot.get("center_x", current_spot["center_x"]),
            y=spot.get("center_y", current_spot["center_y"]),
            w=spot.get("box_width", current_spot["box_width"]),
            h=spot.get("box_height", current_spot["box_height"])
        )

        motors_cfg = rcfg.get("motors", {})
        vs_cfg = rcfg.get("visual_servoing", {})
        servoing_controller = VisualServoingController(
            center_x=robot_context.sweet_spot_x,
            grasp_y=robot_context.sweet_spot_y,
            tol_x=spot.get("align_tolerance_x", 20),
            tol_y=spot.get("align_tolerance_y", 25),
            base_speed=motors_cfg.get("base_speed", 225),
            turn_speed=motors_cfg.get("turn_speed", 240),
            min_overcoming_pwm=motors_cfg.get("min_overcoming_pwm", 230),
            pulse_threshold_px=vs_cfg.get("pulse_threshold_px", 45),
            nudge_duration_ms=motors_cfg.get("nudge_default_ms", 250),
            nudge_pwm=motors_cfg.get("nudge_pwm", 245),
            trim_offset=motors_cfg.get("trim_offset", 0)
        )

        tracker = VisualServoingTracker()

        # Initialize YOLOE Vision Adapter (fallback to MockVisionAdapter if unavailable)
        yoloe_cfg = vcfg.get("yoloe", {})
        model_path = yoloe_cfg.get("model_path", "models/yolo11s.tflite")
        conf_thresh = yoloe_cfg.get("confidence_threshold", 0.25)
        imgsz = yoloe_cfg.get("imgsz", 320)
        target_classes = yoloe_cfg.get("target_classes", None)
        target_classes = yoloe_cfg.get("pickable_classes", yoloe_cfg.get("target_classes", None))
        enforce_limits = yoloe_cfg.get("enforce_limitations", False)
        num_threads = yoloe_cfg.get("num_threads", 4)
        try:
            print(f"[Web] Initializing YOLOEVisionAdapter ({model_path})...")
            vision = YOLOEVisionAdapter(
                model_path=model_path,
                confidence_threshold=conf_thresh,
                imgsz=imgsz,
                target_classes=target_classes,
                pickable_classes=target_classes,
                enforce_limitations=enforce_limits,
                num_threads=num_threads
            )
            if not vision.model_loaded:
                print("[Web] Notice: YOLOE model not ready, falling back to MockVisionAdapter.")
                vision = MockVisionAdapter()
        except Exception as e:
            print(f"[Web] Notice: YOLOE init exception ({e}), falling back to MockVisionAdapter.")
            vision = MockVisionAdapter()

        state_machine = AutonomousStateMachine(
            context=robot_context,
            camera=camera,
            comm=comm,
            vision=vision,
            tracker=tracker,
            servoing=servoing_controller,
            arm=arm
        )

        if activity_manager:
            activity_manager.set_dependencies(camera=camera, comm=comm, vision=vision, vlm_planner=vlm_planner, arm=arm)


def _vision_worker():
    global vision_running, last_detection, vlm_info
    while vision_running:
        if activity_manager and activity_manager.running:
            # Yield camera and inference completely to active activity
            time.sleep(0.08)
            continue

        if vision and camera and camera.is_opened():
            try:
                frame = camera.get_frame()
                if frame is not None:
                    t_start = time.time()
                    res = vision.categorize(frame)
                    dt = time.time() - t_start

                    # Sweet spot alignment calculation
                    sx = current_spot.get("center_x", 324)
                    sy = current_spot.get("center_y", 247)
                    sw = current_spot.get("box_width", 256)
                    sh = current_spot.get("box_height", 231)

                    if res and res.detected:
                        last_detection = res
                        cx, cy = res.center
                        in_x = (sx - sw // 2) <= cx <= (sx + sw // 2)
                        in_y = (sy - sh // 2) <= cy <= (sy + sh // 2)
                        aligned = in_x and in_y
                        dx = cx - sx
                        dy = sy - cy

                        color_name = getattr(res, "material_color", "Unknown")
                        vlm_info["color"] = color_name
                        vlm_info["color_hex"] = "#888888"
                        if getattr(res, "raw_response", None):
                            try:
                                import json
                                j = json.loads(res.raw_response)
                                vlm_info["color_hex"] = j.get("color_hex", "#888888")
                            except Exception:
                                pass
                        vlm_info["backend"] = getattr(vision, "backend_type", "PyTorch") if vision else "PyTorch"

                        vlm_info["object"] = res.category.upper()
                        vlm_info["category"] = res.category.upper()
                        vlm_info["confidence"] = round(float(res.confidence), 3)
                        vlm_info["pickable"] = bool(res.pickable)
                        vlm_info["aligned"] = bool(aligned)
                        vlm_info["dx"] = int(dx)
                        vlm_info["dy"] = int(dy)
                        vlm_info["cx"] = int(cx)
                        vlm_info["cy"] = int(cy)
                        vlm_info["bbox"] = list(res.bounding_box) if res.bounding_box else []
                        vlm_info["latency_ms"] = round(dt * 1000, 1)
                        vlm_info["fps"] = round(1.0 / dt, 1) if dt > 0 else 0.0
                        align_str = "ALIGNED IN SWEET SPOT" if aligned else f"dx={dx:+d}px, dy={dy:+d}px"
                        color_str = f" [{color_name}]" if color_name and color_name != "Unknown" else ""
                        vlm_info["details"] = f"{res.category.upper()}{color_str} ({res.confidence*100:.0f}%) | Pickable: {'YES' if res.pickable else 'NO'} | {align_str}"

                        if vlm_planner and (vlm_auto_advisory or vlm_planner.get_last_plan() is None):
                            vlm_planner.plan_movement(res, dict(current_spot), use_vlm_llm=False)
                    else:
                        last_detection = None
                        vlm_info["object"] = "NONE"
                        vlm_info["category"] = "NONE"
                        vlm_info["color"] = "NONE"
                        vlm_info["color_hex"] = "#888888"
                        vlm_info["backend"] = getattr(vision, "backend_type", "PyTorch") if vision else "PyTorch"
                        vlm_info["confidence"] = 0.0
                        vlm_info["pickable"] = False
                        vlm_info["aligned"] = False
                        vlm_info["dx"] = 0
                        vlm_info["dy"] = 0
                        vlm_info["cx"] = 0
                        vlm_info["cy"] = 0
                        vlm_info["bbox"] = []
                        vlm_info["latency_ms"] = round(dt * 1000, 1)
                        vlm_info["fps"] = round(1.0 / dt, 1) if dt > 0 else 0.0
                        vlm_info["details"] = "Scanning for targets..."
            except Exception as e:
                pass
        time.sleep(0.18)


def start_vision_worker():
    global vision_running, vision_thread
    if not vision_running:
        vision_running = True
        vision_thread = threading.Thread(target=_vision_worker, daemon=True)
        vision_thread.start()


def stop_vision_worker():
    global vision_running
    vision_running = False


def _sm_worker():
    global sm_running
    while sm_running:
        if state_machine:
            try:
                state_machine.step()
            except Exception as e:
                print(f"[Web] State machine notice: {e}")
        time.sleep(0.033)


def start_autonomous_mode():
    global sm_running, sm_thread
    if not state_machine:
        init_autonomous_system()
    if state_machine:
        state_machine.start_autonomous_run()
    if not sm_running:
        sm_running = True
        sm_thread = threading.Thread(target=_sm_worker, daemon=True)
        sm_thread.start()


def stop_autonomous_mode():
    global sm_running
    sm_running = False
    if state_machine:
        state_machine.stop()
    if comm:
        comm.send_stop()


def get_robot_state_name():
    if state_machine and hasattr(state_machine, "state"):
        s = state_machine.state
        return s.value if hasattr(s, "value") else str(s)
    return "IDLE"


def trigger_showcase():
    global showcase_running
    if showcase_running:
        return
    showcase_running = True

    def worker():
        global showcase_running
        try:
            if arm:
                print("[Showcase] Running capability demonstration sequence...")
                arm.center()
                time.sleep(0.4)
                arm.open_gripper()
                time.sleep(0.4)
                arm.arm_down()
                time.sleep(0.6)
                arm.close_gripper()
                time.sleep(0.4)
                arm.stow()
                time.sleep(0.6)
                arm.open_gripper()
                time.sleep(0.4)
                arm.center()
                print("[Showcase] Capability demonstration complete.")
        except Exception as e:
            print(f"[Web] Showcase notice: {e}")
        finally:
            showcase_running = False

    threading.Thread(target=worker, daemon=True).start()


def connect_comm(force_mock=False):
    global comm, arm, is_mock_comm
    with state_lock:
        if comm:
            try:
                comm.send_stop()
                comm.disconnect()
            except Exception:
                pass

        if force_mock:
            print("[Web] Forcing MockCommAdapter (Simulated Arduino)")
            comm = MockCommAdapter()
            comm.connect()
            is_mock_comm = True
        else:
            rcfg = {}
            if os.path.exists(ROBOT_CONFIG_PATH):
                with open(ROBOT_CONFIG_PATH, "r") as f:
                    rcfg = yaml.safe_load(f) or {}
            comm_cfg = rcfg.get("communication", {})
            port = comm_cfg.get("serial_port", "/dev/rfcomm0")
            baud = comm_cfg.get("baud_rate", 9600)
            mac = comm_cfg.get("bluetooth_mac", "20:25:08:00:46:FB")
            chan = comm_cfg.get("rfcomm_channel", 1)

            print(f"[Web] Attempting Bluetooth {mac} on {port} @ {baud} baud...")
            try:
                bt_comm = BluetoothSerialAdapter(port=port, baud_rate=baud, bluetooth_mac=mac, rfcomm_channel=chan)
                if bt_comm.connect():
                    comm = bt_comm
                    is_mock_comm = False
                    print(f"[Web] Connected to hardware Arduino on {port}")
                else:
                    print("[Web] Bluetooth connection failed. Falling back to MockCommAdapter.")
                    comm = MockCommAdapter()
                    comm.connect()
                    is_mock_comm = True
            except Exception as e:
                print(f"[Web] Error opening {port}: {e}. Using MockCommAdapter.")
                comm = MockCommAdapter()
                comm.connect()
                is_mock_comm = True

        arm = ArmController(comm)
        servo_state["live_s1"] = arm.cur_s1
        servo_state["live_s2"] = arm.cur_s2
        servo_state["live_s3"] = arm.cur_s3
        if comm:
            comm.send_servos(arm.cur_s1, arm.cur_s2, arm.cur_s3)

    init_autonomous_system()


_bt_watchdog_started = False

def start_bt_watchdog():
    """Background watchdog thread that continuously reconnects Bluetooth if connection was offline at boot or dropped."""
    global _bt_watchdog_started
    if _bt_watchdog_started:
        return
    _bt_watchdog_started = True

    def watchdog_worker():
        while True:
            time.sleep(3.5)
            if is_mock_comm:
                rcfg = {}
                if os.path.exists(ROBOT_CONFIG_PATH):
                    with open(ROBOT_CONFIG_PATH, "r") as f:
                        rcfg = yaml.safe_load(f) or {}
                port = rcfg.get("communication", {}).get("serial_port", "/dev/rfcomm0")
                if os.path.exists(port):
                    print(f"[Bluetooth Watchdog] Device {port} detected, attempting auto-reconnect...")
                    connect_comm(force_mock=False)
                    if not is_mock_comm:
                        print(f"[Bluetooth Watchdog] Successfully established hardware Bluetooth connection to {port}!")

    threading.Thread(target=watchdog_worker, daemon=True, name="BTWatchdogThread").start()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ErovoutikaGrab - Robot Dashboard</title>
  <style>
    :root {
      --bg: #070d1d;
      --card-bg: #111c44;
      --border: #1e3a8a;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --green: #10b981;
      --red: #ef4444;
      --amber: #f59e0b;
      --lcd-bezel-inset-y: 28px;
      --lcd-bezel-inset-x: 36px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Liberation Sans', 'DejaVu Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 12px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
    
    /* =========================================================
       ROBOT LCD VIEW-ONLY HUD STYLES (?device=lcd)
       ========================================================= */
    html.lcd-view-only,
    body.lcd-view-only {
      padding: 0 !important;
      margin: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      overflow: hidden !important;
      background: #000000 !important;
      cursor: none !important;
      user-select: none !important;
    }
    /* Hide all interactive cockpit controls on LCD */
    html.lcd-view-only .cockpit-header,
    html.lcd-view-only .cockpit-mission-bar,
    html.lcd-view-only .tabs,
    html.lcd-view-only .cockpit-tab-content,
    html.lcd-view-only button,
    html.lcd-view-only input,
    body.lcd-view-only .cockpit-header,
    body.lcd-view-only .cockpit-mission-bar,
    body.lcd-view-only .tabs,
    body.lcd-view-only .cockpit-tab-content,
    body.lcd-view-only button,
    body.lcd-view-only input {
      display: none !important;
    }
    html.lcd-view-only #lcd_hud_container,
    body.lcd-view-only #lcd_hud_container {
      display: flex !important;
      width: 100vw !important;
      height: 100vh !important;
      max-width: 100vw !important;
      max-height: 100vh !important;
      flex-direction: column;
      position: absolute;
      top: 0;
      left: 0;
      box-sizing: border-box !important;
      padding: var(--lcd-bezel-inset-y) var(--lcd-bezel-inset-x) !important;
      background: #000000 !important;
      overflow: hidden !important;
    }
    #lcd_hud_container,
    #lcd_hud_container * {
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: normal !important;
      font-variant-numeric: normal !important;
    }
    body:not(.lcd-view-only) #lcd_hud_container {
      display: none !important;
    }
    html:not(.lcd-view-only) body:not(.lcd-view-only) #lcd_hud_container {
      display: none !important;
    }

    /* LCD Mode A: Fullscreen Live Camera HUD */
    #lcd_view_camera {
      position: relative;
      width: 100%;
      height: 100%;
      background: #000;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
      border-radius: 8px;
    }
    #lcd_view_camera img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .lcd-cam-top-bar {
      position: absolute;
      top: 12px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: flex-end;
      align-items: center;
      pointer-events: none;
    }
    .lcd-hud-logo {
      display: none !important;
    }
    .lcd-pill-badge {
      background: rgba(11, 20, 42, 0.85);
      border: 2px solid #3b82f6;
      border-radius: 8px;
      padding: 6px 16px;
      font-size: 16px;
      font-weight: 800;
      color: #60a5fa;
      backdrop-filter: blur(8px);
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: 0.5px !important;
    }
    .lcd-cam-bottom-bar {
      position: absolute;
      bottom: 16px;
      left: 14px;
      right: 14px;
      background: rgba(7, 13, 29, 0.88);
      border: 1px solid #1e3a8a;
      border-radius: 8px;
      padding: 8px 16px;
      display: flex;
      justify-content: space-around;
      align-items: center;
      backdrop-filter: blur(8px);
      pointer-events: none;
    }
    .lcd-hud-stat {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .lcd-hud-stat-lbl {
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: bold;
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: normal !important;
    }
    .lcd-hud-stat-val {
      font-size: 14px;
      font-weight: 600;
      color: #f8fafc;
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: normal !important;
    }

    /* LCD Mode B: Centered Logo + Large Character Telemetry Gauge Cluster */
    #lcd_view_gauges {
      width: 100%;
      height: 100%;
      max-height: 100%;
      background: radial-gradient(circle at center, #0f1c3f 0%, #060b18 100%);
      border: 1.5px solid #1e3a8a;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      padding: 12px 18px;
      box-sizing: border-box;
      overflow: hidden;
    }
    .lcd-gauges-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1.5px solid #1e3a8a;
      padding-bottom: 8px;
      flex-shrink: 0;
    }
    .lcd-gauges-logo {
      height: 38px;
      filter: drop-shadow(0 3px 10px rgba(56, 189, 248, 0.3));
    }
    .lcd-gauges-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 10px;
      flex: 1;
      margin-top: 10px;
      min-height: 0;
    }
    .lcd-gauge-card {
      background: rgba(17, 28, 68, 0.75);
      border: 1.5px solid #1e3a8a;
      border-radius: 8px;
      padding: 8px 14px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      min-height: 0;
      overflow: hidden;
    }
    .lcd-gauge-title {
      font-size: 23px;
      font-weight: 600;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .lcd-gauge-val-xl {
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: normal !important;
      font-variant-numeric: normal !important;
    }
    .lcd-gauge-sub-xl {
      font-size: 18px;
      color: var(--text-muted);
      margin-top: 3px;
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif !important;
      letter-spacing: normal !important;
    }

    /* =========================================================
       WEB INTERACTIVE COCKPIT STYLES
       ========================================================= */
    .cockpit-header {
      width: 100%;
      max-width: 1480px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      background: var(--card-bg);
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-bottom: 10px;
      gap: 12px;
    }
    .logo-area { display: flex; align-items: center; gap: 12px; }
    .logo-area img { height: 32px; }
    .logo-area h1 { font-size: 18px; color: var(--accent); font-weight: 700; }
    .url-badge { background: #070d1d; border: 1px solid var(--accent); color: var(--accent); font-weight: bold; font-size: 13px; padding: 4px 10px; border-radius: 6px; }

    /* Feed Destination Switcher */
    .feed-switch-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #070d1d;
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .switch-pill-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 700;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .switch-pill-btn.active {
      background: var(--accent);
      color: #070d1d;
    }

    /* Mission Control Bar */
    .cockpit-mission-bar {
      width: 100%;
      max-width: 1480px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 16px;
      margin-bottom: 12px;
      gap: 12px;
    }
    .state-pill {
      background: #070d1d;
      border: 2px solid #3b82f6;
      color: #60a5fa;
      font-size: 14px;
      font-weight: 800;
      border-radius: 6px;
      padding: 6px 14px;
      letter-spacing: 0.5px;
    }
    .mission-actions { display: flex; gap: 8px; }

    /* Navigation Tabs */
    .tabs { display: flex; flex-wrap: wrap; gap: 6px; width: 100%; max-width: 1480px; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: #070d1d; border: 1px solid var(--border); color: var(--text-muted); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; white-space: nowrap; transition: all 0.2s; font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif, 'Noto Color Emoji'; }
    .tab-btn.active { background: var(--accent); color: #070d1d; border-color: var(--accent); }

    /* Content Cards */
    .main-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; width: 100%; max-width: 1480px; }
    @media (min-width: 992px) {
      .cockpit-tab-content {
        grid-template-columns: 1.65fr 1fr;
      }
    }
    .wide-cam-card {
      grid-column: 1 / -1 !important;
      max-height: 72vh;
    }
    .color-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .color-pill {
      border: 2px solid var(--border);
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      background: #070d1d;
      color: var(--text);
      transition: all 0.2s;
    }
    .color-pill.active {
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    .target-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .target-chip {
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      background: #070d1d;
      color: var(--text);
      transition: all 0.2s;
    }
    .target-chip:hover {
      border-color: var(--accent);
    }
    .target-chip.active {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.18);
      color: #38bdf8;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.35);
    }
    #tab-activities {
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)) !important;
    }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; display: flex; flex-direction: column; gap: 14px; }
    .card-title { font-size: 15px; font-weight: 700; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }

    /* Camera View Container (Web Mode) */
    .video-container { position: relative; width: 100%; aspect-ratio: 4/3; background: #000; border-radius: 6px; overflow: hidden; border: 2px solid var(--border); cursor: crosshair; }
    .video-container img { width: 100%; height: 100%; object-fit: contain; }
    .video-overlay-text { position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.65); padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #fff; }

    /* Standby Card (Shown on Web when Feed is on LCD) */
    .standby-card {
      width: 100%;
      aspect-ratio: 4/3;
      background: #070d1d;
      border: 2px dashed #1e3a8a;
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      padding: 24px;
      gap: 12px;
    }
    .standby-logo { height: 50px; opacity: 0.85; margin-bottom: 4px; }
    .standby-title { font-size: 16px; font-weight: 700; color: var(--accent); }
    .standby-desc { font-size: 13px; color: var(--text-muted); max-width: 340px; }

    /* Controls, Buttons, Sliders */
    .slider-group { display: flex; flex-direction: column; gap: 4px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); }
    .slider-val { color: var(--accent); font-weight: bold; }
    input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }

    .btn { background: #1e3a8a; color: #fff; border: 1px solid #3b82f6; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px; cursor: pointer; transition: all 0.15s; text-align: center; }
    .btn:hover { background: #2563eb; }
    .btn:active { transform: scale(0.97); }
    .btn-green { background: #065f46; border-color: #059669; color: #fff; }
    .btn-green:hover { background: #059669; }
    .btn-red { background: #991b1b; border-color: #dc2626; color: #fff; }
    .btn-red:hover { background: #dc2626; }
    .btn-accent { background: #0284c7; border-color: #38bdf8; color: #fff; }
    .btn-accent:hover { background: #38bdf8; color: #070d1d; }

    .dpad-container { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px 0; }
    .dpad-row { display: flex; gap: 8px; }
    .dpad-btn { width: 70px; height: 55px; font-size: 18px; display: flex; align-items: center; justify-content: center; user-select: none; }
    
    .toast { min-height: 20px; font-size: 12px; color: var(--green); text-align: center; }
  </style>
  <script>
    function switchTab(tabId, btn) {
      var contents = document.querySelectorAll('.tab-content');
      for (var i = 0; i < contents.length; i++) {
        contents[i].style.display = 'none';
      }
      var btns = document.querySelectorAll('.tab-btn');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.remove('active');
      }
      var el = document.getElementById(tabId);
      if (el) {
        el.style.display = 'grid';
      }
      var targetBtn = btn;
      if (!targetBtn && typeof event !== 'undefined' && event && event.target) {
        targetBtn = event.target.closest ? event.target.closest('.tab-btn') : event.target;
      }
      if (!targetBtn) {
        for (var i = 0; i < btns.length; i++) {
          var oc = btns[i].getAttribute('onclick') || '';
          if (oc.indexOf(tabId) !== -1) {
            targetBtn = btns[i];
            break;
          }
        }
      }
      if (targetBtn && targetBtn.classList) {
        targetBtn.classList.add('active');
      }
    }
    window.switchTab = switchTab;

    // Fast-path immediate detection for LCD View-Only Mode before DOM paint
    if (window.location.search.indexOf('device=lcd') !== -1 || (window.location.hostname === 'localhost' && window.location.search.indexOf('device=web') === -1)) {
      document.documentElement.classList.add('lcd-view-only');
    }
  </script>
</head>
<body>

  <!-- =========================================================
       ROBOT LCD VIEW-ONLY HUD (?device=lcd)
       ========================================================= -->
  <div id="lcd_hud_container">
    <!-- SUB-VIEW 1: FULLSCREEN CAMERA FEED (When Feed Target is LCD) -->
    <div id="lcd_view_camera">
      <img id="lcd_cam_img" src="/video_feed" alt="Robot LCD Video Feed" />
      <div class="lcd-cam-top-bar">
        <div class="lcd-pill-badge" id="lcd_cam_state_badge">STATE: IDLE</div>
      </div>
      <div class="lcd-cam-bottom-bar">
        <div class="lcd-hud-stat">
          <div class="lcd-hud-stat-lbl">Arm Poses (S1/S2/S3)</div>
          <div class="lcd-hud-stat-val" id="lcd_hud_arm">93° / 45° / 110°</div>
        </div>
        <div class="lcd-hud-stat">
          <div class="lcd-hud-stat-lbl">Motor PWM (L/R)</div>
          <div class="lcd-hud-stat-val" id="lcd_hud_motors">0 / 0</div>
        </div>
        <div class="lcd-hud-stat">
          <div class="lcd-hud-stat-lbl">Vision Target Object</div>
          <div class="lcd-hud-stat-val" id="lcd_hud_vlm" style="color:#38bdf8;">PLASTIC BOTTLE (96%)</div>
        </div>
        <div class="lcd-hud-stat">
          <div class="lcd-hud-stat-lbl">Bluetooth Link</div>
          <div class="lcd-hud-stat-val" id="lcd_hud_link" style="color:#4ade80;">● ONLINE</div>
        </div>
      </div>
    </div>

    <!-- SUB-VIEW 2: LOGO + LARGE CHARACTER TELEMETRY (When Feed Target is Web) -->
    <div id="lcd_view_gauges" style="display:none;">
      <div class="lcd-gauges-header">
        <img src="/logo.webp" class="lcd-gauges-logo" alt="Erovoutika" />
        <div class="lcd-pill-badge" id="lcd_g_state" style="font-size:14px; padding:5px 14px;">STATE: IDLE</div>
      </div>
      <div class="lcd-gauges-grid">
        <div class="lcd-gauge-card">
          <div class="lcd-gauge-title">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
            3-Axis Arm Poses
          </div>
          <div class="lcd-gauge-val-xl" id="lcd_g_arm">S1: 93° | S2: 45° | S3: 110°</div>
          <div class="lcd-gauge-sub-xl">Shoulder • Elbow • Gripper</div>
        </div>
        <div class="lcd-gauge-card">
          <div class="lcd-gauge-title">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path d="M12 2v20M2 12h20"/></svg>
            Drive Propulsion Motors
            Drive Motors
          </div>
          <div class="lcd-gauge-val-xl" id="lcd_g_motors">LEFT: 0 PWM | RIGHT: 0 PWM</div>
          <div class="lcd-gauge-sub-xl" id="lcd_g_motors_sub">Chassis: Stopped (Standby)</div>
        </div>
        <div class="lcd-gauge-card">
          <div class="lcd-gauge-title">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/></svg>
            Hardware Communication
          </div>
          <div class="lcd-gauge-val-xl" id="lcd_g_link">● HC-05 BLUETOOTH CONNECTED</div>
          <div class="lcd-gauge-sub-xl" id="lcd_g_ping">Ping Latency: 440 ms | Port: /dev/rfcomm0</div>
        </div>
        <div class="lcd-gauge-card">
          <div class="lcd-gauge-title">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
            Vision Categorization
          </div>
          <div class="lcd-gauge-val-xl" id="lcd_g_vlm" style="color:#38bdf8;">PLASTIC BOTTLE (96%)</div>
          <div class="lcd-gauge-sub-xl" id="lcd_g_vlm_details">Clear PET / Blue Cap | Pickable: True</div>
        </div>
      </div>
    </div>
  </div>

  <!-- =========================================================
       WEB INTERACTIVE COCKPIT (Regular Browser)
       ========================================================= -->
  <header class="cockpit-header">
    <div class="logo-area">
      <img src="/logo.webp" alt="Logo" />
      <div>
        <h1>ErovoutikaGrab Dashboard</h1>
      </div>
      <span class="url-badge">🔒 https://egrabbot.local:5001</span>
    </div>

    <!-- Live Feed Destination Selector -->
    <div class="feed-switch-bar">
      <span style="font-size:12px; color:var(--text-muted); font-weight:700;">📺 Feed Target:</span>
      <button id="btn_target_lcd" class="switch-pill-btn active" onclick="setFeedDestination('lcd')">🖥️ Robot LCD (Default)</button>
      <button id="btn_target_web" class="switch-pill-btn" onclick="setFeedDestination('web')">🌐 Web Browser</button>
    </div>

    <div style="display:flex; gap:8px; align-items:center;">
      <button class="btn btn-accent" style="padding: 4px 10px; font-size: 11px;" onclick="reconnectBT()">🔄 Reconnect BT</button>
      <button class="btn" style="padding: 4px 10px; font-size: 11px;" onclick="toggleFullScreen()">⛶ Fullscreen</button>
      <button class="btn" style="padding: 4px 10px; font-size: 11px; border-color:#eab308; color:#facc15;" onclick="restartWebService()" title="Restart Web Server Service">⚡ Restart Service</button>
    </div>
  </header>

  <!-- Autonomous Mission Control Bar -->
  <div class="cockpit-mission-bar">
    <div class="state-pill" id="web_state_pill">STATE: IDLE</div>
    <div class="mission-actions">
      <button class="btn btn-green" id="btn_auto_pick" onclick="toggleAuto()">▶ START AUTO PICK</button>
      <button class="btn btn-red" onclick="sendStop()">🛑 EMERGENCY STOP</button>
      <button class="btn btn-accent" id="btn_showcase" onclick="triggerShowcase()">★ SHOWCASE</button>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-teleop')">🕹️ Main Cockpit & Cam</button>
    <button class="tab-btn" onclick="switchTab('tab-activities')">🎯 Activities</button>
    <button class="tab-btn" onclick="switchTab('tab-yoloe')">👁️ Vision Settings</button>
    <button class="tab-btn" onclick="switchTab('tab-arm')">🦾 Robotic Arm (Alternating)</button>
    <button class="tab-btn" onclick="switchTab('tab-motors')">🏎️ Motors & Speeds</button>
    <button class="tab-btn" onclick="switchTab('tab-vision')">🎯 Sweet Spot Calibration</button>
    <button class="tab-btn" onclick="switchTab('tab-network')">📶 WiFi & Hotspot</button>
    <button class="tab-btn" onclick="switchTab('tab-about')">🌐 About Erovoutika</button>
  </div>

  <!-- TAB 1: MAIN COCKPIT & CAMERA -->
  <div id="tab-teleop" class="main-grid tab-content cockpit-tab-content">
    <div class="card" id="cam_card">
      <div class="card-title">
        <span>📷 Camera Feed & Sweet Spot</span>
        <div style="display:flex; align-items:center; gap:8px;">
          <button class="btn" id="btn_cam_expand" style="padding:3px 8px; font-size:11px; background:#070d1d;" onclick="toggleCamExpand()">⛶ Expand Width</button>
          <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Interactive Feed</span>
        </div>
      </div>

      <!-- ACTIVE STREAM (Shown when destination is 'web') -->
      <div id="web_active_video" class="video-container" onclick="handleVideoClick(event)">
        <img id="cam_feed" src="/video_feed" alt="Video stream" />
        <div class="video-overlay-text">Cyan Cross: Grasp Sweet Spot (Tap to recalibrate)</div>
      </div>

      <!-- STANDBY CARD (Shown when destination is 'lcd') -->
      <div id="web_standby_video" class="standby-card" style="display:none;">
        <img src="/logo.webp" class="standby-logo" alt="Erovoutika" />
        <div class="standby-title">Camera Feed Routed to Robot LCD</div>
        <div class="standby-desc">The live video stream is currently active on the robot's physical display.</div>
        <button class="btn btn-accent" style="margin-top:10px;" onclick="setFeedDestination('web')">
          🌐 Switch Feed to Web Browser
        </button>
      </div>

      <div class="grid-2">
        <button class="btn btn-accent" onclick="sendMacro('pick')">🎯 Full Pick & Lift Sequence</button>
        <button class="btn btn-red" onclick="sendStop()">🛑 STOP MOTORS</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">🏎️ Drive Teleoperation (WASD / Touch)</div>
      <div class="dpad-container">
        <button class="btn dpad-btn" onmousedown="startDrive('F')" onmouseup="stopDrive()" ontouchstart="startDrive('F')" ontouchend="stopDrive()">▲</button>
        <div class="dpad-row">
          <button class="btn dpad-btn" onmousedown="startDrive('L')" onmouseup="stopDrive()" ontouchstart="startDrive('L')" ontouchend="stopDrive()">◀</button>
          <button class="btn dpad-btn btn-red" onclick="sendStop()">■</button>
          <button class="btn dpad-btn" onmousedown="startDrive('R')" onmouseup="stopDrive()" ontouchstart="startDrive('R')" ontouchend="stopDrive()">▶</button>
        </div>
        <button class="btn dpad-btn" onmousedown="startDrive('B')" onmouseup="stopDrive()" ontouchstart="startDrive('B')" ontouchend="stopDrive()">▼</button>
      </div>

      <div class="card-title" style="margin-top:6px; display:flex; justify-content:space-between; align-items:center;">
        <span>🦾 Quick Arm Poses & Live Status</span>
        <span id="tab1_pulse_badge" style="font-size:10px; padding:2px 6px; border-radius:4px; background:rgba(56,189,248,0.15); color:#38bdf8; display:none;">⚡ ±1° Pulse (1.0s)</span>
      </div>

      <!-- Live Arm Angle Readout Bar in Tab 1 Cockpit -->
      <div style="display:flex; justify-content:space-between; align-items:center; background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 12px; margin-bottom:8px;">
        <div style="font-size:11px; color:var(--text-muted);">
          Shoulder: <span id="tab1_arm_s1" style="font-weight:700; color:#f8fafc;">__LIVE_S1__°</span>
        </div>
        <div style="font-size:11px; color:var(--text-muted);">
          Elbow: <span id="tab1_arm_s2" style="font-weight:700; color:#f8fafc;">__LIVE_S2__°</span>
        </div>
        <div style="font-size:11px; color:var(--text-muted);">
          Gripper: <span id="tab1_arm_s3" style="font-weight:700; color:#38bdf8;">__LIVE_S3__°</span>
        </div>
      </div>

      <div class="grid-3">
        <button class="btn btn-accent" onclick="sendMacro('stow')">⬆️ UP/STOW</button>
        <button class="btn btn-accent" onclick="sendMacro('center')">⚖️ CENTER</button>
        <button class="btn btn-accent" onclick="sendMacro('down')">⬇️ DOWN/REACH</button>
      </div>
      <div class="grid-2">
        <button class="btn btn-green" onclick="sendMacro('open')">👐 OPEN (<span id="lbl_tab1_open">__S3_OPEN__</span>°)</button>
        <button class="btn btn-green" onclick="sendMacro('close')">✊ CLOSE (<span id="lbl_tab1_close">__S3_CLOSE__</span>°)</button>
      </div>
      <div class="toast" id="msg_teleop"></div>
    </div>

    <!-- Live YOLOE Real-Time AI Vision Telemetry Card -->
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-title">
        <span>👁️ YOLOE Real-Time Object Detection & Alignment</span>
        <span id="yoloe_live_latency" style="font-size:12px; color:var(--text-muted); font-weight:normal;">Latency: -- ms</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px;">
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Target Object</div>
          <div id="yoloe_card_obj" style="font-size:18px; font-weight:900; color:#38bdf8; margin-top:4px;">Scanning...</div>
          <div style="height:6px; background:#1e3a8a; border-radius:3px; margin-top:8px; overflow:hidden;">
            <div id="yoloe_card_conf_bar" style="height:100%; width:0%; background:#10b981; transition:width 0.2s;"></div>
          </div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Pickability Status</div>
          <div id="yoloe_card_pick" style="font-size:15px; font-weight:800; color:var(--text-muted); margin-top:6px;">--</div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Safe Grasp Verification</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Sweet Spot Alignment</div>
          <div id="yoloe_card_align" style="font-size:15px; font-weight:800; color:var(--text-muted); margin-top:6px;">--</div>
          <div id="yoloe_card_err" style="font-size:11px; color:var(--text-muted); margin-top:4px;">dx: -- px | dy: -- px</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Detected Color</div>
          <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
            <div id="yoloe_card_color_dot" style="width:14px; height:14px; border-radius:50%; background:#888888; border:1px solid #ffffff33;"></div>
            <div id="yoloe_card_color" style="font-size:16px; font-weight:800; color:#f8fafc;">--</div>
          </div>
          <div id="yoloe_card_color_hex" style="font-size:11px; color:var(--text-muted); margin-top:4px;">Sub-millisecond HSV</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:12px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Vision Model & Engine</div>
          <div style="display:flex; align-items:center; gap:6px; margin-top:4px;">
            <span id="yoloe_card_backend_badge" style="background:#10b981; color:#000; font-size:10px; font-weight:900; padding:2px 6px; border-radius:4px;">LITERT</span>
            <span id="yoloe_card_model" style="font-size:13px; font-weight:700; color:#f8fafc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">__YOLOE_MODEL__</span>
          </div>
          <div id="yoloe_card_imgsz" style="font-size:11px; color:var(--text-muted); margin-top:4px;">Res: __YOLOE_IMGSZ__x__YOLOE_IMGSZ__ | Conf: __YOLOE_CONF__</div>
        </div>
      </div>
    </div>

    <!-- VLM Trajectory & Anti-Overshoot Motion Planner Card -->
    <div class="card" style="grid-column: 1 / -1; border-left: 4px solid #8b5cf6;">
      <div class="card-title">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>🧠 VLM Trajectory & Anti-Overshoot Motion Planner</span>
          <span id="vlm_badge_source" style="font-size:10px; background:#8b5cf6; color:#fff; font-weight:800; padding:2px 8px; border-radius:4px;">HYBRID VLM</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
          <span class="state-pill" style="font-size:11px; font-weight:700; background:rgba(16, 185, 129, 0.15); color:#10b981; border:1px solid #10b981; padding:3px 10px; border-radius:6px;">
            ● CONTINUOUS ADAPTATION ALWAYS ON
          </span>
          <a href="/api/vlm/adaptation_log" target="_blank" class="btn btn-blue" style="padding:4px 10px; font-size:11px; text-decoration:none; display:inline-flex; align-items:center; gap:4px;">📜 View Adaptation Log</a>
        </div>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:8px;">
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Planned Action</div>
          <div id="vlm_plan_action" style="font-size:18px; font-weight:900; color:#a78bfa; margin-top:4px;">IDLE</div>
          <div id="vlm_plan_status" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Awaiting trigger or target</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">PWM & Pulse Duration</div>
          <div id="vlm_plan_pwm_dur" style="font-size:16px; font-weight:800; color:#38bdf8; margin-top:4px;">-- PWM | -- ms</div>
          <div id="vlm_plan_damping" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Damping Factor: 1.0x</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Anti-Overshoot Guard</div>
          <div id="vlm_plan_overshoot" style="font-size:14px; font-weight:800; color:#10b981; margin-top:4px;">🛡️ LOW (Safe Distance)</div>
          <div id="vlm_plan_area_ratio" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Area Ratio: -- %</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Anti-Stiction Guard</div>
          <div id="vlm_plan_stiction" style="font-size:14px; font-weight:800; color:#10b981; margin-top:4px;">⚡ Baseline Speed Sync</div>
          <div id="vlm_plan_seg_pts" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Contour: -- points</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Servo Arm Grasp Plan</div>
          <div id="vlm_arm_readiness" style="font-size:14px; font-weight:800; color:#38bdf8; margin-top:4px;">🦾 STOWED (Holding)</div>
          <div id="vlm_arm_angles" style="font-size:11px; color:var(--text-muted); margin-top:2px;">S1: --° | S2: --° | Grip: --°</div>
        </div>
      </div>
      <div style="background:rgba(139, 92, 246, 0.08); border:1px solid rgba(139, 92, 246, 0.25); border-radius:6px; padding:10px 14px; margin-top:10px; display:flex; gap:10px; align-items:flex-start;">
        <span style="font-size:16px;">💡</span>
        <div style="flex:1;">
          <div style="font-size:11px; font-weight:800; color:#a78bfa; text-transform:uppercase;">VLM Physical Reasoning & Rationale:</div>
          <div id="vlm_plan_rationale" style="font-size:12px; color:#e2e8f0; margin-top:2px; line-height:1.4;">
            Target coordinates, horizontal error, and segmentation geometry are evaluated against motor friction and inertial stopping distance. Click "Ask VLM to Plan" or enable Auto-Advisory.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: AUTONOMOUS ACTIVITIES (MUTUALLY EXCLUSIVE) -->
  <div id="tab-activities" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <!-- Active Activity Status Card -->
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-title">
        <span>🎯 Autonomous Activities Subsystem</span>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
          <div style="display:flex; align-items:center; gap:6px; background:rgba(255,255,255,0.06); padding:3px 8px; border-radius:6px; border:1px solid var(--border);">
            <span style="font-size:12px; font-weight:600; color:var(--text-main);">Motion:</span>
            <button id="btn_toggle_motion" class="btn" style="padding:4px 10px; font-size:11px; font-weight:700; background:#2563eb; color:#fff;" onclick="toggleActivityMotion()">🚗 AUTONOMOUS DRIVE (PURSUIT)</button>
          </div>
          <button class="btn btn-green" style="padding:6px 14px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE ALL ACTIVITY SETTINGS TO YAML</button>
          <span id="act_status_badge" class="state-pill" style="font-size:12px; padding:3px 10px;">IDLE (NO ACTIVITY)</span>
        </div>
      </div>
      <div style="font-size:13px; color:var(--text-muted); line-height:1.5;">
        Autonomous activities run mutually exclusively. Starting any activity automatically halts any other active task. Any manual teleoperation command (WASD, D-pad, nudge, or stop) immediately acts as a safety override and halts the running activity.
      </div>
      <div class="toast" id="msg_activities"></div>
    </div>

    <!-- Person Follower Card -->
    <div class="card">
      <div class="card-title">
        <span>👤 Person Follower (LiteRT AI)</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Autonomous Pursuit</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Tracks and follows a detected person using on-device LiteRT COCO object detection. Automatically centers yaw heading and maintains safe approach distance.
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Target Distance (Box Height % of Screen)</span>
          <span class="slider-val" id="val_pf_dist">__PF_DIST__%</span>
        </div>
        <input type="range" id="pf_dist" min="20" max="70" step="5" value="__PF_DIST__" oninput="document.getElementById('val_pf_dist').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Follow Speed PWM</span>
          <span class="slider-val" id="val_pf_speed">__PF_SPEED__</span>
        </div>
        <input type="range" id="pf_speed" min="150" max="255" step="5" value="__PF_SPEED__" oninput="document.getElementById('val_pf_speed').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_pf" onclick="startPersonFollower()">▶ START PERSON FOLLOWER</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

    <!-- Color Tracking Card -->
    <div class="card">
      <div class="card-title">
        <span>🎨 Fast Color Tracking (HSV)</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">>300 FPS Vectorized</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Locks onto and pursues calibrated color blobs in real time. Select target color pill below:
      </div>
      <div class="color-pills" id="ct_color_pills">
        <button class="color-pill active" onclick="selectTrackingColor('Red', this)" id="cp_Red" style="border-left:8px solid #ef4444;">🔴 Red</button>
        <button class="color-pill" onclick="selectTrackingColor('Green', this)" id="cp_Green" style="border-left:8px solid #10b981;">🟢 Green</button>
        <button class="color-pill" onclick="selectTrackingColor('Blue', this)" id="cp_Blue" style="border-left:8px solid #3b82f6;">🔵 Blue</button>
        <button class="color-pill" onclick="selectTrackingColor('Yellow', this)" id="cp_Yellow" style="border-left:8px solid #facc15;">🟡 Yellow</button>
        <button class="color-pill" onclick="selectTrackingColor('Orange', this)" id="cp_Orange" style="border-left:8px solid #f97316;">🟠 Orange</button>
        <button class="color-pill" onclick="selectTrackingColor('Purple', this)" id="cp_Purple" style="border-left:8px solid #a855f7;">🟣 Purple</button>
      </div>
      <div class="slider-group" style="margin-top:6px;">
        <div class="slider-header">
          <span>Target Area Ratio (% of Frame)</span>
          <span class="slider-val" id="val_ct_area">__CT_AREA__%</span>
        </div>
        <input type="range" id="ct_area" min="2" max="25" step="1" value="__CT_AREA__" oninput="document.getElementById('val_ct_area').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Pursuit Speed PWM</span>
          <span class="slider-val" id="val_ct_speed">__CT_SPEED__</span>
        </div>
        <input type="range" id="ct_speed" min="150" max="255" step="5" value="__CT_SPEED__" oninput="document.getElementById('val_ct_speed').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_ct" onclick="startColorTracking()">▶ START COLOR TRACKING</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

    <!-- Generic Object Tracking Card -->
    <div class="card">
      <div class="card-title">
        <span>🎯 Object Tracking (LiteRT AI)</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">80 COCO Classes</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Tracks and pursues any specified object class using on-device LiteRT neural inference. Select a quick preset or type custom target:
      </div>
      <div class="target-chips" id="ot_chips">
        <button class="target-chip active" data-obj="bottle" onclick="setTrackingObjectPreset('bottle', this)">🍾 Bottle</button>
        <button class="target-chip" data-obj="cup" onclick="setTrackingObjectPreset('cup', this)">☕ Cup</button>
        <button class="target-chip" data-obj="sports ball" onclick="setTrackingObjectPreset('sports ball', this)">⚽ Ball</button>
        <button class="target-chip" data-obj="cell phone" onclick="setTrackingObjectPreset('cell phone', this)">📱 Phone</button>
        <button class="target-chip" data-obj="book" onclick="setTrackingObjectPreset('book', this)">📖 Book</button>
        <button class="target-chip" data-obj="apple" onclick="setTrackingObjectPreset('apple', this)">🍎 Apple</button>
        <button class="target-chip" data-obj="chair" onclick="setTrackingObjectPreset('chair', this)">🪑 Chair</button>
        <button class="target-chip" data-obj="backpack" onclick="setTrackingObjectPreset('backpack', this)">🎒 Backpack</button>
      </div>
      <div style="display:flex; flex-direction:column; gap:6px; margin-top:6px;">
        <div style="display:flex; gap:8px; align-items:center;">
          <span style="font-size:12px; font-weight:700; color:var(--text-muted); white-space:nowrap;">Select Object:</span>
          <select id="ot_target_select" onchange="onSelectTrackingObject(this.value)" style="flex:1; background:#070d1d; color:#38bdf8; border:1px solid var(--border); padding:7px 10px; border-radius:6px; font-size:13px; font-weight:700; cursor:pointer;">
            <optgroup label="🥤 Drinkware & Containers">
              <option value="bottle">🍾 Bottle</option>
              <option value="spray bottle">🧴 Spray Bottle (Trained)</option>
              <option value="cup">☕ Cup</option>
              <option value="wine glass">🍷 Wine Glass</option>
              <option value="bowl">🥣 Bowl</option>
              <option value="can">🥫 Can</option>
            </optgroup>
            <optgroup label="🍎 Fruits & Small Foods">
              <option value="apple">🍎 Apple</option>
              <option value="banana">🍌 Banana</option>
              <option value="orange">🍊 Orange</option>
              <option value="sandwich">🥪 Sandwich</option>
              <option value="broccoli">🥦 Broccoli</option>
              <option value="carrot">🥕 Carrot</option>
              <option value="hot dog">🌭 Hot Dog</option>
              <option value="pizza">🍕 Pizza</option>
              <option value="donut">🍩 Donut</option>
              <option value="cake">🍰 Cake</option>
            </optgroup>
            <optgroup label="📱 Tech & Office Gadgets">
              <option value="cell phone">📱 Cell Phone</option>
              <option value="mouse">🖱️ Mouse</option>
              <option value="remote">📺 Remote</option>
              <option value="keyboard">⌨️ Keyboard</option>
              <option value="laptop">💻 Laptop</option>
              <option value="book">📖 Book</option>
              <option value="clock">⏰ Clock</option>
            </optgroup>
            <optgroup label="✂️ Tools & Utensils">
              <option value="scissors">✂️ Scissors</option>
              <option value="fork">🍴 Fork</option>
              <option value="knife">🔪 Knife</option>
              <option value="spoon">🥄 Spoon</option>
              <option value="toothbrush">🪥 Toothbrush</option>
              <option value="vase">🏺 Vase</option>
            </optgroup>
            <optgroup label="⚽ Sports & Toys">
              <option value="sports ball">⚽ Sports Ball</option>
              <option value="frisbee">🥏 Frisbee</option>
              <option value="baseball glove">🧤 Baseball Glove</option>
              <option value="teddy bear">🧸 Teddy Bear</option>
            </optgroup>
            <optgroup label="🎒 Personal Items & Bags">
              <option value="backpack">🎒 Backpack</option>
              <option value="handbag">👜 Handbag</option>
              <option value="suitcase">🧳 Suitcase</option>
              <option value="umbrella">☂️ Umbrella</option>
            </optgroup>
            <optgroup label="✍️ Custom / Open Vocabulary">
              <option value="custom">✏️ Custom Object (type below)...</option>
            </optgroup>
          </select>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <span style="font-size:12px; color:var(--text-muted); white-space:nowrap;">Target Object:</span>
          <input type="text" id="ot_target_input" value="__OT_TARGET__" placeholder="e.g. bottle, cup, laptop" style="flex:1; background:#070d1d; color:#fff; border:1px solid var(--border); padding:6px 10px; border-radius:6px; font-size:13px;" oninput="onCustomObjectInput(this.value)">
        </div>
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Target Distance (Box Height % of Screen)</span>
          <span class="slider-val" id="val_ot_dist">__OT_DIST__%</span>
        </div>
        <input type="range" id="ot_dist" min="15" max="70" step="5" value="__OT_DIST__" oninput="document.getElementById('val_ot_dist').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Pursuit Speed PWM</span>
          <span class="slider-val" id="val_ot_speed">__OT_SPEED__</span>
        </div>
        <input type="range" id="ot_speed" min="150" max="255" step="5" value="__OT_SPEED__" oninput="document.getElementById('val_ot_speed').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_ot" onclick="startObjectTracking()">▶ START OBJECT TRACKING</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

    <!-- Color Tracking + Object Classification Card -->
    <div class="card">
      <div class="card-title">
        <span>🔬 Color Tracking + Object Classification</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Fused HSV & LiteRT</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Pursues target color blob while LiteRT neural net identifies what the object is (e.g. "Red Bottle", "Blue Cup").
      </div>
      <div class="color-pills" id="ctc_color_pills">
        <button class="color-pill active" onclick="selectClassifyColor('Red', this)" id="cp_ctc_Red" style="border-left:8px solid #ef4444;">🔴 Red</button>
        <button class="color-pill" onclick="selectClassifyColor('Green', this)" id="cp_ctc_Green" style="border-left:8px solid #10b981;">🟢 Green</button>
        <button class="color-pill" onclick="selectClassifyColor('Blue', this)" id="cp_ctc_Blue" style="border-left:8px solid #3b82f6;">🔵 Blue</button>
        <button class="color-pill" onclick="selectClassifyColor('Yellow', this)" id="cp_ctc_Yellow" style="border-left:8px solid #facc15;">🟡 Yellow</button>
        <button class="color-pill" onclick="selectClassifyColor('Orange', this)" id="cp_ctc_Orange" style="border-left:8px solid #f97316;">🟠 Orange</button>
        <button class="color-pill" onclick="selectClassifyColor('Purple', this)" id="cp_ctc_Purple" style="border-left:8px solid #a855f7;">🟣 Purple</button>
      </div>
      <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
        <span style="font-size:12px; color:var(--text-muted);">AI Classification on Blob:</span>
        <span id="ctc_live_class" style="font-size:13px; font-weight:700; color:var(--text-muted);">Identified: None</span>
      </div>
      <div class="slider-group" style="margin-top:6px;">
        <div class="slider-header">
          <span>Target Area Ratio (% of Frame)</span>
          <span class="slider-val" id="val_ctc_area">__CTC_AREA__%</span>
        </div>
        <input type="range" id="ctc_area" min="2" max="25" step="1" value="__CTC_AREA__" oninput="document.getElementById('val_ctc_area').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Pursuit Speed PWM</span>
          <span class="slider-val" id="val_ctc_speed">__CTC_SPEED__</span>
        </div>
        <input type="range" id="ctc_speed" min="150" max="255" step="5" value="__CTC_SPEED__" oninput="document.getElementById('val_ctc_speed').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_ctc" onclick="startColorTrackAndClassify()">▶ START COLOR + CLASSIFY</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

    <!-- Object Sizing & Dimensioning Card -->
    <div class="card">
      <div class="card-title">
        <span>📏 Intelligent Object Sizing & Gripper Fit</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Calibrated Vision</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Measures physical dimensions (W x H mm, Area, Aspect Ratio) using sweet-spot calibration. Categorizes size (Tiny, Small, Medium, Large, Oversized) and verifies robot gripper jaw fit (max 85mm).
      </div>
      <div class="target-chips" id="os_chips">
        <button class="target-chip active" data-obj="any" onclick="setSizingObjectPreset('any', this)">🔍 Any Object</button>
        <button class="target-chip" data-obj="bottle" onclick="setSizingObjectPreset('bottle', this)">🍾 Bottle</button>
        <button class="target-chip" data-obj="cup" onclick="setSizingObjectPreset('cup', this)">☕ Cup</button>
        <button class="target-chip" data-obj="cell phone" onclick="setSizingObjectPreset('cell phone', this)">📱 Phone</button>
        <button class="target-chip" data-obj="sports ball" onclick="setSizingObjectPreset('sports ball', this)">⚽ Ball</button>
        <button class="target-chip" data-obj="book" onclick="setSizingObjectPreset('book', this)">📖 Book</button>
        <button class="target-chip" data-obj="apple" onclick="setSizingObjectPreset('apple', this)">🍎 Apple</button>
      </div>
      <div style="display:flex; flex-direction:column; gap:6px; margin-top:4px;">
        <div style="display:flex; gap:8px; align-items:center;">
          <span style="font-size:12px; font-weight:700; color:var(--text-muted); white-space:nowrap;">Select Target:</span>
          <select id="os_target_select" onchange="onSelectSizingObject(this.value)" style="flex:1; background:#070d1d; color:#38bdf8; border:1px solid var(--border); padding:7px 10px; border-radius:6px; font-size:13px; font-weight:700; cursor:pointer;">
            <option value="any">🔍 Any Small Object (<=85mm)</option>
            <optgroup label="🥤 Small Containers & Drinkware">
              <option value="bottle">🍾 Bottle</option>
              <option value="cup">☕ Cup</option>
              <option value="can">🥫 Can</option>
              <option value="bowl">🥣 Small Bowl</option>
            </optgroup>
            <optgroup label="🍎 Fruits & Small Foods">
              <option value="apple">🍎 Apple</option>
              <option value="orange">🍊 Orange</option>
              <option value="banana">🍌 Banana</option>
              <option value="sandwich">🥪 Sandwich</option>
            </optgroup>
            <optgroup label="📱 Tech Gadgets & Desktop Items">
              <option value="cell phone">📱 Cell Phone</option>
              <option value="mouse">🖱️ Mouse</option>
              <option value="book">📖 Small Book</option>
              <option value="sports ball">⚽ Ball</option>
            </optgroup>
            <optgroup label="✍️ Custom Target">
              <option value="custom">✏️ Custom Target (type below)...</option>
            </optgroup>
          </select>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <span style="font-size:12px; color:var(--text-muted); white-space:nowrap;">Filter Target:</span>
          <input type="text" id="os_target_input" value="__OS_TARGET__" placeholder="e.g. any, bottle, cup, apple" style="flex:1; background:#070d1d; color:#fff; border:1px solid var(--border); padding:6px 10px; border-radius:6px; font-size:13px;" oninput="onCustomSizingInput(this.value)">
        </div>
      </div>

      <!-- Real-Time Sizing Telemetry Display -->
      <div style="background:#070d1d; border:1px solid var(--border); border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:8px; margin-top:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:6px;">
          <span style="font-size:12px; color:var(--text-muted);">Target Object:</span>
          <span id="os_live_obj" style="font-size:13px; font-weight:700; color:var(--accent);">None</span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
          <div>
            <div style="font-size:11px; color:var(--text-muted);">Calibrated Width:</div>
            <div id="os_live_w" style="font-size:15px; font-weight:700; color:#fff;">-- mm</div>
            <div id="os_live_w_px" style="font-size:10px; color:var(--text-muted);">(-- px)</div>
          </div>
          <div>
            <div style="font-size:11px; color:var(--text-muted);">Calibrated Height:</div>
            <div id="os_live_h" style="font-size:15px; font-weight:700; color:#fff;">-- mm</div>
            <div id="os_live_h_px" style="font-size:10px; color:var(--text-muted);">(-- px)</div>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(255,255,255,0.08); padding-top:6px;">
          <span style="font-size:12px; color:var(--text-muted);">Size Tier:</span>
          <span id="os_live_cat" class="state-pill" style="font-size:11px; padding:2px 8px;">--</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:12px; color:var(--text-muted);">Gripper Compatibility:</span>
          <span id="os_live_gripper" class="state-pill" style="font-size:11px; padding:2px 8px;">--</span>
        </div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Inspection Distance (Box Height % of Screen)</span>
          <span class="slider-val" id="val_os_dist">__OS_DIST__%</span>
        </div>
        <input type="range" id="os_dist" min="15" max="70" step="5" value="__OS_DIST__" oninput="document.getElementById('val_os_dist').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Alignment Speed PWM</span>
          <span class="slider-val" id="val_os_speed">__OS_SPEED__</span>
        </div>
        <input type="range" id="os_speed" min="150" max="255" step="5" value="__OS_SPEED__" oninput="document.getElementById('val_os_speed').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_os" onclick="startObjectSizing()">▶ START OBJECT SIZING</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

    <!-- Cam-Only Obstacle Avoidance Card -->
    <div class="card">
      <div class="card-title">
        <span>🛡️ Cam-Only Obstacle Avoidance</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Monocular Corridor Navigation</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">
        Autonomous vision-based navigation using the camera only. Analyzes ground corridor sectors (Left, Center, Right) via edge density and object proximity to steer clear of walls, legs, and obstacles without ultrasonic or LiDAR sensors.
      </div>

      <!-- Real-Time Corridor Clearance & Navigation Display -->
      <div style="background:#070d1d; border:1px solid var(--border); border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:8px; margin-top:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:6px;">
          <span style="font-size:12px; color:var(--text-muted);">Nav Action / Path:</span>
          <span id="oa_live_action" style="font-size:13px; font-weight:700; color:var(--accent);">IDLE</span>
        </div>
        
        <div style="font-size:11px; color:var(--text-muted); margin-bottom:2px;">Ground Sector Clearances:</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; text-align:center;">
          <div id="oa_box_l" style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:6px; padding:6px 4px;">
            <div style="font-size:10px; color:var(--text-muted);">LEFT</div>
            <div id="oa_live_l" style="font-size:12px; font-weight:700; color:#10b981; margin:2px 0;">CLEAR</div>
            <div id="oa_live_l_score" style="font-size:10px; color:var(--text-muted);">0.0%</div>
          </div>
          <div id="oa_box_c" style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:6px; padding:6px 4px;">
            <div style="font-size:10px; color:var(--text-muted);">CENTER</div>
            <div id="oa_live_c" style="font-size:12px; font-weight:700; color:#10b981; margin:2px 0;">CLEAR</div>
            <div id="oa_live_c_score" style="font-size:10px; color:var(--text-muted);">0.0%</div>
          </div>
          <div id="oa_box_r" style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:6px; padding:6px 4px;">
            <div style="font-size:10px; color:var(--text-muted);">RIGHT</div>
            <div id="oa_live_r" style="font-size:12px; font-weight:700; color:#10b981; margin:2px 0;">CLEAR</div>
            <div id="oa_live_r_score" style="font-size:10px; color:var(--text-muted);">0.0%</div>
          </div>
        </div>
      </div>

      <div class="slider-group" style="margin-top:10px;">
        <div class="slider-header">
          <span>Obstacle Sensitivity (Score Threshold)</span>
          <span class="slider-val" id="val_oa_thresh">__OA_THRESH__%</span>
        </div>
        <input type="range" id="oa_thresh" min="12" max="45" step="1" value="__OA_THRESH__" oninput="document.getElementById('val_oa_thresh').innerText = this.value + '%'">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Cruising Speed PWM</span>
          <span class="slider-val" id="val_oa_cruise">__OA_CRUISE__</span>
        </div>
        <input type="range" id="oa_cruise" min="150" max="255" step="5" value="__OA_CRUISE__" oninput="document.getElementById('val_oa_cruise').innerText = this.value">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span>Avoidance Turn Speed PWM</span>
          <span class="slider-val" id="val_oa_turn">__OA_TURN__</span>
        </div>
        <input type="range" id="oa_turn" min="160" max="255" step="5" value="__OA_TURN__" oninput="document.getElementById('val_oa_turn').innerText = this.value">
      </div>
      <div class="grid-2" style="margin-top:8px;">
        <button class="btn btn-green" id="btn_start_oa" onclick="startObstacleAvoidance()">▶ START OBSTACLE AVOIDANCE</button>
        <button class="btn btn-red" onclick="stopAnyActivity()">■ HALT ACTIVITY</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px; font-size:12px; font-weight:700; background:linear-gradient(135deg, #059669 0%, #047857 100%);" onclick="saveActivitySettings()">💾 SAVE SETTINGS TO YAML</button>
    </div>

  </div>

  <!-- TAB: VISION SETTINGS -->
  <div id="tab-yoloe" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <div class="card">
      <div class="card-title">
        <span>⚙️ Detection Parameters & Model Selection</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Real-Time Config</span>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Active Model Weights</span>
        </div>
        <select id="yoloe_sel_model" style="background:#070d1d; color:#fff; border:1px solid var(--border); padding:8px; border-radius:6px; font-size:13px;">
          <option value="models/yolo11s.tflite" selected>⚡ YOLO11 Small LiteRT (yolo11s.tflite - Recommended 140ms)</option>
          <option value="models/yolo11n.tflite">🚀 YOLO11 Nano LiteRT (yolo11n.tflite - Ultra-Fast 58ms)</option>
          <option value="models/yolo11m.tflite">⚖️ YOLO11 Medium LiteRT (yolo11m.tflite - 380ms)</option>
          <option value="models/yolo11l.tflite">🔍 YOLO11 Large LiteRT (yolo11l.tflite - High Precision 488ms)</option>
          <option value="models/yoloe-11s-seg-pf.pt">🎯 YOLOE Prompt-Free Small (yoloe-11s-seg-pf.pt - 4585 classes 27MB)</option>
          <option value="models/yoloe-11m-seg-pf.pt">🎯 YOLOE Prompt-Free Medium (yoloe-11m-seg-pf.pt - 4585 classes 61MB)</option>
          <option value="models/yoloe-11l-seg-pf.pt">🎯 YOLOE Prompt-Free Large (yoloe-11l-seg-pf.pt - 4585 classes 71MB)</option>
          <option value="models/yoloe-26s-seg-pf.pt">🎯 YOLOE-26 Prompt-Free Small (yoloe-26s-seg-pf.pt - 39MB)</option>
          <option value="models/yoloe-26n-seg-pf.pt">🚀 YOLOE-26 Prompt-Free Nano (yoloe-26n-seg-pf.pt - 16MB)</option>
          <option value="models/ssd_mobilenet_v1_coco.tflite">📱 MobileNet SSD v1 COCO (ssd_mobilenet_v1_coco.tflite - 45ms)</option>
          <option value="models/efficientdet_lite0_coco.tflite">🎯 EfficientDet-Lite0 COCO (efficientdet_lite0_coco.tflite - 65ms)</option>
        </select>
        <div style="font-size:11px; color:var(--text-muted);">LiteRT models detect 80 COCO classes at up to 25 FPS. Prompt-Free (PF) YOLOE models detect 4,585 classes with segmentation masks without runtime CLIP overhead.</div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Confidence Threshold</span>
          <span class="slider-val" id="val_yoloe_conf">__YOLOE_CONF__</span>
        </div>
        <input type="range" id="yoloe_conf" min="0.05" max="0.90" step="0.05" value="__YOLOE_CONF__" oninput="document.getElementById('val_yoloe_conf').innerText = this.value">
        <div style="font-size:11px; color:var(--text-muted);">Lower values detect more objects; higher values filter false positives. Default: 0.25.</div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Inference Input Resolution (imgsz)</span>
          <span class="slider-val" id="val_yoloe_imgsz">__YOLOE_IMGSZ__px</span>
        </div>
        <select id="yoloe_imgsz" style="background:#070d1d; color:#fff; border:1px solid var(--border); padding:8px; border-radius:6px; font-size:13px;" onchange="document.getElementById('val_yoloe_imgsz').innerText = this.value + 'px'">
          <option value="256">256 x 256 (Ultra-Fast ~100ms)</option>
          <option value="320" selected>320 x 320 (Balanced ~140ms - Recommended for Pi 5)</option>
          <option value="480">480 x 480 (High Accuracy ~300ms)</option>
          <option value="640">640 x 640 (Maximum Detail ~550ms)</option>
        </select>
      </div>

      <div class="grid-2" style="margin-top:10px;">
        <button class="btn btn-accent" onclick="saveYoloeSettings(false)">⚡ APPLY LIVE (MEMORY)</button>
        <button class="btn btn-green" onclick="saveYoloeSettings(true)">💾 SAVE TO YAML</button>
      </div>
      <div class="toast" id="msg_yoloe"></div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>🤏 Pickable Object Classes</span>
        <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Grasp Eligibility</span>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Pickable Classes (Comma-Separated)</span>
        </div>
        <textarea id="yoloe_classes" rows="4" style="background:#070d1d; color:#fff; border:1px solid var(--border); padding:10px; border-radius:6px; font-family:monospace; font-size:13px; resize:vertical;">__YOLOE_CLASSES__</textarea>
        <div style="font-size:11px; color:var(--text-muted);">The vision model detects and reports <strong>all objects</strong>. Only classes listed here are eligible for autonomous grasp.</div>
      </div>

      <div style="font-size:12px; font-weight:700; color:var(--accent); margin-top:4px;">⚡ Quick Pickable Presets:</div>
      <div class="grid-2" style="margin-bottom:6px;">
        <button class="btn" style="background:#0f172a; font-size:11px; padding:6px;" onclick="setYoloePreset('all_objects')">🌐 All COCO Objects</button>
        <button class="btn" style="background:#0f172a; font-size:11px; padding:6px;" onclick="setYoloePreset('all_trash')">🗑️ Recyclables Only</button>
      </div>
      <div class="grid-2">
        <button class="btn" style="background:#0f172a; font-size:11px; padding:6px;" onclick="setYoloePreset('containers')">🥤 Bottles & Cans</button>
        <button class="btn" style="background:#0f172a; font-size:11px; padding:6px;" onclick="setYoloePreset('office')">📦 Boxes & Office</button>
      </div>

      <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px; font-size:12px; color:var(--text-muted); margin-top:6px;">
        <strong style="color:#38bdf8;">Detect Everything, Pick Selectively:</strong> The vision model always scans and classifies all visible objects (80 COCO classes). Only objects whose class matches your pickable list above will be eligible for autonomous grasp alignment. Non-pickable objects are still detected and reported in telemetry.
      </div>
    </div>
  </div>

  <!-- TAB 2: ROBOTIC ARM CONTROLS & CALIBRATION -->
  <div id="tab-arm" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <div class="card">
      <div class="card-title">🦾 Live Arm Control (Non-Stretching Motion)</div>
      
      <div class="slider-group">
        <div class="slider-header"><span>Servo 1 (Shoulder)</span><span class="slider-val" id="val_live_s1">__LIVE_S1__°</span></div>
        <input type="range" id="live_s1" min="__S1_MIN__" max="__S1_MAX__" value="__LIVE_S1__" oninput="moveArmLive('s1')">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Servo 2 (Elbow)</span><span class="slider-val" id="val_live_s2">__LIVE_S2__°</span></div>
        <input type="range" id="live_s2" min="__S2_MIN__" max="__S2_MAX__" value="__LIVE_S2__" oninput="moveArmLive('s2')">
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Servo 3 (Gripper)</span>
          <div style="display:flex; align-items:center; gap:6px;">
            <span id="gripper_pulse_badge" style="font-size:10px; padding:2px 6px; border-radius:4px; background:rgba(56,189,248,0.15); color:#38bdf8; display:none;">⚡ ±1° Pulse (1.0s)</span>
            <span class="slider-val" id="val_live_s3">__LIVE_S3__°</span>
          </div>
        </div>
        <input type="range" id="live_s3" min="__S3_LIVE_MIN__" max="__S3_LIVE_MAX__" value="__LIVE_S3__" oninput="onGripperSliderInput()" onchange="onGripperSliderChange()">
      </div>

      <div class="grid-3" style="margin-top: 10px;">
        <button class="btn btn-accent" onclick="sendMacro('stow')">UP / STOW</button>
        <button class="btn btn-accent" onclick="sendMacro('center')">NEUTRAL</button>
        <button class="btn btn-accent" onclick="sendMacro('down')">DOWN / REACH</button>
      </div>
      <div class="grid-2">
        <button class="btn btn-green" onclick="sendMacro('open')">OPEN (<span id="lbl_macro_open">__S3_OPEN__</span>°)</button>
        <button class="btn btn-green" onclick="sendMacro('close')">CLOSE (<span id="lbl_macro_close">__S3_CLOSE__</span>°)</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">💾 Calibrated Pose Angles (Saved to YAML)</div>

      <div class="grid-3">
        <div class="slider-group">
          <div class="slider-header"><span>S1 Stow/Up Angle</span><span class="slider-val" id="val_s1_stow">__S1_STOW__°</span></div>
          <input type="range" id="s1_stow" min="0" max="180" value="__S1_STOW__" oninput="updateServoCal('s1_stow')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S1 Down Angle</span><span class="slider-val" id="val_s1_down">__S1_DOWN__°</span></div>
          <input type="range" id="s1_down" min="0" max="180" value="__S1_DOWN__" oninput="updateServoCal('s1_down')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S1 Center Angle</span><span class="slider-val" id="val_s1_center">__S1_CENTER__°</span></div>
          <input type="range" id="s1_center" min="0" max="180" value="__S1_CENTER__" oninput="updateServoCal('s1_center')">
        </div>
      </div>

      <div class="grid-3">
        <div class="slider-group">
          <div class="slider-header"><span>S2 Stow/Up Angle</span><span class="slider-val" id="val_s2_stow">__S2_STOW__°</span></div>
          <input type="range" id="s2_stow" min="0" max="180" value="__S2_STOW__" oninput="updateServoCal('s2_stow')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S2 Down Angle</span><span class="slider-val" id="val_s2_down">__S2_DOWN__°</span></div>
          <input type="range" id="s2_down" min="0" max="180" value="__S2_DOWN__" oninput="updateServoCal('s2_down')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S2 Center Angle</span><span class="slider-val" id="val_s2_center">__S2_CENTER__°</span></div>
          <input type="range" id="s2_center" min="0" max="180" value="__S2_CENTER__" oninput="updateServoCal('s2_center')">
        </div>
      </div>

      <div class="grid-3">
        <div class="slider-group">
          <div class="slider-header"><span>S3 Open Angle</span><span class="slider-val" id="val_s3_open">__S3_OPEN__°</span></div>
          <input type="range" id="s3_open" min="0" max="180" value="__S3_OPEN__" oninput="updateServoCal('s3_open')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S3 Close Angle</span><span class="slider-val" id="val_s3_close">__S3_CLOSE__°</span></div>
          <input type="range" id="s3_close" min="0" max="180" value="__S3_CLOSE__" oninput="updateServoCal('s3_close')">
        </div>
        <div class="slider-group">
          <div class="slider-header"><span>S3 Center Angle</span><span class="slider-val" id="val_s3_center">__S3_CENTER__°</span></div>
          <input type="range" id="s3_center" min="0" max="180" value="__S3_CENTER__" oninput="updateServoCal('s3_center')">
        </div>
      </div>

      <div class="grid-2" style="margin-top:6px;">
        <button class="btn" style="background:#1e3a8a; font-size:11px;" onclick="captureLiveAsStow()">📍 Set Live S1/S2 as STOW</button>
        <button class="btn" style="background:#1e3a8a; font-size:11px;" onclick="captureLiveAsDown()">📍 Set Live S1/S2 as DOWN</button>
      </div>
      <div class="grid-3" style="margin-top:4px;">
        <button class="btn" style="background:#1e3a8a; font-size:11px;" onclick="captureLiveAsCenter()">📍 Set Live All as CENTER</button>
        <button class="btn" style="background:#1e3a8a; font-size:11px;" onclick="captureLiveAsOpen()">📍 Set Live S3 as OPEN</button>
        <button class="btn" style="background:#1e3a8a; font-size:11px;" onclick="captureLiveAsClose()">📍 Set Live S3 as CLOSE</button>
      </div>
      <button class="btn btn-green" style="margin-top:8px;" onclick="saveServos()">💾 SAVE SERVO CALIBRATION TO YAML</button>
      <div class="toast" id="msg_servos"></div>
    </div>
  </div>

  <!-- TAB 3: MOTORS -->
  <div id="tab-motors" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <div class="card">
      <div class="card-title">🏎️ Motor Micro-Nudges</div>
      <div class="dpad-container">
        <button class="btn dpad-btn" onclick="sendNudge('F')">▲</button>
        <div class="dpad-row">
          <button class="btn dpad-btn" onclick="sendNudge('L')">◀</button>
          <button class="btn dpad-btn btn-red" onclick="sendStop()">■</button>
          <button class="btn dpad-btn" onclick="sendNudge('R')">▶</button>
        </div>
        <button class="btn dpad-btn" onclick="sendNudge('B')">▼</button>
      </div>
      <div style="font-size:12px; color:var(--text-muted); text-align:center;">
        Nudge commands send discrete 250ms pulses for fine alignment.
      </div>
    </div>

    <div class="card">
      <div class="card-title">⚙️ Motor Calibration & Speeds</div>

      <div class="slider-group">
        <div class="slider-header"><span>Base Speed PWM</span><span class="slider-val" id="val_base_speed">__BASE_SPD__</span></div>
        <input type="range" id="base_speed" min="150" max="255" value="__BASE_SPD__" oninput="updateMotorVal('base_speed')">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Turn Speed PWM</span><span class="slider-val" id="val_turn_speed">__TURN_SPD__</span></div>
        <input type="range" id="turn_speed" min="150" max="255" value="__TURN_SPD__" oninput="updateMotorVal('turn_speed')">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Nudge PWM</span><span class="slider-val" id="val_nudge_pwm">__NUDGE_PWM__</span></div>
        <input type="range" id="nudge_pwm" min="180" max="255" value="__NUDGE_PWM__" oninput="updateMotorVal('nudge_pwm')">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Right Motor Trim (+Right / -Left)</span><span class="slider-val" id="val_trim_offset">__TRIM__</span></div>
        <input type="range" id="trim_offset" min="-30" max="30" value="__TRIM__" oninput="updateMotorVal('trim_offset')">
      </div>

      <button class="btn btn-green" onclick="saveMotors()">💾 SAVE MOTOR PARAMETERS TO YAML</button>
      <div class="toast" id="msg_motors"></div>
    </div>
  </div>

  <!-- TAB 4: SWEET SPOT -->
  <div id="tab-vision" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <div class="card">
      <div class="card-title">🎯 Grasp Sweet Spot Preview</div>
      <div class="video-container" onclick="handleVideoClick(event)">
        <img src="/video_feed" alt="Video stream" />
        <div class="video-overlay-text">Tap anywhere to re-center target box</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">📐 Bounding Box Geometry</div>

      <div class="slider-group">
        <div class="slider-header"><span>Center X</span><span class="slider-val" id="val_cx">__CX__</span></div>
        <input type="range" id="cx" min="50" max="590" value="__CX__" oninput="updateSpot()">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Center Y</span><span class="slider-val" id="val_cy">__CY__</span></div>
        <input type="range" id="cy" min="50" max="430" value="__CY__" oninput="updateSpot()">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Box Width</span><span class="slider-val" id="val_bw">__BW__</span></div>
        <input type="range" id="bw" min="40" max="500" value="__BW__" oninput="updateSpot()">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Box Height</span><span class="slider-val" id="val_bh">__BH__</span></div>
        <input type="range" id="bh" min="40" max="400" value="__BH__" oninput="updateSpot()">
      </div>

      <button class="btn btn-green" onclick="saveVision()">💾 SAVE SWEET SPOT TO YAML</button>
      <div class="toast" id="msg_vision"></div>
    </div>
  </div>

  <!-- TAB: WIRELESS NETWORK & HOTSPOT -->
  <div id="tab-network" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <!-- Network Status Card -->
    <div class="card" style="grid-column: 1 / -1; border-left: 4px solid #3b82f6;">
      <div class="card-title">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>📶 Wireless Connection & Hotspot Status</span>
          <span id="wifi_status_badge" class="state-pill" style="font-size:11px; padding:3px 10px;">SCANNING...</span>
        </div>
        <button class="btn btn-blue" style="padding:4px 10px; font-size:11px;" onclick="refreshWiFiStatus()">🔄 Refresh Status</button>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:10px;">
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Active Wireless Mode</div>
          <div id="wifi_live_mode" style="font-size:18px; font-weight:900; color:#38bdf8; margin-top:4px;">CLIENT</div>
          <div id="wifi_live_dev" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Interface: wlan0</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Current SSID / Network</div>
          <div id="wifi_live_ssid" style="font-size:18px; font-weight:900; color:#a78bfa; margin-top:4px;">GFiber_ef820</div>
          <div id="wifi_live_sec" style="font-size:11px; color:var(--text-muted); margin-top:2px;">Security: WPA2</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">IP Address (Web Cockpit)</div>
          <div id="wifi_live_ip" style="font-size:18px; font-weight:900; color:#10b981; margin-top:4px;">192.168.254.130</div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Port: 5001 (HTTPS)</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:10px;">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Signal Strength</div>
          <div id="wifi_live_signal" style="font-size:18px; font-weight:900; color:#f59e0b; margin-top:4px;">100%</div>
          <div id="wifi_live_bars" style="font-size:11px; color:var(--text-muted); margin-top:2px;">▂▄▆█ Excellent</div>
        </div>
      </div>

      <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:6px; padding:10px 14px; margin-top:12px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:12px; font-weight:700; color:var(--text-main);">Bootup Default Mode:</span>
          <label style="font-size:12px; color:var(--text-muted); display:inline-flex; align-items:center; gap:4px; cursor:pointer;">
            <input type="radio" name="rb_boot_mode" value="ap" id="rb_boot_ap" onchange="setBootNetworkMode('ap')"> Access Point Hotspot (Default)
          </label>
          <label style="font-size:12px; color:var(--text-muted); display:inline-flex; align-items:center; gap:4px; cursor:pointer; margin-left:8px;">
            <input type="radio" name="rb_boot_mode" value="wifi" id="rb_boot_wifi" onchange="setBootNetworkMode('wifi')"> WiFi Client Station
          </label>
        </div>
        <span style="font-size:11px; color:var(--text-muted);">Hotspot IP: 10.42.0.1 | Client IP: Dynamic DHCP</span>
      </div>
      <div class="toast" id="msg_wifi_global"></div>
    </div>

    <!-- WiFi Scanner & Connect Card -->
    <div class="card">
      <div class="card-title">
        <span>🔍 WiFi Network Scanner & Connection</span>
        <button class="btn btn-blue" id="btn_scan_wifi" style="padding:4px 10px; font-size:11px;" onclick="scanWiFiNetworks()">🔍 Scan Nearby Networks</button>
      </div>
      <div style="font-size:12px; color:var(--text-muted); line-height:1.4;">
        Click "Scan" to detect nearby 2.4 GHz / 5 GHz access points. Select a network from the list, enter the WPA/WPA2 passphrase, and connect.
      </div>

      <div style="margin-top:12px;">
        <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:var(--text-muted);">Available Networks List</label>
        <div id="wifi_networks_container" style="max-height:160px; overflow-y:auto; border:1px solid var(--border); border-radius:6px; background:#070d1d; margin-top:4px; padding:6px;">
          <div style="font-size:12px; color:var(--text-muted); text-align:center; padding:12px;">Click "Scan Nearby Networks" to discover available routers.</div>
        </div>
      </div>

      <div style="margin-top:12px;">
        <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:var(--text-muted);">Selected Network SSID</label>
        <input type="text" id="wifi_input_ssid" placeholder="Enter or select SSID..." style="width:100%; box-sizing:border-box; padding:8px 10px; background:#070d1d; border:1px solid var(--border); border-radius:6px; color:#fff; font-size:13px; margin-top:4px;">
      </div>

      <div style="margin-top:10px;">
        <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:var(--text-muted);">WiFi Password / WPA Passphrase</label>
        <div style="position:relative; margin-top:4px;">
          <input type="password" id="wifi_input_pwd" placeholder="Enter WiFi password (leave empty if open)..." style="width:100%; box-sizing:border-box; padding:8px 36px 8px 10px; background:#070d1d; border:1px solid var(--border); border-radius:6px; color:#fff; font-size:13px;">
          <button type="button" onclick="togglePasswordVisibility('wifi_input_pwd')" style="position:absolute; right:8px; top:50%; transform:translateY(-50%); background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:14px;">👁️</button>
        </div>
      </div>

      <div style="margin-top:14px; display:flex; gap:10px;">
        <button class="btn btn-green" id="btn_connect_wifi" style="flex:1; padding:8px 14px; font-weight:700;" onclick="connectWiFi()">🔗 Connect to Network</button>
      </div>
      <div class="toast" id="msg_wifi_connect"></div>
    </div>

    <!-- Hotspot / Access Point Mode Card -->
    <div class="card">
      <div class="card-title">
        <span>📡 Access Point (Hotspot) Controller</span>
        <span class="state-pill" style="font-size:11px; color:#f59e0b; border-color:#f59e0b;">AP HOST</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted); line-height:1.4;">
        Switch the robot into autonomous Access Point mode. The robot creates its own private WiFi network without needing an external router.
      </div>

      <div style="margin-top:12px;">
        <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:var(--text-muted);">Hotspot SSID</label>
        <input type="text" id="ap_input_ssid" value="egrabbot-ap" style="width:100%; box-sizing:border-box; padding:8px 10px; background:#070d1d; border:1px solid var(--border); border-radius:6px; color:#fff; font-size:13px; margin-top:4px;">
      </div>

      <div style="margin-top:10px;">
        <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:var(--text-muted);">Hotspot Password (min. 8 characters)</label>
        <div style="position:relative; margin-top:4px;">
          <input type="password" id="ap_input_pwd" value="egrabbot1234" style="width:100%; box-sizing:border-box; padding:8px 36px 8px 10px; background:#070d1d; border:1px solid var(--border); border-radius:6px; color:#fff; font-size:13px;">
          <button type="button" onclick="togglePasswordVisibility('ap_input_pwd')" style="position:absolute; right:8px; top:50%; transform:translateY(-50%); background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:14px;">👁️</button>
        </div>
      </div>

      <div style="background:rgba(245, 158, 11, 0.08); border:1px solid rgba(245, 158, 11, 0.25); border-radius:6px; padding:10px 12px; margin-top:12px; font-size:11px; color:#fbbf24; line-height:1.4;">
        ⚠️ Note: When switching to Hotspot mode, existing client WiFi connections will disconnect. Connect your phone/laptop to <b>egrabbot-ap</b> and navigate to <b>https://10.42.0.1:5001</b>.
      </div>

      <div style="margin-top:14px; display:flex; gap:10px;">
        <button class="btn btn-purple" id="btn_start_ap" style="flex:1; padding:8px 14px; font-weight:700; background:linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); color:#fff;" onclick="activateHotspot()">📡 Switch to E-GrabBot Hotspot Mode</button>
      </div>
      <div class="toast" id="msg_ap"></div>
    </div>
  </div>

  <!-- TAB: ABOUT EROVOUTIKA -->
  <div id="tab-about" class="main-grid tab-content cockpit-tab-content" style="display:none;">
    <div class="card" style="grid-column: 1 / -1; border-left: 4px solid #10b981; background:linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, rgba(7, 13, 29, 0.6) 100%);">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:15px;">
        <div>
          <div style="font-size:22px; font-weight:900; color:#fff; display:flex; align-items:center; gap:10px;">
            <span>🌐 Erovoutika International Corporation</span>
            <span class="state-pill" style="font-size:11px; background:#10b981; color:#000; font-weight:800; border:none;">INNOVATION HUB</span>
          </div>
          <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
            Pioneering Next-Generation Robotics, AI, Industrial Automation & STEAM Education in the Philippines
          </div>
        </div>
        <a href="https://www.erovoutika.ph/" target="_blank" rel="noopener noreferrer" class="btn btn-green" style="padding:8px 16px; font-size:12px; font-weight:700; text-decoration:none; display:inline-flex; align-items:center; gap:6px;">
          🔗 Visit Official Website (erovoutika.ph) ↗
        </a>
      </div>
    </div>

    <!-- Company Profile Card -->
    <div class="card">
      <div class="card-title">
        <span>🏢 Company Profile & Mission</span>
      </div>
      <div style="font-size:13px; color:var(--text-muted); line-height:1.6;">
        <p><b>Erovoutika International Corporation</b> is a leading Filipino technology enterprise specializing in robotics manufacturing, artificial intelligence, industrial IoT automation, and high-performance embedded systems.</p>
        <p style="margin-top:8px;">Through pioneering research and engineering, Erovoutika delivers commercial robotics, custom automated manufacturing machinery, automated guided vehicles (AGVs), and cutting-edge STEAM educational kits across Southeast Asia.</p>
      </div>

      <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:10px;">
        <div style="font-size:11px; font-weight:800; color:#38bdf8; text-transform:uppercase;">Core Competencies:</div>
        <ul style="font-size:12px; color:var(--text-muted); margin-top:6px; padding-left:18px; line-height:1.6;">
          <li><b>Autonomous Robotics:</b> Edge AI perception, visual servoing, and inverse-kinematics manipulators.</li>
          <li><b>Industrial Automation:</b> PLC systems, machine vision inspection, and smart factory integration.</li>
          <li><b>Electronics Design:</b> Custom PCB layout, firmware architecture, and power electronics.</li>
          <li><b>AI & VLM Edge Computing:</b> On-device computer vision and physical reasoning agents.</li>
        </ul>
      </div>
    </div>

    <!-- Robot Platform Specifications Card -->
    <div class="card">
      <div class="card-title">
        <span>🤖 E-GrabBot System Architecture</span>
        <span class="state-pill" style="font-size:11px;">v2.0 DUAL-CORE</span>
      </div>
      <div style="font-size:13px; color:var(--text-muted); line-height:1.6;">
        The <b>Erovoutika GrabBot (E-GrabBot)</b> is an intelligent mobile manipulation robotics platform built for autonomous visual tracking, grasping, and inspection.
      </div>

      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:12px;">
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 10px;">
          <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Chassis Drive</div>
          <div style="font-size:12px; font-weight:800; color:#38bdf8; margin-top:2px;">Dual Continuous PWM + Optical Odometry</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 10px;">
          <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Arm Manipulator</div>
          <div style="font-size:12px; font-weight:800; color:#a78bfa; margin-top:2px;">3-DOF Servo Linkage with Sweet-Spot IK</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 10px;">
          <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Vision Subsystem</div>
          <div style="font-size:12px; font-weight:800; color:#10b981; margin-top:2px;">Wide-FOV + LiteRT / YOLOE Segmentation</div>
        </div>
        <div style="background:#070d1d; border:1px solid var(--border); border-radius:6px; padding:8px 10px;">
          <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Cognitive Intelligence</div>
          <div style="font-size:12px; font-weight:800; color:#f59e0b; margin-top:2px;">Adaptive VLM Sense-Think-Act Motion</div>
        </div>
      </div>

      <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:10px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
          <div style="font-size:11px; font-weight:700; color:var(--text-main);">Official Contact & Support:</div>
          <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">Email: info@erovoutika.ph | Quezon City, Philippines</div>
        </div>
        <a href="https://www.erovoutika.ph/" target="_blank" rel="noopener noreferrer" class="btn" style="padding:5px 12px; font-size:11px; border-color:#38bdf8; color:#38bdf8; text-decoration:none;">
          🌐 www.erovoutika.ph
        </a>
      </div>
    </div>
  </div>

  <script>
    // Detect if this instance is running on the Robot Physical LCD
    const urlParams = new URLSearchParams(window.location.search);
    const isLCD = document.documentElement.classList.contains('lcd-view-only') ||
                  document.body.classList.contains('lcd-view-only') ||
                  urlParams.get('device') === 'lcd' ||
                  (window.location.hostname === 'localhost' && urlParams.get('device') !== 'web');

    if (isLCD) {
      document.documentElement.classList.add('lcd-view-only');
      document.body.classList.add('lcd-view-only');
    }

    const targetTab = urlParams.get('tab');
    if (targetTab) {
      const tabId = targetTab.startsWith('tab-') ? targetTab : 'tab-' + targetTab;
      const el = document.getElementById(tabId);
      if (el) {
        document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        el.style.display = 'grid';
        const tabBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => (b.getAttribute('onclick') || '').indexOf(tabId) !== -1);
        if (tabBtn) tabBtn.classList.add('active');
      }
    }

    // Set initial YOLOE options from backend configuration
    const initModel = '__YOLOE_MODEL__';
    const initImgsz = '__YOLOE_IMGSZ__';
    if (document.getElementById('yoloe_sel_model')) {
      document.getElementById('yoloe_sel_model').value = initModel;
    }
    if (document.getElementById('yoloe_imgsz')) {
      document.getElementById('yoloe_imgsz').value = initImgsz;
    }
    // Tab switching is provided by switchTab(tabId, btn) defined in <head>

    // Toggle Fullscreen helper
    function toggleFullScreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
      }
    }

    // Set Feed Destination
    function setFeedDestination(target) {
      fetch('/api/feed_destination', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ destination: target })
      });
    }

    // Autonomous Mode Controls
    let autoRunning = false;
    function toggleAuto() {
      const endpoint = autoRunning ? '/api/auto/stop' : '/api/auto/start';
      fetch(endpoint, { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          showToast('msg_teleop', autoRunning ? 'Autonomous mode stopped.' : 'Autonomous mode started.');
        });
    }

    function triggerShowcase() {
      fetch('/api/showcase', { method: 'POST' })
        .then(r => r.json())
        .then(d => showToast('msg_teleop', 'Capabilities showcase initiated!'));
    }

    // Real-Time Telemetry Loop (300ms)
    let lastServerStartTime = null;
    setInterval(() => {
      fetch('/api/telemetry')
        .then(r => r.json())
        .then(d => {
          // Detect service restart and automatically reload page to fetch fresh HTML/CSS/font-sizes
          if (d.server_start_time) {
            if (lastServerStartTime !== null && d.server_start_time !== lastServerStartTime) {
              console.log("[Web] Server restart detected, reloading page...");
              window.location.reload();
              return;
            }
            lastServerStartTime = d.server_start_time;
          }

          const dest = d.feed_destination || 'lcd';
          const stateStr = d.state || 'IDLE';
          autoRunning = stateStr !== 'IDLE';

          const y = d.yoloe || {};
          const objName = y.object || d.vlm_object || 'NONE';
          const confVal = Math.round((y.confidence || d.vlm_confidence || 0) * 100);
          const isPick = y.pickable || false;
          const isAlign = y.aligned || false;

          // 1. Synchronize Robot LCD View-Only Display
          if (isLCD) {
            document.getElementById('lcd_cam_state_badge').innerText = 'STATE: ' + stateStr;
            document.getElementById('lcd_g_state').innerText = 'STATE: ' + stateStr;

            document.getElementById('lcd_hud_arm').innerText = `${d.s1}° / ${d.s2}° / ${d.s3}°`;
            document.getElementById('lcd_hud_motors').innerText = `${d.left_pwm} / ${d.right_pwm}`;
            const lcdYoloeText = objName !== 'NONE' ? `${objName} (${confVal}%)` : 'Scanning...';
            document.getElementById('lcd_hud_vlm').innerText = lcdYoloeText;
            document.getElementById('lcd_hud_link').innerText = d.connected ? '● ONLINE' : '○ DISCONNECTED';
            document.getElementById('lcd_hud_link').style.color = d.connected ? '#4ade80' : '#ef4444';

            document.getElementById('lcd_g_arm').innerText = `S1: ${d.s1}° | S2: ${d.s2}° | S3: ${d.s3}°`;
            document.getElementById('lcd_g_motors').innerText = `LEFT: ${d.left_pwm} PWM | RIGHT: ${d.right_pwm} PWM`;
            document.getElementById('lcd_g_motors_sub').innerText = (d.left_pwm || d.right_pwm) ? 'Chassis: Active Driving' : 'Chassis: Stopped (Standby)';
            document.getElementById('lcd_g_link').innerText = d.connected ? '● HC-05 BLUETOOTH CONNECTED' : '○ BLUETOOTH OFFLINE';
            document.getElementById('lcd_g_link').style.color = d.connected ? '#4ade80' : '#ef4444';
            document.getElementById('lcd_g_ping').innerText = `Ping Latency: ${d.ping_ms} ms | Port: /dev/rfcomm0`;
            document.getElementById('lcd_g_vlm').innerText = lcdYoloeText;
            document.getElementById('lcd_g_vlm_details').innerText = objName !== 'NONE' ?
              `Pickable: ${isPick ? 'YES' : 'NO'} | ${isAlign ? 'ALIGNED IN SWEET SPOT' : 'dx=' + y.dx + 'px, dy=' + y.dy + 'px'} | Latency: ${y.latency_ms}ms` :
              'Scanning for targets...';

            // Route between Fullscreen Camera (LCD mode) vs Logo + Gauges (Web mode)
            if (dest === 'lcd') {
              document.getElementById('lcd_view_camera').style.display = 'flex';
              document.getElementById('lcd_view_gauges').style.display = 'none';
            } else {
              document.getElementById('lcd_view_camera').style.display = 'none';
              document.getElementById('lcd_view_gauges').style.display = 'flex';
            }
          }

          // 2. Synchronize Web Interactive Cockpit
          if (!isLCD) {
            // Update Feed destination switcher buttons
            if (dest === 'lcd') {
              document.getElementById('btn_target_lcd').classList.add('active');
              document.getElementById('btn_target_web').classList.remove('active');
              document.getElementById('web_active_video').style.display = 'none';
              document.getElementById('web_standby_video').style.display = 'flex';
            } else {
              document.getElementById('btn_target_web').classList.add('active');
              document.getElementById('btn_target_lcd').classList.remove('active');
              document.getElementById('web_active_video').style.display = 'block';
              document.getElementById('web_standby_video').style.display = 'none';
            }

            // Synchronize all arm panels from telemetry if user is not actively dragging or in local pulse
            if (!isUserDraggingSlider && gripperPulseTimer === null && d.s1 !== undefined && d.s2 !== undefined && d.s3 !== undefined) {
              updateAllArmPanels(d.s1, d.s2, d.s3, false);
            }

            // State & Mission control button
            document.getElementById('web_state_pill').innerText = 'STATE: ' + stateStr;
            const btnAuto = document.getElementById('btn_auto_pick');
            if (autoRunning) {
              btnAuto.innerText = '⏸ PAUSE AUTO';
              btnAuto.className = 'btn btn-red';
            } else {
              btnAuto.innerText = '▶ START AUTO PICK';
              btnAuto.className = 'btn btn-green';
            }

            const btnShowcase = document.getElementById('btn_showcase');
            if (d.showcase_running) {
              btnShowcase.innerText = '★ SHOWCASE RUNNING...';
              btnShowcase.style.background = '#854d0e';
            } else {
              btnShowcase.innerText = '★ SHOWCASE';
              btnShowcase.style.background = '#0284c7';
            }

            // YOLOE Telemetry Card Updates
            const cardObj = document.getElementById('yoloe_card_obj');
            if (cardObj) {
              if (objName !== 'NONE') {
                cardObj.innerText = `${objName} (${confVal}%)`;
                cardObj.style.color = '#38bdf8';
              } else {
                cardObj.innerText = 'Scanning for targets...';
                cardObj.style.color = '#94a3b8';
              }
            }
            const confBar = document.getElementById('yoloe_card_conf_bar');
            if (confBar) confBar.style.width = confVal + '%';

            const cardPick = document.getElementById('yoloe_card_pick');
            if (cardPick) {
              if (objName !== 'NONE') {
                cardPick.innerText = isPick ? 'READY TO GRASP' : 'NON-PICKABLE';
                cardPick.style.color = isPick ? '#4ade80' : '#f87171';
              } else {
                cardPick.innerText = '--';
                cardPick.style.color = '#94a3b8';
              }
            }

            const cardAlign = document.getElementById('yoloe_card_align');
            if (cardAlign) {
              if (objName !== 'NONE') {
                cardAlign.innerText = isAlign ? 'ALIGNED [READY]' : 'APPROACHING TARGET';
                cardAlign.style.color = isAlign ? '#4ade80' : '#fbbf24';
              } else {
                cardAlign.innerText = 'Awaiting Target';
                cardAlign.style.color = '#94a3b8';
              }
            }

            const cardErr = document.getElementById('yoloe_card_err');
            if (cardErr) {
              if (objName !== 'NONE') {
                cardErr.innerText = `dx: ${y.dx > 0 ? '+' : ''}${y.dx}px | dy: ${y.dy > 0 ? '+' : ''}${y.dy}px`;
              } else {
                cardErr.innerText = 'dx: -- px | dy: -- px';
              }
            }

            const cardLat = document.getElementById('yoloe_live_latency');
            if (cardLat && y.latency_ms > 0) {
              cardLat.innerText = `Latency: ${y.latency_ms} ms (~${y.fps} FPS)`;
            }

            const cardColor = document.getElementById('yoloe_card_color');
            const cardColorDot = document.getElementById('yoloe_card_color_dot');
            const cardColorHex = document.getElementById('yoloe_card_color_hex');
            if (cardColor) {
              if (y.color && y.color !== 'NONE' && y.color !== 'Unknown') {
                cardColor.innerText = y.color;
                if (cardColorDot) {
                  cardColorDot.style.background = y.color_hex || '#888888';
                  cardColorDot.style.boxShadow = `0 0 8px ${y.color_hex || '#888888'}66`;
                }
                if (cardColorHex) cardColorHex.innerText = `${y.color_hex || ''} (Sub-ms HSV)`;
              } else {
                cardColor.innerText = objName !== 'NONE' ? 'Analyzing...' : '--';
                if (cardColorDot) {
                  cardColorDot.style.background = '#888888';
                  cardColorDot.style.boxShadow = 'none';
                }
                if (cardColorHex) cardColorHex.innerText = 'Sub-millisecond HSV';
              }
            }

            const backendBadge = document.getElementById('yoloe_card_backend_badge');
            if (backendBadge && y.backend) {
              backendBadge.innerText = y.backend.toUpperCase();
              if (y.backend === 'LiteRT') {
                backendBadge.style.background = '#38bdf8';
                backendBadge.style.color = '#000';
              } else {
                backendBadge.style.background = '#10b981';
                backendBadge.style.color = '#000';
              }
            }

            // VLM Motion Planner UI updates
            if (d.vlm_plan) {
              updateVLMPlanUI(d.vlm_plan);
            }
            if (d.vlm_auto_advisory !== undefined) {
              const chk = document.getElementById('chk_vlm_auto_advisory');
              if (chk && document.activeElement !== chk) chk.checked = d.vlm_auto_advisory;
            }

            // Autonomous Activities telemetry sync
            if (d.activity) {
              const isMotion = (d.activity.config && d.activity.config.motion_enabled !== undefined) ? d.activity.config.motion_enabled : true;
              if (isMotion !== activityMotionEnabled) {
                activityMotionEnabled = isMotion;
                updateMotionBtnUI(activityMotionEnabled);
              }

              const actBadge = document.getElementById('act_status_badge');
              if (actBadge) {
                if (d.activity.running) {
                  let actName = (d.activity.active_activity || 'Activity').replace(/_/g, ' ').toUpperCase();
                  if (d.activity.active_activity === 'object_tracking' && d.activity.target_object) {
                    actName = `OBJECT TRACKING [${d.activity.target_object.toUpperCase()}]`;
                  } else if (d.activity.active_activity === 'color_track_and_classify' && d.activity.target_color) {
                    actName = `COLOR + CLASSIFY [${d.activity.target_color.toUpperCase()}]`;
                  } else if (d.activity.active_activity === 'color_tracking' && d.activity.target_color) {
                    actName = `COLOR TRACKING [${d.activity.target_color.toUpperCase()}]`;
                  } else if (d.activity.active_activity === 'object_sizing') {
                    const sObj = (d.activity.sizing && d.activity.sizing.object !== 'none') ? d.activity.sizing.object.toUpperCase() : 'ANY';
                    actName = `OBJECT SIZING [${sObj}]`;
                  } else if (d.activity.active_activity === 'obstacle_avoidance') {
                    const oaAct = (d.activity.obstacle && d.activity.obstacle.action) ? d.activity.obstacle.action : (d.activity.action || 'CRUISING');
                    actName = `OBSTACLE AVOIDANCE [${oaAct}]`;
                  }
                  if (!isMotion) {
                    actName += ' [DETECT ONLY]';
                  }
                  actBadge.innerText = `● ACTIVE: ${actName} (${d.activity.status || 'RUNNING'})`;
                  actBadge.style.color = isMotion ? '#10b981' : '#f59e0b';
                  actBadge.style.borderColor = isMotion ? '#10b981' : '#f59e0b';
                } else {
                  actBadge.innerText = isMotion ? 'IDLE (NO ACTIVITY)' : 'IDLE [DETECT ONLY]';
                  actBadge.style.color = '#60a5fa';
                  actBadge.style.borderColor = '#3b82f6';
                }
              }
              const ctcBadge = document.getElementById('ctc_live_class');
              if (ctcBadge) {
                if (d.activity.running && d.activity.active_activity === 'color_track_and_classify') {
                  const cls = d.activity.classified_object || 'Scanning...';
                  ctcBadge.innerText = `Identified: ${cls.toUpperCase()}`;
                  ctcBadge.style.color = '#38bdf8';
                } else {
                  ctcBadge.innerText = 'Identified: None';
                  ctcBadge.style.color = 'var(--text-muted)';
                }
              }
              // Object Sizing telemetry updates
              const sz = d.activity.sizing;
              const osObj = document.getElementById('os_live_obj');
              const osW = document.getElementById('os_live_w');
              const osWPx = document.getElementById('os_live_w_px');
              const osH = document.getElementById('os_live_h');
              const osHPx = document.getElementById('os_live_h_px');
              const osCat = document.getElementById('os_live_cat');
              const osGrip = document.getElementById('os_live_gripper');
              if (sz && d.activity.running && d.activity.active_activity === 'object_sizing' && sz.object !== 'none') {
                if (osObj) osObj.innerText = sz.object.toUpperCase();
                if (osW) osW.innerText = `${sz.width_mm} mm`;
                if (osWPx) osWPx.innerText = `(${sz.width_px} px)`;
                if (osH) osH.innerText = `${sz.height_mm} mm`;
                if (osHPx) osHPx.innerText = `(${sz.height_px} px)`;
                if (osCat) {
                  osCat.innerText = sz.size_category;
                  osCat.style.color = '#38bdf8';
                  osCat.style.borderColor = '#38bdf8';
                }
                if (osGrip) {
                  osGrip.innerText = sz.gripper_fit;
                  if (sz.gripper_fit === 'GRASPABLE') {
                    osGrip.style.color = '#10b981';
                    osGrip.style.borderColor = '#10b981';
                  } else {
                    osGrip.style.color = '#ef4444';
                    osGrip.style.borderColor = '#ef4444';
                  }
                }
              } else if (osObj) {
                osObj.innerText = 'None';
                if (osW) osW.innerText = '-- mm';
                if (osWPx) osWPx.innerText = '(-- px)';
                if (osH) osH.innerText = '-- mm';
                if (osHPx) osHPx.innerText = '(-- px)';
                if (osCat) { osCat.innerText = '--'; osCat.style.color = 'var(--text-muted)'; osCat.style.borderColor = 'var(--border)'; }
                if (osGrip) { osGrip.innerText = '--'; osGrip.style.color = 'var(--text-muted)'; osGrip.style.borderColor = 'var(--border)'; }
              }

              // Cam Obstacle Avoidance telemetry updates
              const oa = d.activity.obstacle;
              const oaActEl = document.getElementById('oa_live_action');
              const oaLEl = document.getElementById('oa_live_l');
              const oaCEl = document.getElementById('oa_live_c');
              const oaREl = document.getElementById('oa_live_r');
              const oaLSc = document.getElementById('oa_live_l_score');
              const oaCSc = document.getElementById('oa_live_c_score');
              const oaRSc = document.getElementById('oa_live_r_score');
              const oaBoxL = document.getElementById('oa_box_l');
              const oaBoxC = document.getElementById('oa_box_c');
              const oaBoxR = document.getElementById('oa_box_r');

              if (oa && d.activity.running && d.activity.active_activity === 'obstacle_avoidance') {
                if (oaActEl) {
                  oaActEl.innerText = `${oa.action || 'CRUISING'} (${oa.clear_path || 'FORWARD'})`;
                  oaActEl.style.color = (oa.action && (oa.action.startsWith('AVOID') || oa.action.startsWith('ESCAPE'))) ? '#ef4444' : '#10b981';
                }
                const updateSector = (zone, score, el, scoreEl, boxEl) => {
                  if (el) {
                    el.innerText = zone;
                    el.style.color = zone === 'BLOCKED' ? '#ef4444' : '#10b981';
                  }
                  if (scoreEl) scoreEl.innerText = `${score}%`;
                  if (boxEl) {
                    boxEl.style.borderColor = zone === 'BLOCKED' ? '#ef4444' : 'var(--border)';
                    boxEl.style.background = zone === 'BLOCKED' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(255, 255, 255, 0.03)';
                  }
                };
                updateSector(oa.left_zone || 'CLEAR', oa.left_score || 0, oaLEl, oaLSc, oaBoxL);
                updateSector(oa.center_zone || 'CLEAR', oa.center_score || 0, oaCEl, oaCSc, oaBoxC);
                updateSector(oa.right_zone || 'CLEAR', oa.right_score || 0, oaREl, oaRSc, oaBoxR);
              } else if (oaActEl) {
                oaActEl.innerText = 'IDLE';
                oaActEl.style.color = 'var(--accent)';
                if (oaLEl) { oaLEl.innerText = 'CLEAR'; oaLEl.style.color = '#10b981'; }
                if (oaCEl) { oaCEl.innerText = 'CLEAR'; oaCEl.style.color = '#10b981'; }
                if (oaREl) { oaREl.innerText = 'CLEAR'; oaREl.style.color = '#10b981'; }
                if (oaLSc) oaLSc.innerText = '0.0%';
                if (oaCSc) oaCSc.innerText = '0.0%';
                if (oaRSc) oaRSc.innerText = '0.0%';
                if (oaBoxL) { oaBoxL.style.borderColor = 'var(--border)'; oaBoxL.style.background = 'rgba(255,255,255,0.03)'; }
                if (oaBoxC) { oaBoxC.style.borderColor = 'var(--border)'; oaBoxC.style.background = 'rgba(255,255,255,0.03)'; }
                if (oaBoxR) { oaBoxR.style.borderColor = 'var(--border)'; oaBoxR.style.background = 'rgba(255,255,255,0.03)'; }
              }

            }
          }
          if (d.servo_cal) {
            const serverMtime = d.servo_cal_mtime || 0;
            if (lastServerCalMtime === 0) {
              lastServerCalMtime = serverMtime;
            } else if (serverMtime > lastServerCalMtime && !isCalDirty) {
              lastServerCalMtime = serverMtime;
              applyServoCalToUI(d.servo_cal, true);
              showToast('msg_servos', '🔄 Calibration synchronized from disk.');
            }
          }
        })
        .catch(() => {});
    }, 300);

    // Motor Teleop (Web Cockpit)
    function startDrive(dir) {
      fetch('/api/drive_dir', { method: 'POST', body: JSON.stringify({ dir: dir }) });
    }
    function stopDrive() {
      fetch('/api/stop', { method: 'POST' });
    }
    function sendStop() {
      fetch('/api/stop', { method: 'POST' });
      showToast('msg_teleop', 'Robot Motors Stopped.');
    }
    function sendNudge(dir) {
      fetch('/api/nudge', { method: 'POST', body: JSON.stringify({ dir: dir }) });
      showToast('msg_motors', `Nudged ${dir}`);
    }

    // Keyboard WASD Controls (Active in Web Cockpit)
    const activeKeys = {};
    window.addEventListener('keydown', e => {
      if (isLCD) return;
      const k = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', ' '].includes(k) && !activeKeys[k]) {
        activeKeys[k] = true;
        if (k === 'w') startDrive('F');
        else if (k === 's') startDrive('B');
        else if (k === 'a') startDrive('L');
        else if (k === 'd') startDrive('R');
        else if (k === ' ') sendStop();
      }
    });
    window.addEventListener('keyup', e => {
      if (isLCD) return;
      const k = e.key.toLowerCase();
      if (['w', 'a', 's', 'd'].includes(k)) {
        delete activeKeys[k];
        if (Object.keys(activeKeys).length === 0) stopDrive();
      }
    });

    // =========================================================================
    // UNIFIED SERVO ARM SUBSYSTEM & 1.0s GRIPPER PULSE
    // Synchronizes ALL panels that control or display the servo arm:
    // - Tab 1 Cockpit Readouts & Macro Buttons
    // - Tab 4 Live Sliders, Displays & Calibration Inputs
    // - Cockpit HUD & LCD Gauges
    // =========================================================================

    let gripperPulseTimer = null;
    let gripperBaseAngle = null;
    let gripperPulseOffset = 0;
    let gripperDebounceTimer = null;
    let isUserDraggingSlider = false;

    function updateAllArmPanels(s1, s2, s3, isPulsing = null) {
      s1 = parseInt(s1);
      s2 = parseInt(s2);
      s3 = parseInt(s3);

      // 1. Tab 4 (Robotic Arm Alternating) Sliders & Value Labels
      const elS1 = document.getElementById('live_s1');
      const elS2 = document.getElementById('live_s2');
      const elS3 = document.getElementById('live_s3');
      if (elS1 && document.activeElement !== elS1) elS1.value = s1;
      if (elS2 && document.activeElement !== elS2) elS2.value = s2;
      if (elS3 && document.activeElement !== elS3) elS3.value = s3;

      const valS1 = document.getElementById('val_live_s1');
      const valS2 = document.getElementById('val_live_s2');
      const valS3 = document.getElementById('val_live_s3');
      if (valS1) valS1.innerText = s1 + '°';
      if (valS2) valS2.innerText = s2 + '°';
      if (valS3) valS3.innerText = s3 + '°';

      // 2. Tab 1 (Main Cockpit) Arm Readouts
      const t1S1 = document.getElementById('tab1_arm_s1');
      const t1S2 = document.getElementById('tab1_arm_s2');
      const t1S3 = document.getElementById('tab1_arm_s3');
      if (t1S1) t1S1.innerText = s1 + '°';
      if (t1S2) t1S2.innerText = s2 + '°';
      if (t1S3) t1S3.innerText = s3 + '°';

      // 3. Pulse Badges across both panels (Tab 1 and Tab 4)
      const pulseActive = (isPulsing !== null) ? isPulsing : (gripperPulseTimer !== null);
      const bTab4 = document.getElementById('gripper_pulse_badge');
      const bTab1 = document.getElementById('tab1_pulse_badge');
      if (bTab4) bTab4.style.display = pulseActive ? 'inline-block' : 'none';
      if (bTab1) bTab1.style.display = pulseActive ? 'inline-block' : 'none';

      // 4. Cockpit LCD HUD & Gauges
      const hudArm = document.getElementById('lcd_hud_arm');
      const gArm = document.getElementById('lcd_g_arm');
      if (hudArm) hudArm.innerText = `${s1}° / ${s2}° / ${s3}°`;
      if (gArm) gArm.innerText = `S1: ${s1}° | S2: ${s2}° | S3: ${s3}°`;
    }

    function stopGripperPulse() {
      if (gripperPulseTimer) {
        clearInterval(gripperPulseTimer);
        gripperPulseTimer = null;
      }
      if (gripperDebounceTimer) {
        clearTimeout(gripperDebounceTimer);
        gripperDebounceTimer = null;
      }
      if (gripperBaseAngle !== null && gripperPulseOffset !== 0) {
        gripperPulseOffset = 0;
        const curS1 = parseInt(document.getElementById('live_s1').value) || 93;
        const curS2 = parseInt(document.getElementById('live_s2').value) || 45;
        updateAllArmPanels(curS1, curS2, gripperBaseAngle, false);
      } else {
        const curS1 = parseInt(document.getElementById('live_s1').value) || 93;
        const curS2 = parseInt(document.getElementById('live_s2').value) || 45;
        const curS3 = parseInt(document.getElementById('live_s3').value) || 93;
        updateAllArmPanels(curS1, curS2, curS3, false);
      }
    }

    function syncLiveSliderBounds(cal) {
      if (!cal) return;
      const elS1 = document.getElementById('live_s1');
      if (elS1) {
        if (cal.s1_min !== undefined) elS1.min = cal.s1_min;
        if (cal.s1_max !== undefined) elS1.max = cal.s1_max;
        const s1Min = parseInt(elS1.min) || 60;
        const s1Max = parseInt(elS1.max) || 170;
        const curS1 = parseInt(elS1.value) || s1Min;
        if (curS1 < s1Min) elS1.value = s1Min;
        if (curS1 > s1Max) elS1.value = s1Max;
      }

      const elS2 = document.getElementById('live_s2');
      if (elS2) {
        if (cal.s2_min !== undefined) elS2.min = cal.s2_min;
        if (cal.s2_max !== undefined) elS2.max = cal.s2_max;
        const s2Min = parseInt(elS2.min) || 0;
        const s2Max = parseInt(elS2.max) || 45;
        const curS2 = parseInt(elS2.value) || s2Min;
        if (curS2 < s2Min) elS2.value = s2Min;
        if (curS2 > s2Max) elS2.value = s2Max;
      }

      const elS3 = document.getElementById('live_s3');
      if (elS3) {
        const s3CloseEl = document.getElementById('s3_close');
        const s3Close = (cal.s3_close !== undefined) ? parseInt(cal.s3_close) : (s3CloseEl ? (parseInt(s3CloseEl.value) || 40) : 40);
        const s3OpenEl = document.getElementById('s3_open');
        const s3Open = (cal.s3_open !== undefined) ? parseInt(cal.s3_open) : (s3OpenEl ? (parseInt(s3OpenEl.value) || 170) : 170);
        const s3MaxCfg = (cal.s3_max !== undefined) ? parseInt(cal.s3_max) : 170;
        const liveMin = Math.min(s3Close, s3Open);
        const liveMax = Math.max(s3Close, s3Open, s3MaxCfg);
        elS3.min = liveMin;
        elS3.max = liveMax;
        const curS3 = parseInt(elS3.value) || liveMin;
        if (curS3 < liveMin) elS3.value = liveMin;
        if (curS3 > liveMax) elS3.value = liveMax;
        if (gripperBaseAngle !== null) {
          gripperBaseAngle = Math.max(liveMin, Math.min(liveMax, gripperBaseAngle));
        }
      }
    }

    function setGripperAngleOnly(angle) {
      stopGripperPulse();
      const elS3 = document.getElementById('live_s3');
      const minS3 = (elS3 && elS3.min) ? (parseInt(elS3.min) || 40) : 40;
      const maxS3 = (elS3 && elS3.max) ? (parseInt(elS3.max) || 170) : 170;
      gripperBaseAngle = Math.max(minS3, Math.min(maxS3, parseInt(angle)));
      gripperPulseOffset = 0;

      const s1 = parseInt(document.getElementById('live_s1').value) || 93;
      const s2 = parseInt(document.getElementById('live_s2').value) || 45;

      // Update all panels immediately to the base target angle
      updateAllArmPanels(s1, s2, gripperBaseAngle, true);

      // Send target angle to servo
      fetch('/api/servo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ s1: s1, s2: s2, s3: gripperBaseAngle, pulse: false })
      }).catch(() => {});

      // Every 1.0 second (1000 ms): manually adjust slider adding and subtracting 1 degree, and send change to servo
      gripperPulseTimer = setInterval(() => {
        if (gripperBaseAngle === null) return;
        const curS1 = parseInt(document.getElementById('live_s1').value) || 93;
        const curS2 = parseInt(document.getElementById('live_s2').value) || 45;
        const el = document.getElementById('live_s3');
        const s3Min = (el && el.min) ? (parseInt(el.min) || 40) : 40;
        const s3Max = (el && el.max) ? (parseInt(el.max) || 170) : 170;

        if (gripperPulseOffset === 0) {
          // If at or above max limit, pulse downward; otherwise pulse upward
          if (gripperBaseAngle >= s3Max) {
            gripperPulseOffset = -1;
          } else if (gripperBaseAngle <= s3Min) {
            gripperPulseOffset = 1;
          } else {
            gripperPulseOffset = 1;
          }
        } else {
          // Phase 2 (+2.0s): return to base angle
          gripperPulseOffset = 0;
        }

        const currentAngle = Math.max(s3Min, Math.min(s3Max, gripperBaseAngle + gripperPulseOffset));

        // Manually adjust the slider and text in ALL panels
        updateAllArmPanels(curS1, curS2, currentAngle, true);

        // Send that data change to the servo
        fetch('/api/servo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ s1: curS1, s2: curS2, s3: currentAngle, pulse: false })
        }).catch(() => {});
      }, 1000);
    }

    function onGripperSliderInput() {
      isUserDraggingSlider = true;
      const el = document.getElementById('live_s3');
      const minS3 = (el && el.min) ? (parseInt(el.min) || 40) : 40;
      const maxS3 = (el && el.max) ? (parseInt(el.max) || 170) : 170;
      const val = Math.max(minS3, Math.min(maxS3, parseInt(el.value)));
      el.value = val;
      const s1 = parseInt(document.getElementById('live_s1').value) || 93;
      const s2 = parseInt(document.getElementById('live_s2').value) || 45;
      updateAllArmPanels(s1, s2, val, false);

      if (gripperPulseTimer) {
        clearInterval(gripperPulseTimer);
        gripperPulseTimer = null;
      }
      if (gripperDebounceTimer) {
        clearTimeout(gripperDebounceTimer);
      }

      // Debounce: once user pauses for 120ms, initiate 1.0s gripper pulse
      gripperDebounceTimer = setTimeout(() => {
        isUserDraggingSlider = false;
        setGripperAngleOnly(val);
      }, 120);
    }

    function onGripperSliderChange() {
      if (gripperDebounceTimer) {
        clearTimeout(gripperDebounceTimer);
        gripperDebounceTimer = null;
      }
      isUserDraggingSlider = false;
      const el = document.getElementById('live_s3');
      const minS3 = (el && el.min) ? (parseInt(el.min) || 40) : 40;
      const maxS3 = (el && el.max) ? (parseInt(el.max) || 170) : 170;
      const val = Math.max(minS3, Math.min(maxS3, parseInt(el.value)));
      el.value = val;
      setGripperAngleOnly(val);
    }

    // Macros
    function sendMacro(name) {
      if (name === 'open') {
        const openVal = parseInt(document.getElementById('s3_open').value) || 170;
        setGripperAngleOnly(openVal);
        showToast('msg_teleop', `Gripper OPEN (${openVal}°)`);
        return;
      }
      if (name === 'close') {
        const closeVal = parseInt(document.getElementById('s3_close').value) || 40;
        setGripperAngleOnly(closeVal);
        showToast('msg_teleop', `Gripper CLOSE (${closeVal}°)`);
        return;
      }
      // If moving full arm joints (stow, center, down): stop gripper pulse
      stopGripperPulse();
      fetch('/api/macro', { method: 'POST', body: JSON.stringify({ name: name }) })
        .then(r => r.json())
        .then(d => {
          showToast('msg_teleop', `Pose: ${name.toUpperCase()} executed.`);
          if (name === 'stow' || name === 'up') {
            const s1 = parseInt(document.getElementById('s1_stow').value) || 95;
            const s2 = parseInt(document.getElementById('s2_stow').value) || 44;
            const curS3 = (gripperBaseAngle !== null) ? gripperBaseAngle : (parseInt(document.getElementById('live_s3').value) || 93);
            updateAllArmPanels(s1, s2, curS3, false);
          } else if (name === 'down') {
            const s1 = parseInt(document.getElementById('s1_down').value) || 168;
            const s2 = parseInt(document.getElementById('s2_down').value) || 2;
            const curS3 = (gripperBaseAngle !== null) ? gripperBaseAngle : (parseInt(document.getElementById('live_s3').value) || 93);
            updateAllArmPanels(s1, s2, curS3, false);
          } else if (name === 'center') {
            const s1 = parseInt(document.getElementById('s1_center').value) || 93;
            const s2 = parseInt(document.getElementById('s2_center').value) || 45;
            const s3 = parseInt(document.getElementById('s3_center').value) || 93;
            updateAllArmPanels(s1, s2, s3, false);
          }
        });
    }

    // Arm live sliders
    let armDebounce = null;
    function moveArmLive(joint = null) {
      if (joint === 's3') {
        onGripperSliderInput();
        return;
      }
      if (joint === 's1' || joint === 's2') {
        stopGripperPulse();
      }
      const elS1 = document.getElementById('live_s1');
      const elS2 = document.getElementById('live_s2');
      const elS3 = document.getElementById('live_s3');
      const s1Min = (elS1 && elS1.min) ? (parseInt(elS1.min) || 60) : 60;
      const s1Max = (elS1 && elS1.max) ? (parseInt(elS1.max) || 170) : 170;
      const s2Min = (elS2 && elS2.min) ? (parseInt(elS2.min) || 0) : 0;
      const s2Max = (elS2 && elS2.max) ? (parseInt(elS2.max) || 45) : 45;
      const s3Min = (elS3 && elS3.min) ? (parseInt(elS3.min) || 40) : 40;
      const s3Max = (elS3 && elS3.max) ? (parseInt(elS3.max) || 170) : 170;

      const s1 = Math.max(s1Min, Math.min(s1Max, parseInt(elS1.value) || s1Min));
      const s2 = Math.max(s2Min, Math.min(s2Max, parseInt(elS2.value) || s2Min));
      const s3Raw = (gripperBaseAngle !== null) ? gripperBaseAngle : (parseInt(elS3.value) || s3Min);
      const s3 = Math.max(s3Min, Math.min(s3Max, s3Raw));
      updateAllArmPanels(s1, s2, s3, false);

      clearTimeout(armDebounce);
      armDebounce = setTimeout(() => {
        fetch('/api/servo', { method: 'POST', body: JSON.stringify({ s1: s1, s2: s2, s3: s3 }) });
      }, 50);
    }

    let lastServerCalMtime = 0;
    let isCalDirty = false;

    function updateServoCal(field) {
      isCalDirty = true;
      const el = document.getElementById(field);
      if (!el) return;
      const val = parseInt(el.value);
      const valEl = document.getElementById('val_' + field);
      if (valEl) valEl.innerText = val + '°';
      if (field === 's3_open') {
        const lbl = document.getElementById('lbl_macro_open');
        if (lbl) lbl.innerText = val;
        const lbl1 = document.getElementById('lbl_tab1_open');
        if (lbl1) lbl1.innerText = val;
        syncLiveSliderBounds({ s3_open: val });
      } else if (field === 's3_close') {
        const lbl = document.getElementById('lbl_macro_close');
        if (lbl) lbl.innerText = val;
        const lbl1 = document.getElementById('lbl_tab1_close');
        if (lbl1) lbl1.innerText = val;
        syncLiveSliderBounds({ s3_close: val });
      } else if (field === 's1_stow' || field === 's1_down') {
        const s1StowEl = document.getElementById('s1_stow');
        const s1DownEl = document.getElementById('s1_down');
        const s1_stow = s1StowEl ? (parseInt(s1StowEl.value) || 93) : 93;
        const s1_down = s1DownEl ? (parseInt(s1DownEl.value) || 170) : 170;
        syncLiveSliderBounds({ s1_min: Math.min(s1_stow, s1_down), s1_max: Math.max(s1_stow, s1_down) });
      } else if (field === 's2_stow' || field === 's2_down') {
        const s2StowEl = document.getElementById('s2_stow');
        const s2DownEl = document.getElementById('s2_down');
        const s2_stow = s2StowEl ? (parseInt(s2StowEl.value) || 45) : 45;
        const s2_down = s2DownEl ? (parseInt(s2DownEl.value) || 0) : 0;
        syncLiveSliderBounds({ s2_min: Math.min(s2_stow, s2_down), s2_max: Math.max(s2_stow, s2_down) });
      }
    }

    function updateMotorVal(field) {
      const val = document.getElementById(field).value;
      document.getElementById('val_' + field).innerText = val;
    }

    function applyServoCalToUI(cal, force = false) {
      if (!cal) return;
      if (isCalDirty && !force) return;
      const activeId = document.activeElement ? document.activeElement.id : null;
      for (const [k, v] of Object.entries(cal)) {
        if (k.startsWith('_')) continue;
        if (k === activeId) continue;
        const el = document.getElementById(k);
        const valEl = document.getElementById('val_' + k);
        if (el) el.value = v;
        if (valEl) valEl.innerText = v + '°';
      }
      const lblOpen = document.getElementById('lbl_macro_open');
      if (lblOpen && cal.s3_open !== undefined) lblOpen.innerText = cal.s3_open;
      const lblTab1Open = document.getElementById('lbl_tab1_open');
      if (lblTab1Open && cal.s3_open !== undefined) lblTab1Open.innerText = cal.s3_open;

      const lblClose = document.getElementById('lbl_macro_close');
      if (lblClose && cal.s3_close !== undefined) lblClose.innerText = cal.s3_close;
      const lblTab1Close = document.getElementById('lbl_tab1_close');
      if (lblTab1Close && cal.s3_close !== undefined) lblTab1Close.innerText = cal.s3_close;

      syncLiveSliderBounds(cal);
    }

    function captureLiveAsStow() {
      const s1 = document.getElementById('live_s1').value;
      const s2 = document.getElementById('live_s2').value;
      document.getElementById('s1_stow').value = s1;
      document.getElementById('val_s1_stow').innerText = s1 + '°';
      document.getElementById('s2_stow').value = s2;
      document.getElementById('val_s2_stow').innerText = s2 + '°';
      isCalDirty = true;
      showToast('msg_servos', `Captured live S1 (${s1}°) and S2 (${s2}°) as STOW.`);
    }

    function captureLiveAsDown() {
      const s1 = document.getElementById('live_s1').value;
      const s2 = document.getElementById('live_s2').value;
      document.getElementById('s1_down').value = s1;
      document.getElementById('val_s1_down').innerText = s1 + '°';
      document.getElementById('s2_down').value = s2;
      document.getElementById('val_s2_down').innerText = s2 + '°';
      isCalDirty = true;
      showToast('msg_servos', `Captured live S1 (${s1}°) and S2 (${s2}°) as DOWN.`);
    }

    function captureLiveAsCenter() {
      const s1 = document.getElementById('live_s1').value;
      const s2 = document.getElementById('live_s2').value;
      const s3 = (gripperBaseAngle !== null) ? gripperBaseAngle : document.getElementById('live_s3').value;
      document.getElementById('s1_center').value = s1;
      document.getElementById('val_s1_center').innerText = s1 + '°';
      document.getElementById('s2_center').value = s2;
      document.getElementById('val_s2_center').innerText = s2 + '°';
      document.getElementById('s3_center').value = s3;
      document.getElementById('val_s3_center').innerText = s3 + '°';
      isCalDirty = true;
      showToast('msg_servos', `Captured live all (${s1}°, ${s2}°, ${s3}°) as CENTER.`);
    }

    function captureLiveAsOpen() {
      const s3 = (gripperBaseAngle !== null) ? gripperBaseAngle : document.getElementById('live_s3').value;
      document.getElementById('s3_open').value = s3;
      document.getElementById('val_s3_open').innerText = s3 + '°';
      const lbl = document.getElementById('lbl_macro_open');
      if (lbl) lbl.innerText = s3;
      const lbl1 = document.getElementById('lbl_tab1_open');
      if (lbl1) lbl1.innerText = s3;
      isCalDirty = true;
      showToast('msg_servos', `Captured live S3 (${s3}°) as OPEN.`);
    }

    function captureLiveAsClose() {
      const s3 = (gripperBaseAngle !== null) ? gripperBaseAngle : document.getElementById('live_s3').value;
      document.getElementById('s3_close').value = s3;
      document.getElementById('val_s3_close').innerText = s3 + '°';
      const lbl = document.getElementById('lbl_macro_close');
      if (lbl) lbl.innerText = s3;
      const lbl1 = document.getElementById('lbl_tab1_close');
      if (lbl1) lbl1.innerText = s3;
      isCalDirty = true;
      showToast('msg_servos', `Captured live S3 (${s3}°) as CLOSE.`);
    }

    function captureCurrentPositions() {
      captureLiveAsStow();
      captureLiveAsOpen();
      isCalDirty = true;
      showToast('msg_servos', 'Captured live angles into stow/open calibration.');
    }

    function saveServos() {
      const getVal = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? parseInt(el.value) : fallback;
      };
      const payload = {
        s1_min: getVal('s1_min', 60),
        s1_max: getVal('s1_max', 170),
        s1_stow: getVal('s1_stow', 93),
        s1_down: getVal('s1_down', 170),
        s1_center: getVal('s1_center', 93),
        s2_min: getVal('s2_min', 0),
        s2_max: getVal('s2_max', 45),
        s2_stow: getVal('s2_stow', 45),
        s2_down: getVal('s2_down', 0),
        s2_center: getVal('s2_center', 45),
        s3_min: getVal('s3_min', 70),
        s3_max: getVal('s3_max', 110),
        s3_open: getVal('s3_open', 110),
        s3_close: getVal('s3_close', 70),
        s3_center: getVal('s3_center', 110),
      };
      fetch('/api/save_servos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then(r => r.json())
        .then(d => {
          isCalDirty = false;
          if (d.servo_cal) {
            lastServerCalMtime = d.servo_cal_mtime || (d.servo_cal._mtime || 0);
            applyServoCalToUI(d.servo_cal, true);
          }
          showToast('msg_servos', '✅ Servos saved to config/robot_config.yaml');
        })
        .catch(e => showToast('msg_servos', 'Error saving servos: ' + e));
    }

    function saveMotors() {
      const payload = {
        base_speed: parseInt(document.getElementById('base_speed').value),
        turn_speed: parseInt(document.getElementById('turn_speed').value),
        nudge_pwm: parseInt(document.getElementById('nudge_pwm').value),
        trim_offset: parseInt(document.getElementById('trim_offset').value),
      };
      fetch('/api/save_motors', { method: 'POST', body: JSON.stringify(payload) })
        .then(r => r.json())
        .then(d => showToast('msg_motors', '✅ Motors saved to config/robot_config.yaml'));
    }

    function handleVideoClick(e) {
      const rect = e.target.getBoundingClientRect();
      const x = Math.round((e.clientX - rect.left) / rect.width * 640);
      const y = Math.round((e.clientY - rect.top) / rect.height * 480);
      if (document.getElementById('cx')) {
        document.getElementById('cx').value = x;
        document.getElementById('cy').value = y;
        document.getElementById('val_cx').innerText = x;
        document.getElementById('val_cy').innerText = y;
      }
      fetch('/api/sweet_spot', { method: 'POST', body: JSON.stringify({ center_x: x, center_y: y, box_width: current_spot.box_width, box_height: current_spot.box_height }) });
    }

    function updateSpot() {
      const cx = parseInt(document.getElementById('cx').value);
      const cy = parseInt(document.getElementById('cy').value);
      const bw = parseInt(document.getElementById('bw').value);
      const bh = parseInt(document.getElementById('bh').value);
      document.getElementById('val_cx').innerText = cx;
      document.getElementById('val_cy').innerText = cy;
      document.getElementById('val_bw').innerText = bw;
      document.getElementById('val_bh').innerText = bh;
      fetch('/api/sweet_spot', { method: 'POST', body: JSON.stringify({ center_x: cx, center_y: cy, box_width: bw, box_height: bh }) });
    }

    function saveVision() {
      fetch('/api/save_vision', { method: 'POST' })
        .then(r => r.json())
        .then(d => showToast('msg_vision', '✅ Sweet spot saved to config/vision_config.yaml'));
    }

    function setYoloePreset(preset) {
      const el = document.getElementById('yoloe_classes');
      if (preset === 'all_objects') {
        el.value = 'bottle, cup, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, backpack, umbrella, handbag, suitcase, frisbee, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, book, clock, vase, scissors, teddy bear, toothbrush, cell phone, remote, keyboard, mouse, laptop, fork, knife, spoon, wine glass';
      } else if (preset === 'containers') {
        el.value = 'bottle, cup, bowl, wine glass, vase';
      } else if (preset === 'office') {
        el.value = 'book, laptop, mouse, keyboard, remote, cell phone, scissors, clock, backpack, suitcase, handbag';
      } else if (preset === 'all_trash') {
        el.value = 'bottle, cup, bowl, banana, apple, sandwich, orange, donut, cake, fork, knife, spoon, wine glass';
      }
      showToast('msg_yoloe', 'Pickable preset populated. Click Apply Live to activate.');
    }

    function saveYoloeSettings(persist) {
      const model = document.getElementById('yoloe_sel_model').value;
      const conf = parseFloat(document.getElementById('yoloe_conf').value);
      const imgsz = parseInt(document.getElementById('yoloe_imgsz').value);
      const classes = document.getElementById('yoloe_classes').value;

      fetch('/api/yoloe/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: model,
          confidence_threshold: conf,
          imgsz: imgsz,
          pickable_classes: classes,
          save_yaml: persist
        })
      })
      .then(r => r.json())
      .then(d => {
        const msg = persist ? '✅ Vision settings committed & saved to YAML!' : '⚡ Vision parameters updated in memory!';
        showToast('msg_yoloe', msg);
        if (document.getElementById('yoloe_card_model')) {
          document.getElementById('yoloe_card_model').innerText = model;
        }
        if (document.getElementById('yoloe_card_imgsz')) {
          document.getElementById('yoloe_card_imgsz').innerText = `Res: ${imgsz}x${imgsz} | Conf: ${conf}`;
        }
      })
      .catch(e => showToast('msg_yoloe', 'Error saving YOLOE config: ' + e));
    }

    function reconnectBT() {
      fetch('/api/reconnect_bt', { method: 'POST' })
        .then(r => r.json())
        .then(d => showToast('msg_teleop', d.message));
    }

    function restartWebService() {
      if (confirm('Restart web service now?')) {
        fetch('/api/restart_service', { method: 'POST' })
          .then(r => r.json())
          .then(d => showToast('msg_teleop', '⚡ Restarting web service... page will reload.'))
          .catch(() => {});
      }
    }

    function toggleCamExpand() {
      const camCard = document.getElementById('cam_card');
      const btn = document.getElementById('btn_cam_expand');
      if (camCard) {
        camCard.classList.toggle('wide-cam-card');
        const isWide = camCard.classList.contains('wide-cam-card');
        btn.innerText = isWide ? '⛶ Normal Width' : '⛶ Expand Width';
      }
    }

    let selectedColor = '__CT_COLOR__' || 'Red';
    function selectTrackingColor(color, btn) {
      selectedColor = color || 'Red';
      const container = document.getElementById('ct_color_pills');
      if (container) {
        container.querySelectorAll('.color-pill').forEach(b => b.classList.remove('active'));
      }
      if (btn) {
        btn.classList.add('active');
      } else {
        const activeBtn = document.getElementById('cp_' + selectedColor);
        if (activeBtn) activeBtn.classList.add('active');
      }
    }

    let selectedClassifyColor = '__CTC_COLOR__' || 'Red';
    function selectClassifyColor(color, btn) {
      selectedClassifyColor = color || 'Red';
      const container = document.getElementById('ctc_color_pills');
      if (container) {
        container.querySelectorAll('.color-pill').forEach(b => b.classList.remove('active'));
      }
      if (btn) {
        btn.classList.add('active');
      } else {
        const activeBtn = document.getElementById('cp_ctc_' + selectedClassifyColor);
        if (activeBtn) activeBtn.classList.add('active');
      }
    }

    function onSelectTrackingObject(val) {
      if (val === 'custom') {
        const inp = document.getElementById('ot_target_input');
        if (inp) inp.focus();
        return;
      }
      setTrackingObjectPreset(val, null);
    }

    function setTrackingObjectPreset(objName, chipEl) {
      const clean = (objName || 'bottle').trim().toLowerCase();
      const container = document.getElementById('ot_chips');
      if (container) {
        container.querySelectorAll('.target-chip').forEach(c => c.classList.remove('active'));
        if (!chipEl) {
          chipEl = container.querySelector(`.target-chip[data-obj="${clean}"]`);
        }
      }
      if (chipEl) chipEl.classList.add('active');
      const inp = document.getElementById('ot_target_input');
      if (inp) inp.value = clean;

      const sel = document.getElementById('ot_target_select');
      if (sel) {
        let matched = false;
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === clean) {
            sel.selectedIndex = i;
            matched = true;
            break;
          }
        }
        if (!matched) sel.value = 'custom';
      }
    }

    function onCustomObjectInput(val) {
      const clean = (val || '').trim().toLowerCase();
      const container = document.getElementById('ot_chips');
      if (container) {
        container.querySelectorAll('.target-chip').forEach(c => {
          if (c.getAttribute('data-obj') === clean) {
            c.classList.add('active');
          } else {
            c.classList.remove('active');
          }
        });
      }
      const sel = document.getElementById('ot_target_select');
      if (sel) {
        let matched = false;
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === clean) {
            sel.selectedIndex = i;
            matched = true;
            break;
          }
        }
        if (!matched) sel.value = 'custom';
      }
    }

    function startPersonFollower() {
      const dist = parseInt(document.getElementById('pf_dist').value) / 100.0;
      const speed = parseInt(document.getElementById('pf_speed').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'person_follower',
          params: { target_dist_ratio: dist, follow_speed: speed }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', '✅ Person Follower started!');
        } else {
          showToast('msg_activities', '❌ Could not start Person Follower: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    function startColorTracking() {
      const area = parseInt(document.getElementById('ct_area').value) / 100.0;
      const speed = parseInt(document.getElementById('ct_speed').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'color_tracking',
          params: { target_color: selectedColor, target_area_ratio: area, follow_speed: speed }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', `✅ Color Tracking started for ${selectedColor}!`);
        } else {
          showToast('msg_activities', '❌ Could not start Color Tracking: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    function startObjectTracking() {
      const inp = document.getElementById('ot_target_input');
      const targetObj = (inp ? inp.value : 'bottle').trim().toLowerCase() || 'bottle';
      const dist = parseInt(document.getElementById('ot_dist').value) / 100.0;
      const speed = parseInt(document.getElementById('ot_speed').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'object_tracking',
          params: { target_object: targetObj, object_target_height_ratio: dist, follow_speed: speed }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', `✅ Object Tracking started for "${targetObj}"!`);
        } else {
          showToast('msg_activities', '❌ Could not start Object Tracking: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    function startColorTrackAndClassify() {
      const area = parseInt(document.getElementById('ctc_area').value) / 100.0;
      const speed = parseInt(document.getElementById('ctc_speed').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'color_track_and_classify',
          params: { target_color: selectedClassifyColor, target_area_ratio: area, follow_speed: speed }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', `✅ Color Tracking + Classification started for ${selectedClassifyColor}!`);
        } else {
          showToast('msg_activities', '❌ Could not start Color Tracking + Classify: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    
    function onSelectSizingObject(val) {
      if (val === 'custom') {
        const inp = document.getElementById('os_target_input');
        if (inp) inp.focus();
        return;
      }
      setSizingObjectPreset(val, null);
    }

    function setSizingObjectPreset(objName, chipEl) {
      const clean = (objName || 'any').trim().toLowerCase();
      const container = document.getElementById('os_chips');
      if (container) {
        container.querySelectorAll('.target-chip').forEach(c => c.classList.remove('active'));
        if (!chipEl) {
          chipEl = container.querySelector(`.target-chip[data-obj="${clean}"]`);
        }
      }
      if (chipEl) chipEl.classList.add('active');
      const inp = document.getElementById('os_target_input');
      if (inp) inp.value = clean;

      const sel = document.getElementById('os_target_select');
      if (sel) {
        let matched = false;
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === clean) {
            sel.selectedIndex = i;
            matched = true;
            break;
          }
        }
        if (!matched) sel.value = 'custom';
      }
    }

    function onCustomSizingInput(val) {
      const clean = (val || '').trim().toLowerCase();
      const container = document.getElementById('os_chips');
      if (container) {
        container.querySelectorAll('.target-chip').forEach(c => {
          if (c.getAttribute('data-obj') === clean) {
            c.classList.add('active');
          } else {
            c.classList.remove('active');
          }
        });
      }
      const sel = document.getElementById('os_target_select');
      if (sel) {
        let matched = false;
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === clean) {
            sel.selectedIndex = i;
            matched = true;
            break;
          }
        }
        if (!matched) sel.value = 'custom';
      }
    }

    function startObjectSizing() {
      const inp = document.getElementById('os_target_input');
      const targetObj = (inp ? inp.value : 'any').trim().toLowerCase() || 'any';
      const dist = parseInt(document.getElementById('os_dist').value) / 100.0;
      const speed = parseInt(document.getElementById('os_speed').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'object_sizing',
          params: { sizing_target_object: targetObj, sizing_target_height_ratio: dist, follow_speed: speed }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', '✅ Object Sizing started for ' + targetObj + '!');
        } else {
          showToast('msg_activities', '❌ Could not start Object Sizing: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    function startObstacleAvoidance() {
      const thresh = parseInt(document.getElementById('oa_thresh').value) / 100.0;
      const cruiseSpeed = parseInt(document.getElementById('oa_cruise').value);
      const turnSpeed = parseInt(document.getElementById('oa_turn').value);
      fetch('/api/activities/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          activity: 'obstacle_avoidance',
          params: {
            obstacle_threshold: thresh,
            obstacle_cruise_speed: cruiseSpeed,
            obstacle_turn_speed: turnSpeed
          }
        })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', '✅ Cam Obstacle Avoidance started!');
        } else {
          showToast('msg_activities', '❌ Could not start Obstacle Avoidance: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error starting activity: ' + e));
    }

    let activityMotionEnabled = true;

    function toggleActivityMotion() {
      activityMotionEnabled = !activityMotionEnabled;
      updateMotionBtnUI(activityMotionEnabled);
      fetch('/api/activities/toggle_motion', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ enabled: activityMotionEnabled })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          activityMotionEnabled = d.motion_enabled;
          updateMotionBtnUI(activityMotionEnabled);
          showToast('msg_activities', activityMotionEnabled ? '🚗 Autonomous Drive Mode (Pursuit Enabled)' : '🎯 Stationary Track Mode Enabled (Sideways Yaw + Arm IK Active)');
        }
      })
      .catch(e => showToast('msg_activities', 'Error toggling motion: ' + e));
    }

    function updateMotionBtnUI(enabled) {
      const btn = document.getElementById('btn_toggle_motion');
      if (!btn) return;
      if (enabled) {
        btn.innerHTML = '🚗 AUTONOMOUS DRIVE (PURSUIT)';
        btn.style.background = '#2563eb';
        btn.style.color = '#fff';
      } else {
        btn.innerHTML = '🎯 STATIONARY TRACK (YAW + ARM IK)';
        btn.style.background = '#d97706';
        btn.style.color = '#fff';
      }
    }

    function stopAnyActivity() {
      fetch('/api/activities/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      })
      .then(r => r.json())
      .then(d => {
        showToast('msg_activities', '■ Activity halted.');
      })
      .catch(e => showToast('msg_activities', 'Error halting activity: ' + e));
    }

    function saveActivitySettings() {
      const getVal = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? parseInt(el.value) : fallback;
      };
      const getStr = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? el.value.trim() : fallback;
      };
      const payload = {
        motion_enabled: activityMotionEnabled,
        pf_dist: getVal('pf_dist', 45),
        pf_speed: getVal('pf_speed', 210),
        ct_color: selectedColor || 'Red',
        ct_area: getVal('ct_area', 8),
        ct_speed: getVal('ct_speed', 210),
        ot_target: getStr('ot_target_input', 'bottle') || 'bottle',
        ot_dist: getVal('ot_dist', 35),
        ot_speed: getVal('ot_speed', 210),
        ctc_color: selectedClassifyColor || 'Red',
        ctc_area: getVal('ctc_area', 8),
        ctc_speed: getVal('ctc_speed', 210),
        os_target: getStr('os_target_input', 'any') || 'any',
        os_dist: getVal('os_dist', 35),
        os_speed: getVal('os_speed', 210),
        oa_thresh: getVal('oa_thresh', 22),
        oa_cruise: getVal('oa_cruise', 205),
        oa_turn: getVal('oa_turn', 225)
      };
      fetch('/api/activities/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_activities', '✅ Activity settings saved to config/vision_config.yaml');
        } else {
          showToast('msg_activities', '❌ Error saving activities: ' + (d.message || 'error'));
        }
      })
      .catch(e => showToast('msg_activities', 'Network error saving activities: ' + e));
    }

    // Initialize active color pills and target chips from saved config
    selectTrackingColor(selectedColor);
    selectClassifyColor(selectedClassifyColor);
    setTrackingObjectPreset('__OT_TARGET__');
    setSizingObjectPreset('__OS_TARGET__');

    function updateVLMPlanUI(plan) {
      if (!plan) return;
      const elAct = document.getElementById('vlm_plan_action');
      const elStat = document.getElementById('vlm_plan_status');
      const elPwmDur = document.getElementById('vlm_plan_pwm_dur');
      const elDamp = document.getElementById('vlm_plan_damping');
      const elOvershoot = document.getElementById('vlm_plan_overshoot');
      const elArea = document.getElementById('vlm_plan_area_ratio');
      const elStiction = document.getElementById('vlm_plan_stiction');
      const elSeg = document.getElementById('vlm_plan_seg_pts');
      const elRat = document.getElementById('vlm_plan_rationale');
      const elSrc = document.getElementById('vlm_badge_source');

      if (elAct) {
        elAct.innerText = plan.action || 'IDLE';
        if (plan.action === 'ALIGNED_GRASP') elAct.style.color = '#10b981';
        else if (plan.action && plan.action.startsWith('DRIVE')) elAct.style.color = '#38bdf8';
        else if (plan.action && plan.action.startsWith('PIVOT')) elAct.style.color = '#a78bfa';
        else if (plan.action && plan.action.startsWith('NUDGE')) elAct.style.color = '#f59e0b';
        else elAct.style.color = '#94a3b8';
      }
      if (elStat) {
        const dyn = plan.target_is_moving ? 'Target In Motion' : 'Target Stationary';
        const mode = plan.movement_mode || (plan.target_is_moving ? 'SMOOTH_PULSE' : 'SMOOTH_FAST');
        elStat.innerText = `Heading: ${plan.alignment_status || 'UNKNOWN'} (dx=${plan.horizontal_error_px > 0 ? '+' : ''}${plan.horizontal_error_px}px) • ${dyn} [${mode}]`;
      }
      const base = plan.base_speed || 230;
      if (elPwmDur) elPwmDur.innerText = `${plan.recommended_pwm || 0} PWM | ${plan.duration_ms || 0} ms`;
      if (elDamp) elDamp.innerText = `Damping: ${(plan.damping_factor || 1.0).toFixed(2)}x (Base: ${base} PWM)`;
      if (elOvershoot) {
        const risk = plan.overshoot_risk || 'LOW';
        elOvershoot.innerText = risk === 'HIGH' ? '⚠️ HIGH (Braking Nudge)' : (risk === 'MEDIUM' ? '⚡ MEDIUM (Damped Approach)' : '🛡️ LOW (Safe Distance)');
        elOvershoot.style.color = risk === 'HIGH' ? '#ef4444' : (risk === 'MEDIUM' ? '#f59e0b' : '#10b981');
      }
      if (elArea) elArea.innerText = `Area Ratio: ${((plan.area_ratio || 0) * 100).toFixed(1)}% of Sweet Spot`;
      if (elStiction) {
        const p = plan.recommended_pwm || 0;
        elStiction.innerText = p > 0 ? `⚡ Active (PWM ${p} | Base ${base})` : '⚠️ Motor Idle';
        elStiction.style.color = p > 0 ? '#10b981' : '#94a3b8';
      }
      const elArmReady = document.getElementById('vlm_arm_readiness');
      const elArmAngles = document.getElementById('vlm_arm_angles');
      const ap = plan.arm_plan || {};
      if (elArmReady) {
        const r = ap.grasp_readiness || 'NO_TARGET';
        if (r === 'READY') {
          elArmReady.innerText = '🎯 READY TO GRASP';
          elArmReady.style.color = '#10b981';
        } else if (r === 'APPROACHING') {
          elArmReady.innerText = '🚀 APPROACHING TARGET';
          elArmReady.style.color = '#38bdf8';
        } else if (r === 'ALIGNING') {
          elArmReady.innerText = '🔄 ALIGNING HEADING';
          elArmReady.style.color = '#f59e0b';
        } else if (r === 'OVERSHOT') {
          elArmReady.innerText = '⚠️ OVERSHOT (REVERSING)';
          elArmReady.style.color = '#ef4444';
        } else {
          elArmReady.innerText = '🦾 STOWED (Holding)';
          elArmReady.style.color = '#94a3b8';
        }
      }
      if (elArmAngles && ap.calibrated_angles) {
        const ca = ap.calibrated_angles;
        elArmAngles.innerText = `S1: ${ca.s1_stow}°->${ca.s1_down}° | S2: ${ca.s2_stow}°->${ca.s2_down}° | Grip: ${ca.grip_target}°`;
      }
      if (elSeg) elSeg.innerText = `Contour: ${plan.target_info?.segmentation_points || 0} polygon pts`;
      if (elRat && plan.rationale) elRat.innerText = plan.rationale;
      if (elSrc) elSrc.innerText = plan.source || 'HYBRID VLM';
    }

    function triggerVLMPlan() {
      const btn = event?.target;
      if (btn) btn.innerText = '🧠 Planning...';
      fetch('/api/vlm/plan', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          if (btn) btn.innerText = '🧠 Ask VLM to Plan';
          if (d.plan) {
            updateVLMPlanUI(d.plan);
            showToast('msg_main', `VLM Plan: ${d.plan.action} (${d.plan.recommended_pwm} PWM, ${d.plan.duration_ms}ms)`, '#8b5cf6');
          }
        })
        .catch(e => {
          if (btn) btn.innerText = '🧠 Ask VLM to Plan';
          console.error('VLM Plan error:', e);
        });
    }

    function executeVLMPlan() {
      const btn = document.getElementById('btn_vlm_exec');
      if (btn) btn.innerText = '⚡ Executing...';
      fetch('/api/vlm/execute', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          if (btn) btn.innerText = '⚡ Execute Planned Pulse';
          if (d.executed === 'ALIGNED_GRASP') {
            showToast('msg_main', '🦾 Executing VLM-planned servo arm grasp sequence', '#10b981');
          } else {
            showToast('msg_main', `Executed: ${d.executed} (${d.pwm || 0} PWM, ${d.duration_ms || 0}ms)`, '#10b981');
          }
        })
        .catch(e => {
          if (btn) btn.innerText = '⚡ Execute Planned Pulse';
          console.error('VLM Exec error:', e);
        });
    }

    function executeVLMGrab() {
      const btn = document.getElementById('btn_vlm_grab');
      if (btn) btn.innerText = '🦾 Grasping...';
      fetch('/api/vlm/grab', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          if (btn) btn.innerText = '🦾 Execute Arm Grasp';
          showToast('msg_main', '🦾 Initiated VLM-planned arm grasp sequence', '#a78bfa');
        })
        .catch(err => {
          if (btn) btn.innerText = '🦾 Execute Arm Grasp';
          console.error(err);
        });
    }

    function toggleVLMAutoPlan(enabled) {
      fetch('/api/vlm/auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_advisory: enabled })
      })
      .then(r => r.json())
      .then(d => {
        showToast('msg_main', `VLM Auto-Advisory: ${d.auto_advisory ? 'ENABLED' : 'DISABLED'}`, d.auto_advisory ? '#10b981' : '#64748b');
      });
    }

    function showToast(id, msg, color) {
      const el = document.getElementById(id);
      if (el) {
        el.innerText = msg;
        if (color) el.style.color = color;
        setTimeout(() => { if (el.innerText === msg) el.innerText = ''; }, 3500);
      }
    }

    // --- WiFi & Hotspot Client Subsystem ---
    function refreshWiFiStatus() {
      fetch('/api/wifi/status')
        .then(r => r.json())
        .then(d => {
          const badge = document.getElementById('wifi_status_badge');
          const modeEl = document.getElementById('wifi_live_mode');
          const ssidEl = document.getElementById('wifi_live_ssid');
          const ipEl = document.getElementById('wifi_live_ip');
          const sigEl = document.getElementById('wifi_live_signal');
          const barsEl = document.getElementById('wifi_live_bars');
          const apRadio = document.getElementById('rb_boot_ap');
          const wifiRadio = document.getElementById('rb_boot_wifi');

          if (badge) {
            if (d.mode === 'hotspot') {
              badge.innerText = '● HOTSPOT ACTIVE';
              badge.style.color = '#f59e0b';
              badge.style.borderColor = '#f59e0b';
            } else if (d.mode === 'client') {
              badge.innerText = '● WIFI CONNECTED';
              badge.style.color = '#10b981';
              badge.style.borderColor = '#10b981';
            } else {
              badge.innerText = '● DISCONNECTED';
              badge.style.color = '#ef4444';
              badge.style.borderColor = '#ef4444';
            }
          }
          if (modeEl) modeEl.innerText = (d.mode || 'CLIENT').toUpperCase();
          if (ssidEl) ssidEl.innerText = d.ssid || 'None';
          if (ipEl) ipEl.innerText = d.ip || 'Disconnected';
          if (sigEl) sigEl.innerText = (d.mode === 'hotspot') ? '100% (AP)' : (d.signal ? d.signal + '%' : '--');
          if (barsEl) barsEl.innerText = (d.mode === 'hotspot') ? '📡 Host Access Point' : (d.security || 'WPA2');

          if (d.startup_mode === 'ap' && apRadio) apRadio.checked = true;
          if (d.startup_mode === 'wifi' && wifiRadio) wifiRadio.checked = true;
        })
        .catch(e => console.error('WiFi status error:', e));
    }

    function scanWiFiNetworks() {
      const btn = document.getElementById('btn_scan_wifi');
      const container = document.getElementById('wifi_networks_container');
      if (btn) btn.innerText = '⏳ Scanning...';
      if (container) container.innerHTML = '<div style="font-size:12px; color:var(--text-muted); text-align:center; padding:12px;">Discovering nearby 2.4/5GHz access points...</div>';

      fetch('/api/wifi/scan')
        .then(r => r.json())
        .then(d => {
          if (btn) btn.innerText = '🔍 Scan Nearby Networks';
          if (d.status === 'ok' && d.networks && d.networks.length > 0) {
            let html = '';
            d.networks.forEach(net => {
              const sec = net.security || 'Open';
              const lock = sec === 'Open' ? '🔓' : '🔒';
              const savedBadge = net.is_saved ? '<span style="background:#10b981; color:#070d1d; font-size:10px; font-weight:800; padding:1px 6px; border-radius:4px; margin-left:4px;">★ SAVED</span>' : '';
              html += `
                <div onclick="selectScannedSSID('${net.ssid}', ${net.is_saved ? 'true' : 'false'})" style="display:flex; justify-content:space-between; align-items:center; padding:7px 10px; margin-bottom:4px; border-radius:5px; background:rgba(255,255,255,0.03); cursor:pointer; border:1px solid rgba(255,255,255,0.08); transition:background 0.15s;" onmouseover="this.style.background='rgba(59,130,246,0.15)'" onmouseout="this.style.background='rgba(255,255,255,0.03)'">
                  <div style="font-size:13px; font-weight:700; color:#fff; display:flex; align-items:center; gap:6px;">
                    <span>${lock}</span>
                    <span>${net.ssid}</span>
                    ${savedBadge}
                  </div>
                  <div style="font-size:11px; color:#38bdf8; display:flex; align-items:center; gap:8px;">
                    <span>${net.bars || ''} ${net.signal}%</span>
                    <span style="color:var(--text-muted); font-size:10px;">${sec}</span>
                  </div>
                </div>
              `;
            });
            if (container) container.innerHTML = html;
          } else {
            if (container) container.innerHTML = '<div style="font-size:12px; color:var(--text-muted); text-align:center; padding:12px;">No wireless networks detected in range.</div>';
          }
        })
        .catch(e => {
          if (btn) btn.innerText = '🔍 Scan Nearby Networks';
          if (container) container.innerHTML = `<div style="font-size:12px; color:#ef4444; text-align:center; padding:12px;">Scan error: ${e}</div>`;
        });
    }

    function selectScannedSSID(ssid, isSaved) {
      const input = document.getElementById('wifi_input_ssid');
      if (input) input.value = ssid;
      const pwd = document.getElementById('wifi_input_pwd');
      if (pwd) {
        if (isSaved) {
          pwd.placeholder = "Saved profile detected — leave empty to connect directly";
          pwd.value = "";
        } else {
          pwd.placeholder = "Enter WiFi password (leave empty if open)...";
        }
        pwd.focus();
      }
    }

    function togglePasswordVisibility(inputId) {
      const el = document.getElementById(inputId);
      if (!el) return;
      el.type = el.type === 'password' ? 'text' : 'password';
    }

    function connectWiFi() {
      const ssidEl = document.getElementById('wifi_input_ssid');
      const pwdEl = document.getElementById('wifi_input_pwd');
      const btn = document.getElementById('btn_connect_wifi');
      const ssid = ssidEl ? ssidEl.value.trim() : '';
      const password = pwdEl ? pwdEl.value : '';

      if (!ssid) {
        showToast('msg_wifi_connect', '⚠️ Please enter or select a network SSID.', '#f59e0b');
        return;
      }

      if (btn) {
        btn.disabled = true;
        btn.innerText = '⏳ Connecting...';
      }
      showToast('msg_wifi_connect', `Connecting to ${ssid}... Please wait.`, '#38bdf8');

      fetch('/api/wifi/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ssid: ssid, password: password })
      })
      .then(r => r.json())
      .then(d => {
        if (btn) {
          btn.disabled = false;
          btn.innerText = '🔗 Connect to Network';
        }
        if (d.status === 'ok') {
          showToast('msg_wifi_connect', `✅ ${d.message} (IP: ${d.ip})`, '#10b981');
          refreshWiFiStatus();
        } else {
          showToast('msg_wifi_connect', `❌ ${d.message}`, '#ef4444');
        }
      })
      .catch(e => {
        if (btn) {
          btn.disabled = false;
          btn.innerText = '🔗 Connect to Network';
        }
        showToast('msg_wifi_connect', `Network error connecting: ${e}`, '#ef4444');
      });
    }

    function activateHotspot() {
      const ssidEl = document.getElementById('ap_input_ssid');
      const pwdEl = document.getElementById('ap_input_pwd');
      const btn = document.getElementById('btn_start_ap');
      const ssid = ssidEl ? ssidEl.value.trim() : 'egrabbot-ap';
      const password = pwdEl ? pwdEl.value : 'egrabbot1234';

      if (password && password.length < 8) {
        showToast('msg_ap', '⚠️ Hotspot password must be at least 8 characters.', '#f59e0b');
        return;
      }

      if (!confirm(`Switch robot into Access Point Hotspot mode?\n\nSSID: ${ssid}\nPassword: ${password}\n\nExisting WiFi client connection will drop. Connect to "${ssid}" and open https://10.42.0.1:5001.`)) {
        return;
      }

      if (btn) {
        btn.disabled = true;
        btn.innerText = '⏳ Activating Hotspot...';
      }
      showToast('msg_ap', 'Switching to Hotspot AP mode...', '#38bdf8');

      fetch('/api/wifi/hotspot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ssid: ssid, password: password })
      })
      .then(r => r.json())
      .then(d => {
        if (btn) {
          btn.disabled = false;
          btn.innerText = '📡 Switch to E-GrabBot Hotspot Mode';
        }
        if (d.status === 'ok') {
          showToast('msg_ap', `✅ ${d.message}`, '#10b981');
          refreshWiFiStatus();
        } else {
          showToast('msg_ap', `❌ ${d.message}`, '#ef4444');
        }
      })
      .catch(e => {
        if (btn) {
          btn.disabled = false;
          btn.innerText = '📡 Switch to E-GrabBot Hotspot Mode';
        }
        showToast('msg_ap', `Hotspot activation initiated. Connect to ${ssid} at https://10.42.0.1:5001.`, '#10b981');
      });
    }

    function setBootNetworkMode(mode) {
      fetch('/api/wifi/startup_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
      })
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          showToast('msg_wifi_global', `✅ Default startup mode set to: ${mode === 'ap' ? 'Access Point Hotspot' : 'WiFi Client Station'}`, '#10b981');
        }
      })
      .catch(e => showToast('msg_wifi_global', 'Error setting startup mode: ' + e, '#ef4444'));
    }

    // Initial sync of dropdown selects with target object inputs
    setTimeout(() => {
      const otInp = document.getElementById('ot_target_input');
      if (otInp && otInp.value) {
        onCustomObjectInput(otInp.value);
      }
      const osInp = document.getElementById('os_target_input');
      if (osInp && osInp.value) {
        onCustomSizingInput(osInp.value);
      }
    }, 200);

    // Call initial WiFi status refresh
    setTimeout(refreshWiFiStatus, 1000);
  </script>
</body>
</html>
"""

class DualStackServer(ThreadingHTTPServer):
    """Dual-stack HTTP server accepting both IPv4 and IPv6 requests on the same port."""
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        host, port = server_address
        if host in ("0.0.0.0", "", "localhost", "127.0.0.1") and socket.has_ipv6:
            self.address_family = socket.AF_INET6
            server_address = ("::", port)
        elif ":" in host:
            self.address_family = socket.AF_INET6
        else:
            self.address_family = socket.AF_INET
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

    def server_bind(self):
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception:
                pass
        super().server_bind()


class Port80RedirectHandler(BaseHTTPRequestHandler):
    """Auto-redirects requests to port 80 to secure target port (5001)."""
    protocol = "https"
    target_port = 5001

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        host = self.headers.get("Host", "egrabbot.local").split(":")[0]
        self.send_response(302)
        self.send_header("Location", f"{self.protocol}://{host}:{self.target_port}{self.path}")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if self.path != "/api/telemetry" and not self.path.startswith("/video_feed"):
            client = self.client_address[0] if self.client_address else "unknown"
            print(f"[Web] {client} - {self.command} {self.path}")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def _send_json(self, data, status=200):
        def _json_default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return str(obj)

        body = json.dumps(data, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                raw = self.rfile.read(length).decode("utf-8")
                return json.loads(raw)
        except Exception:
            pass
        return {}

    def do_GET(self):
        global feed_destination
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            check_reload_robot_config()
            html = HTML_PAGE
            s1_min = int(servo_state.get("s1_min", 60))
            s1_max = int(servo_state.get("s1_max", 170))
            s2_min = int(servo_state.get("s2_min", 0))
            s2_max = int(servo_state.get("s2_max", 45))
            s3_close = int(servo_state.get("s3_close", 40))
            s3_open = int(servo_state.get("s3_open", 170))
            s3_cfg_max = int(servo_state.get("s3_max", 170))
            s3_live_min = min(s3_close, s3_open)
            s3_live_max = max(s3_close, s3_open, s3_cfg_max)

            live_s1 = max(s1_min, min(s1_max, int(servo_state.get("live_s1", servo_state.get("s1_center", 93)))))
            live_s2 = max(s2_min, min(s2_max, int(servo_state.get("live_s2", servo_state.get("s2_center", 45)))))
            live_s3 = max(s3_live_min, min(s3_live_max, int(servo_state.get("live_s3", s3_close))))
            servo_state["live_s1"] = live_s1
            servo_state["live_s2"] = live_s2
            servo_state["live_s3"] = live_s3

            html = html.replace("__LIVE_S1__", str(live_s1))
            html = html.replace("__LIVE_S2__", str(live_s2))
            html = html.replace("__LIVE_S3__", str(live_s3))
            html = html.replace("__S1_MIN__", str(s1_min))
            html = html.replace("__S1_MAX__", str(s1_max))
            html = html.replace("__S1_STOW__", str(servo_state["s1_stow"]))
            html = html.replace("__S1_DOWN__", str(servo_state["s1_down"]))
            html = html.replace("__S1_CENTER__", str(servo_state.get("s1_center", 93)))
            html = html.replace("__S2_MIN__", str(s2_min))
            html = html.replace("__S2_MAX__", str(s2_max))
            html = html.replace("__S2_STOW__", str(servo_state["s2_stow"]))
            html = html.replace("__S2_DOWN__", str(servo_state["s2_down"]))
            html = html.replace("__S2_CENTER__", str(servo_state.get("s2_center", 45)))
            html = html.replace("__S3_MIN__", str(servo_state.get("s3_min", s3_live_min)))
            html = html.replace("__S3_MAX__", str(s3_cfg_max))
            html = html.replace("__S3_OPEN__", str(s3_open))
            html = html.replace("__S3_CLOSE__", str(s3_close))
            html = html.replace("__S3_CENTER__", str(servo_state.get("s3_center", 90)))
            html = html.replace("__S3_LIVE_MIN__", str(s3_live_min))
            html = html.replace("__S3_LIVE_MAX__", str(s3_live_max))
            html = html.replace("__BASE_SPD__", str(motor_state["base_speed"]))
            html = html.replace("__TURN_SPD__", str(motor_state["turn_speed"]))
            html = html.replace("__NUDGE_PWM__", str(motor_state["nudge_pwm"]))
            html = html.replace("__TRIM__", str(motor_state["trim_offset"]))
            html = html.replace("__CX__", str(current_spot["center_x"]))
            html = html.replace("__CY__", str(current_spot["center_y"]))
            html = html.replace("__BW__", str(current_spot["box_width"]))
            html = html.replace("__BH__", str(current_spot["box_height"]))
            html = html.replace("__YOLOE_CONF__", str(yoloe_state.get("confidence_threshold", 0.35)))
            html = html.replace("__YOLOE_IMGSZ__", str(yoloe_state.get("imgsz", 320)))
            html = html.replace("__YOLOE_CLASSES__", str(yoloe_state.get("target_classes", "")))
            html = html.replace("__YOLOE_CLASSES__", str(yoloe_state.get("pickable_classes", "")))
            html = html.replace("__YOLOE_MODEL__", str(yoloe_state.get("model_path", "models/yoloe-26n-seg.pt")))
            html = html.replace("__PF_DIST__", str(activities_state["pf_dist"]))
            html = html.replace("__PF_SPEED__", str(activities_state["pf_speed"]))
            html = html.replace("__CT_COLOR__", str(activities_state["ct_color"]))
            html = html.replace("__CT_AREA__", str(activities_state["ct_area"]))
            html = html.replace("__CT_SPEED__", str(activities_state["ct_speed"]))
            html = html.replace("__OT_TARGET__", str(activities_state["ot_target"]))
            html = html.replace("__OT_DIST__", str(activities_state["ot_dist"]))
            html = html.replace("__OT_SPEED__", str(activities_state["ot_speed"]))
            html = html.replace("__CTC_COLOR__", str(activities_state["ctc_color"]))
            html = html.replace("__CTC_AREA__", str(activities_state["ctc_area"]))
            html = html.replace("__CTC_SPEED__", str(activities_state["ctc_speed"]))
            html = html.replace("__OS_TARGET__", str(activities_state["os_target"]))
            html = html.replace("__OS_DIST__", str(activities_state["os_dist"]))
            html = html.replace("__OS_SPEED__", str(activities_state["os_speed"]))
            html = html.replace("__OA_THRESH__", str(activities_state["oa_thresh"]))
            html = html.replace("__OA_CRUISE__", str(activities_state["oa_cruise"]))
            html = html.replace("__OA_TURN__", str(activities_state["oa_turn"]))

            # Server-side LCD View-Only HUD injection
            client_ip = self.client_address[0] if hasattr(self, "client_address") and self.client_address else "127.0.0.1"
            is_local = client_ip in ("127.0.0.1", "::1", "localhost")
            is_lcd = ("device=lcd" in parsed.query) or (is_local and "device=web" not in parsed.query and feed_destination == "lcd")
            if is_lcd:
                html = html.replace('<html lang="en">', '<html lang="en" class="lcd-view-only">')
                html = html.replace('<body>', '<body class="lcd-view-only">')

            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path == "/logo.webp" or parsed.path == "/company_asset/Erovoutika-Light-Logo-1.webp":
            if os.path.exists(LOGO_PATH):
                with open(LOGO_PATH, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/webp")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Logo Not Found")

        elif parsed.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                frame = None
                if camera and camera.is_opened():
                    frame = camera.get_frame()

                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAM FEED UNAVAILABLE", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                # Draw Sweet Spot Overlay Box
                sx, sy = current_spot["center_x"], current_spot["center_y"]
                sw, sh = current_spot["box_width"], current_spot["box_height"]
                x1, y1 = max(0, sx - sw // 2), max(0, sy - sh // 2)
                x2, y2 = min(frame.shape[1], sx + sw // 2), min(frame.shape[0], sy + sh // 2)

                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.drawMarker(vis, (sx, sy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, "GRASP SWEET SPOT", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                # Draw Live YOLOE Detection Box, Mask & Visual Servoing Linkage (only when no activity is running)
                det = last_detection
                if not (activity_manager and activity_manager.running) and det and getattr(det, "detected", False):
                    bb = getattr(det, "bounding_box", None)
                    if bb and len(bb) == 4:
                        ymin, xmin, ymax, xmax = bb
                        color = (0, 255, 0) if getattr(det, "pickable", False) else (0, 165, 255)

                        # 1. Draw segmentation polygon if present
                        poly = getattr(det, "mask_polygon", None)
                        if poly and len(poly) >= 3:
                            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                            overlay = vis.copy()
                            cv2.fillPoly(overlay, [pts], color)
                            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

                        # 2. Draw bounding box and label
                        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
                        lbl = f"{det.category} ({det.confidence:.2f})"
                        cv2.putText(vis, lbl, (xmin, max(20, ymin - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # 3. Horizontal centering linkage line between center crosshair and object centroid
                        if not (activity_manager and activity_manager.running):
                            cx, cy = det.center
                            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                            dx = cx - sx
                            is_x_centered = abs(dx) <= 20
                            link_col = (0, 255, 0) if is_x_centered else (0, 165, 255)
                            cv2.circle(vis, (sx, sy), 5, link_col, -1)
                            cv2.line(vis, (sx, sy), (cx, cy), link_col, 2)
                            align_txt = f"dx={dx:+d}px [X-CENTERED]" if is_x_centered else f"dx={dx:+d}px [{'TURN RIGHT' if dx > 0 else 'TURN LEFT'}]"
                            cv2.putText(vis, align_txt, (xmin, min(vis.shape[0] - 8, ymax + 18)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, link_col, 2)

                # Draw Autonomous Activity Overlay
                if activity_manager and activity_manager.running:
                    try:
                        act_status = activity_manager.get_status()
                        act_raw = act_status.get("active_activity", "Activity")
                        act_name = act_raw.replace("_", " ").title()
                        tb = act_status.get("target_box")
                        target_label = f"TARGET: {act_name}"
                        d_cm = act_status.get("distance_cm", 0.0)
                        dist_tag = f" {d_cm:.0f}cm" if d_cm > 0 else ""
                        box_color = (0, 255, 0)
                        if act_raw == "object_tracking":
                            t_obj = act_status.get("target_object", "object")
                            target_label = f"TARGET: {t_obj.upper()}{dist_tag}"
                            box_color = (56, 189, 248)
                        elif act_raw == "color_track_and_classify":
                            c_color = act_status.get("target_color", "Color")
                            c_cls = act_status.get("classified_object", "Scanning...")
                            target_label = f"TARGET: {c_color.upper()} [{c_cls.upper()}]{dist_tag}"
                            box_color = (0, 255, 128)
                        elif act_raw == "color_tracking":
                            c_color = act_status.get("target_color", "Color")
                            target_label = f"TARGET: {c_color.upper()}{dist_tag}"
                            box_color = (0, 255, 128)
                        elif act_raw == "person_follower":
                            tt = act_status.get("target_type", "PERSON")
                            gy = act_status.get("ground_y", 0.0)
                            target_label = f"TARGET: {tt.replace('_', ' ')} [{d_cm:.0f}cm | {gy:.0f}%]"
                            box_color = (0, 255, 128) if "LEG" in tt else (255, 0, 255)
                        elif act_raw == "object_sizing":
                            sz = act_status.get("sizing", {})
                            s_cat = sz.get("size_category", "MEASURING")
                            s_w = sz.get("width_mm", 0.0)
                            s_h = sz.get("height_mm", 0.0)
                            s_fit = sz.get("gripper_fit", "UNKNOWN")
                            s_name = sz.get("object", "object").upper()
                            target_label = f"SIZE: {s_name} {s_cat} ({s_w:.0f}x{s_h:.0f}mm) [{s_fit}]"
                            box_color = (0, 255, 255)
                        elif act_raw == "obstacle_avoidance":
                            oa = act_status.get("obstacle", {})
                            oa_act = oa.get("action", act_status.get("action", "CRUISING"))
                            target_label = f"NAV: {oa_act} [{oa.get('clear_path', 'FORWARD')}]"
                            box_color = (0, 0, 255) if (oa_act.startswith("AVOID") or oa_act.startswith("ESCAPE")) else (0, 255, 0)

                        if tb and len(tb) == 4:
                            ymin = max(0, min(vis.shape[0] - 1, int(tb[0])))
                            xmin = max(0, min(vis.shape[1] - 1, int(tb[1])))
                            ymax = max(0, min(vis.shape[0] - 1, int(tb[2])))
                            xmax = max(0, min(vis.shape[1] - 1, int(tb[3])))
                            if xmax > xmin and ymax > ymin:
                                cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), box_color, 2)
                                cv2.putText(vis, target_label, (xmin, max(20, ymin - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

                            # Determine target anchor point for autonomous activity linkage
                            act_tx = (xmin + xmax) // 2
                            act_ty = (ymin + ymax) // 2

                            if act_raw == "person_follower":
                                lb = act_status.get("leg_box")
                                if lb and len(lb) == 4:
                                    lymin, lxmin, lymax, lxmax = [int(v) for v in lb]
                                    cv2.rectangle(vis, (lxmin, lymin), (lxmax, lymax), (0, 255, 255), 2)
                                    cv2.line(vis, (lxmin, lymax), (lxmax, lymax), (0, 255, 0), 3)
                                    foot_cx = (lxmin + lxmax) // 2
                                    cv2.circle(vis, (foot_cx, lymax), 5, (0, 0, 255), -1)
                                    cv2.putText(vis, "FEET ANCHOR", (max(0, lxmin), min(vis.shape[0] - 6, lymax + 16)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
                                    act_tx = foot_cx
                                    act_ty = lymax
                                else:
                                    act_ty = ymax  # bottom ground anchor
                            elif act_raw == "object_sizing":
                                sz = act_status.get("sizing", {})
                                w_str = f"{sz.get('width_mm', 0):.0f}mm"
                                h_str = f"{sz.get('height_mm', 0):.0f}mm"
                                cv2.arrowedLine(vis, (xmin, max(0, ymin - 4)), (xmax, max(0, ymin - 4)), (0, 255, 255), 1, tipLength=0.08)
                                cv2.arrowedLine(vis, (xmax, max(0, ymin - 4)), (xmin, max(0, ymin - 4)), (0, 255, 255), 1, tipLength=0.08)
                                cv2.arrowedLine(vis, (min(vis.shape[1]-1, xmax + 4), ymin), (min(vis.shape[1]-1, xmax + 4), ymax), (0, 255, 255), 1, tipLength=0.08)
                                cv2.arrowedLine(vis, (min(vis.shape[1]-1, xmax + 4), ymax), (min(vis.shape[1]-1, xmax + 4), ymin), (0, 255, 255), 1, tipLength=0.08)
                                cv2.putText(vis, f"W:{w_str}", (xmin + max(1, (xmax - xmin)//4), max(12, ymin - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                                cv2.putText(vis, f"H:{h_str}", (min(vis.shape[1] - 55, xmax + 8), ymin + max(1, (ymax - ymin)//2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

                            tp = act_status.get("target_polygon")
                            if tp and len(tp) >= 3:
                                poly_pts = np.array(tp, dtype=np.int32).reshape((-1, 1, 2))
                                overlay = vis.copy()
                                poly_color = (0, 255, 255)
                                if act_raw == "person_follower":
                                    poly_color = (0, 255, 128)  # Neon green for human/legs
                                elif act_raw in ("color_tracking", "color_track_and_classify"):
                                    tc = str(act_status.get("target_color", "")).lower()
                                    color_bgr_map = {
                                        "red": (0, 0, 255),
                                        "green": (0, 255, 0),
                                        "blue": (255, 128, 0),
                                        "yellow": (0, 255, 255),
                                        "orange": (0, 140, 255),
                                        "purple": (200, 0, 200),
                                        "cyan": (255, 255, 0),
                                    }
                                    poly_color = color_bgr_map.get(tc, (0, 255, 255))
                                elif act_raw == "object_tracking":
                                    poly_color = (56, 189, 248)  # Sky blue
                                cv2.fillPoly(overlay, [poly_pts], poly_color)
                                cv2.addWeighted(overlay, 0.28, vis, 0.72, 0, vis)
                                cv2.polylines(vis, [poly_pts], isClosed=True, color=poly_color, thickness=2)

                            # Draw Visual Servoing Linkage Line for Active Activity Target
                            cv2.circle(vis, (act_tx, act_ty), 5, (0, 0, 255), -1)
                            dx = act_tx - sx
                            is_x_centered = abs(dx) <= 25
                            link_col = (0, 255, 0) if is_x_centered else (0, 165, 255)
                            cv2.circle(vis, (sx, sy), 5, link_col, -1)
                            cv2.line(vis, (sx, sy), (act_tx, act_ty), link_col, 2)
                            align_txt = f"dx={dx:+d}px [X-CENTERED]" if is_x_centered else f"dx={dx:+d}px [{'TURN RIGHT' if dx > 0 else 'TURN LEFT'}]"
                            cv2.putText(vis, align_txt, (xmin, min(vis.shape[0] - 8, ymax + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, link_col, 2)

                        if act_raw == "obstacle_avoidance":
                            oa = act_status.get("obstacle", {})
                            ih, iw = vis.shape[:2]
                            col_w = iw // 3
                            y_t = int(ih * 0.45)
                            y_b = int(ih * 0.95)

                            l_blk = oa.get("left_zone") == "BLOCKED"
                            c_blk = oa.get("center_zone") == "BLOCKED"
                            r_blk = oa.get("right_zone") == "BLOCKED"

                            cv2.line(vis, (col_w, y_t), (col_w, y_b), (220, 220, 220), 1)
                            cv2.line(vis, (2 * col_w, y_t), (2 * col_w, y_b), (220, 220, 220), 1)
                            cv2.line(vis, (0, y_t), (iw, y_t), (180, 180, 180), 1)
                            cv2.line(vis, (0, y_b), (iw, y_b), (180, 180, 180), 1)

                            col_l = (0, 0, 230) if l_blk else (0, 200, 0)
                            col_c = (0, 0, 230) if c_blk else (0, 200, 0)
                            col_r = (0, 0, 230) if r_blk else (0, 200, 0)

                            cv2.rectangle(vis, (2, y_t + 2), (col_w - 2, y_b - 2), col_l, 2)
                            cv2.rectangle(vis, (col_w + 2, y_t + 2), (2 * col_w - 2, y_b - 2), col_c, 2)
                            cv2.rectangle(vis, (2 * col_w + 2, y_t + 2), (iw - 2, y_b - 2), col_r, 2)

                            dist_l_val = oa.get('left_dist_cm', 200.0)
                            dist_c_val = oa.get('center_dist_cm', 200.0)
                            dist_r_val = oa.get('right_dist_cm', 200.0)
                            cv2.putText(vis, f"L:{dist_l_val:.0f}cm ({oa.get('left_score', 0)}%)", (10, y_b - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_l, 1)
                            cv2.putText(vis, f"C:{dist_c_val:.0f}cm ({oa.get('center_score', 0)}%)", (col_w + 10, y_b - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_c, 1)
                            cv2.putText(vis, f"R:{dist_r_val:.0f}cm ({oa.get('right_score', 0)}%)", (2 * col_w + 10, y_b - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_r, 1)

                            nav_banner = f"NAV: {oa.get('action', 'CRUISING')} | L:{dist_l_val:.0f} C:{dist_c_val:.0f} R:{dist_r_val:.0f}cm"
                            cv2.putText(vis, nav_banner, (10, min(ih - 10, y_b + 18)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

                            obs_cnts = oa.get("contours", [])
                            for c in obs_cnts:
                                if len(c) >= 3:
                                    poly_pts = np.array(c, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(vis, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)

                        is_stat = not act_status.get("config", {}).get("motion_enabled", True)
                        motion_tag = " [STATIONARY TRACK]" if is_stat else " [AUTO DRIVE]"
                        cv2.putText(vis, f"ACTIVITY: {act_name.upper()} ({act_status.get('status', 'RUNNING')}){motion_tag}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255) if is_stat else (0, 255, 128), 2)

                        arm_ik = act_status.get("arm_ik", {})
                        if arm_ik and arm_ik.get("active"):
                            s1_val = arm_ik.get("s1", 93)
                            s2_val = arm_ik.get("s2", 45)
                            v_st = arm_ik.get("vertical_status", "CENTERED")
                            lat_ms = arm_ik.get("latency_ms", 0.0)
                            arm_text = f"ARM IK: S1:{s1_val}deg S2:{s2_val}deg [{v_st}] ({lat_ms:.0f}ms)"
                            cv2.putText(vis, arm_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 2)
                    except Exception as act_draw_err:
                        print(f"[Web] Error drawing activity overlay: {act_draw_err}")
                ret, jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    time.sleep(0.05)
                    continue

                b = jpeg.tobytes()
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    time.sleep(0.04)  # ~25 FPS
                except (BrokenPipeError, ConnectionResetError):
                    break

        elif parsed.path == "/api/telemetry":
            telem = None
            if comm:
                try:
                    telem = comm.get_latest_telemetry()
                except Exception:
                    pass

            left_pwm = telem.left_pwm if telem else 0
            right_pwm = telem.right_pwm if telem else 0
            s1 = arm.cur_s1 if arm else servo_state["live_s1"]
            s2 = arm.cur_s2 if arm else servo_state["live_s2"]
            s3 = arm.cur_s3 if arm else servo_state["live_s3"]
            ping_ms = telem.ping_ms if telem else 0.0
            connected = telem.connected if telem else (comm.is_connected() if comm else False)

            yoloe_payload = {
                "object": vlm_info.get("object", "NONE"),
                "category": vlm_info.get("category", "NONE"),
                "color": vlm_info.get("color", "NONE"),
                "color_hex": vlm_info.get("color_hex", "#888888"),
                "backend": vlm_info.get("backend", "PyTorch"),
                "confidence": vlm_info.get("confidence", 0.0),
                "pickable": vlm_info.get("pickable", False),
                "aligned": vlm_info.get("aligned", False),
                "dx": vlm_info.get("dx", 0),
                "dy": vlm_info.get("dy", 0),
                "cx": vlm_info.get("cx", 0),
                "cy": vlm_info.get("cy", 0),
                "bbox": vlm_info.get("bbox", []),
                "latency_ms": vlm_info.get("latency_ms", 0.0),
                "fps": vlm_info.get("fps", 0.0),
                "model_path": yoloe_state.get("model_path", "models/yoloe-11s-seg.pt"),
                "imgsz": yoloe_state.get("imgsz", 320),
                "confidence_threshold": yoloe_state.get("confidence_threshold", 0.25),
                "enforce_limitations": yoloe_state.get("enforce_limitations", False),
            }

            check_reload_robot_config()
            status_str = "Hardware (Bluetooth)" if not is_mock_comm else "Simulation (Mock)"
            vlm_obj_disp = vlm_info.get("category", "NONE")
            vlm_conf_disp = vlm_info.get("confidence", 0.0)
            vlm_det_disp = vlm_info.get("details", "--")
            if activity_manager and activity_manager.running:
                act_st = activity_manager.get_status()
                act_raw_name = act_st.get("active_activity", "none")
                if act_st.get("target_found"):
                    vlm_obj_disp = (act_st.get("target_type") or act_st.get("target_object") or act_st.get("target_color") or act_raw_name).upper()
                    vlm_conf_disp = 0.95
                    vlm_det_disp = act_st.get("details", "--")
                else:
                    vlm_obj_disp = f"SEARCHING ({act_raw_name.upper()})"
                    vlm_det_disp = act_st.get("details", "Scanning for targets...")

            self._send_json({
                "link_status": status_str,
                "is_mock": is_mock_comm,
                "connected": connected,
                "ping_ms": ping_ms,
                "left_pwm": left_pwm,
                "right_pwm": right_pwm,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "arduino_s1": getattr(telem, "s1_shoulder", None) if telem else None,
                "arduino_s2": getattr(telem, "s2_elbow", None) if telem else None,
                "arduino_s3": getattr(telem, "s3_gripper", None) if telem else None,
                "telem_age_s": round(time.time() - telem.timestamp, 2) if (telem and getattr(telem, "timestamp", 0) > 0) else None,
                "feed_destination": feed_destination,
                "state": get_robot_state_name(),
                "showcase_running": showcase_running,
                "vlm_object": vlm_info.get("category", "NONE"),
                "vlm_confidence": vlm_info.get("confidence", 0.0),
                "vlm_details": vlm_info.get("details", "--"),
                "vlm_object": vlm_obj_disp,
                "vlm_confidence": vlm_conf_disp,
                "vlm_details": vlm_det_disp,
                "activity": activity_manager.get_status() if activity_manager else None,
                "yoloe": yoloe_payload,
                "vlm_plan": vlm_planner.get_last_plan().to_dict() if (vlm_planner and vlm_planner.get_last_plan()) else None,
                "vlm_auto_advisory": vlm_auto_advisory,
                "servo_cal": get_servo_cal_dict(),
                "servo_cal_mtime": _robot_config_mtime,
                "server_start_time": _SERVER_START_TIME
            })

        elif parsed.path == "/api/servo_cal":
            check_reload_robot_config()
            self._send_json({"status": "ok", "servo_cal": get_servo_cal_dict(), "servo_cal_mtime": _robot_config_mtime})

        elif parsed.path == "/api/activities/status":
            if activity_manager:
                self._send_json(activity_manager.get_status())
            else:
                self._send_json({
                    "active_activity": "none",
                    "running": False,
                    "status": "OFFLINE",
                    "target_box": None,
                    "fps": 0.0
                })

        elif parsed.path == "/api/activities/config":
            self._send_json({"status": "ok", "activities": activities_state})

        elif parsed.path == "/api/yoloe/config":
            models_dir = os.path.join(BASE_DIR, "models")
            found_models = []
            if os.path.exists(models_dir):
                for f in sorted(os.listdir(models_dir)):
                    if f.endswith(".pt") or f.endswith(".tflite"):
                        found_models.append(f"models/{f}")
            if not found_models:
                found_models = [
                    "models/yoloe-11s-seg.pt",
                    "models/yoloe-11m-seg.pt",
                    "models/yoloe-26s-seg.pt",
                    "models/yoloe-26m-seg.pt",
                    "models/yoloe-26n-seg.pt",
                    "models/yoloe-26n-seg-pf.pt"
                ]

            self._send_json({
                "status": "ok",
                "backend": getattr(vision, "backend_type", "PyTorch") if vision else "PyTorch",
                "model_path": yoloe_state.get("model_path", "models/yoloe-11s-seg.pt"),
                "confidence_threshold": yoloe_state.get("confidence_threshold", 0.25),
                "imgsz": yoloe_state.get("imgsz", 320),
                "target_classes": yoloe_state.get("target_classes", ""),
                "enforce_limitations": yoloe_state.get("enforce_limitations", False),
                "available_models": found_models
            })

        elif parsed.path == "/api/vlm/plan":
            plan = vlm_planner.get_last_plan() if vlm_planner else None
            self._send_json({"status": "ok", "plan": plan.to_dict() if plan else None})

        elif parsed.path == "/api/vlm/adaptation_log":
            if vlm_planner:
                self._send_json({"status": "ok", "adaptation": vlm_planner.get_adaptation_stats()})
            else:
                self._send_json({"status": "error", "message": "VLM planner not initialized"}, 500)

        elif parsed.path == "/api/vlm/trained_objects":
            if vlm_planner:
                self._send_json({"status": "ok", "trained_objects": vlm_planner.get_trained_objects()})
            else:
                self._send_json({"status": "error", "message": "VLM planner not initialized"}, 500)

        elif parsed.path == "/api/wifi/status":
            self._send_json(wifi_manager.get_wifi_status())

        elif parsed.path == "/api/wifi/scan":
            self._send_json({"status": "ok", "networks": wifi_manager.scan_wifi_networks()})
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        global feed_destination
        parsed = urlparse(self.path)

        # Safety override: any manual drive or stop command immediately halts running autonomous activity
        if parsed.path in ("/api/drive_dir", "/api/nudge", "/api/stop"):
            if activity_manager and activity_manager.running:
                activity_manager.stop_activity()

        if parsed.path == "/api/feed_destination":
            data = self._read_json()
            dest = data.get("destination", "lcd")
            if dest in ("lcd", "web"):
                feed_destination = dest
            self._send_json({"status": "ok", "feed_destination": feed_destination})

        elif parsed.path == "/api/activities/toggle_motion":
            data = self._read_json()
            enabled = data.get("enabled", True)
            if activity_manager:
                activity_manager.toggle_motion(enabled)
                activities_state["motion_enabled"] = activity_manager.motion_enabled
                self._send_json({"status": "ok", "motion_enabled": activity_manager.motion_enabled})
            else:
                self._send_json({"status": "error", "message": "ActivityManager not initialized"}, 500)

        elif parsed.path == "/api/activities/start":
            data = self._read_json()
            act = data.get("activity")
            params = data.get("params", {})
            if "target_color" in params:
                c_val = str(params["target_color"]).strip().capitalize()
                activities_state["ct_color"] = c_val
                if activity_manager:
                    activity_manager.config["target_color"] = c_val
            if "target_object" in params:
                o_val = str(params["target_object"]).strip().lower()
                activities_state["ot_target"] = o_val
                if activity_manager:
                    activity_manager.config["target_object"] = o_val
            if activity_manager:
                ok = activity_manager.start_activity(act, params)
                self._send_json({"status": "ok" if ok else "error", "activity": act, "running": ok})
            else:
                self._send_json({"status": "error", "message": "ActivityManager not initialized"}, 500)

        elif parsed.path == "/api/activities/stop":
            if activity_manager:
                activity_manager.stop_activity()
            self._send_json({"status": "ok", "running": False})

        elif parsed.path in ("/api/activities/save", "/api/save_activities"):
            data = self._read_json()
            save_activities_config(data)
            self._send_json({"status": "ok", "activities": activities_state})

        elif parsed.path == "/api/auto/start":
            start_autonomous_mode()
            self._send_json({"status": "ok", "state": get_robot_state_name()})

        elif parsed.path == "/api/auto/stop":
            stop_autonomous_mode()
            self._send_json({"status": "ok", "state": get_robot_state_name()})

        elif parsed.path == "/api/showcase":
            trigger_showcase()
            self._send_json({"status": "ok", "showcase_running": showcase_running})

        elif parsed.path == "/api/drive_dir":
            data = self._read_json()
            direction = data.get("dir", "S")
            base = max(235, int(motor_state.get("base_speed", 235)))
            turn = max(235, int(motor_state.get("turn_speed", 235)))
            trim = motor_state.get("trim_offset", 6)

            left, right = 0, 0
            if direction == "F":
                left = base - trim
                right = base + trim
            elif direction == "B":
                left = -base + trim
                right = -base - trim
            elif direction == "L":
                left = -turn
                right = turn
            elif direction == "R":
                left = turn
                right = -turn

            if comm:
                comm.send_drive(left, right)
            self._send_json({"status": "ok", "left": left, "right": right})

        elif parsed.path == "/api/nudge":
            data = self._read_json()
            direction = data.get("dir", data.get("direction", "F"))
            pwm = int(data.get("pwm", motor_state.get("nudge_pwm", 245)))
            ms = int(data.get("duration_ms", data.get("ms", motor_state.get("nudge_default_ms", 250))))
            if comm:
                comm.send_nudge(direction, ms, pwm)
            self._send_json({"status": "ok", "nudge": direction, "ms": ms, "pwm": pwm})

        elif parsed.path == "/api/stop":
            if comm:
                comm.send_stop()
            self._send_json({"status": "ok"})

        elif parsed.path == "/api/servo":
            check_reload_robot_config()
            data = self._read_json()
            raw_s1 = int(data.get("s1", servo_state["live_s1"]))
            raw_s2 = int(data.get("s2", servo_state["live_s2"]))
            raw_s3 = int(data.get("s3", servo_state["live_s3"]))

            s1_min = int(servo_state.get("s1_min", 60))
            s1_max = int(servo_state.get("s1_max", 170))
            s2_min = int(servo_state.get("s2_min", 0))
            s2_max = int(servo_state.get("s2_max", 45))
            s3_close = int(servo_state.get("s3_close", 40))
            s3_open = int(servo_state.get("s3_open", 170))
            s3_cfg_max = int(servo_state.get("s3_max", 170))
            s3_live_min = min(s3_close, s3_open)
            s3_live_max = max(s3_close, s3_open, s3_cfg_max)

            s1 = max(s1_min, min(s1_max, raw_s1))
            s2 = max(s2_min, min(s2_max, raw_s2))
            s3 = max(s3_live_min, min(s3_live_max, raw_s3))

            servo_state["live_s1"] = s1
            servo_state["live_s2"] = s2
            servo_state["live_s3"] = s3

            if arm:
                arm.move_arm_alternate(s1, s2, s3, clamp_to_bounds=True, wait_for_s1=False)
                servo_state["live_s1"] = arm.cur_s1
                servo_state["live_s2"] = arm.cur_s2
                servo_state["live_s3"] = arm.cur_s3
            elif comm:
                comm.send_servos(s1, s2, s3, pulse=False)
            self._send_json({"status": "ok", "s1": servo_state["live_s1"], "s2": servo_state["live_s2"], "s3": servo_state["live_s3"]})

        elif parsed.path == "/api/macro":
            data = self._read_json()
            macro = data.get("name", "").lower()
            res = False

            if arm:
                if macro == "stow" or macro == "up":
                    res = arm.stow(s3=arm.cur_s3)
                elif macro == "down":
                    res = arm.arm_down()
                elif macro == "center":
                    res = arm.center()
                elif macro == "open":
                    res = arm.open_gripper()
                elif macro == "close":
                    res = arm.close_gripper()
                elif macro == "pick":
                    threading.Thread(target=arm.execute_pick_sequence, daemon=True).start()
                    res = True

                servo_state["live_s1"] = arm.cur_s1
                servo_state["live_s2"] = arm.cur_s2
                servo_state["live_s3"] = arm.cur_s3

            self._send_json({"status": "ok", "macro": macro, "success": res})

        elif parsed.path == "/api/sweet_spot":
            data = self._read_json()
            current_spot["center_x"] = int(data.get("center_x", current_spot["center_x"]))
            current_spot["center_y"] = int(data.get("center_y", current_spot["center_y"]))
            current_spot["box_width"] = int(data.get("box_width", current_spot["box_width"]))
            current_spot["box_height"] = int(data.get("box_height", current_spot["box_height"]))
            self._send_json({"status": "ok", "sweet_spot": current_spot})

        elif parsed.path == "/api/save_servos":
            data = self._read_json()
            for k, v in data.items():
                if k in servo_state:
                    servo_state[k] = int(v)
            save_robot_servos()
            self._send_json({"status": "ok", "servo_cal": get_servo_cal_dict(), "servo_cal_mtime": _robot_config_mtime})

        elif parsed.path == "/api/save_motors":
            data = self._read_json()
            for k, v in data.items():
                if k in motor_state:
                    motor_state[k] = int(v)
            save_robot_motors()
            self._send_json({"status": "ok"})

        elif parsed.path == "/api/save_vision":
            save_vision_config()
            self._send_json({"status": "ok"})

        elif parsed.path == "/api/reconnect_bt":
            connect_comm(force_mock=False)
            status_str = "Hardware Bluetooth connected" if not is_mock_comm else "MockCommAdapter active"
            self._send_json({"status": "ok", "is_mock": is_mock_comm, "message": status_str})

        elif parsed.path == "/api/yoloe/config":
            data = self._read_json()
            persist = bool(data.get("save_yaml", False))
            apply_yoloe_config(data, persist=persist)
            self._send_json({
                "status": "ok",
                "model_path": yoloe_state.get("model_path"),
                "confidence_threshold": yoloe_state.get("confidence_threshold"),
                "imgsz": yoloe_state.get("imgsz"),
                "target_classes": yoloe_state.get("target_classes"),
                "enforce_limitations": yoloe_state.get("enforce_limitations", False),
                "persisted": persist
            })

        elif parsed.path == "/api/vlm/plan":
            check_reload_robot_config()
            sp = dict(current_spot)
            det = last_detection
            # Run VLM trajectory planning with YOLOE readings, coordinates, and segmentation
            plan = vlm_planner.plan_movement(det, sp, use_vlm_llm=True, vlm_timeout_s=3.5)
            self._send_json({"status": "ok", "plan": plan.to_dict()})

        elif parsed.path == "/api/vlm/execute":
            plan = vlm_planner.get_last_plan()
            if not plan:
                if comm:
                    comm.send_stop()
                self._send_json({"status": "ok", "executed": "STOP", "message": "No active motion plan."})
            elif plan.action == "ALIGNED_GRASP":
                if comm:
                    comm.send_stop()
                grip_target = plan.arm_plan.get("calibrated_angles", {}).get("grip_target", None)
                if arm:
                    def _run_grab():
                        arm.open_gripper(wait=True)
                        time.sleep(0.15)
                        arm.move_arm_simultaneous(arm.s1_down, arm.s2_down, arm.s3_open, wait_for_s1=True)
                        time.sleep(0.25)
                        arm.close_gripper(wait=True, angle=grip_target)
                        time.sleep(0.25)
                        arm.move_arm_simultaneous(arm.s1_stow, arm.s2_stow, arm.cur_s3, wait_for_s1=True)
                    threading.Thread(target=_run_grab, daemon=True).start()
                self._send_json({
                    "status": "ok",
                    "executed": "ALIGNED_GRASP",
                    "arm_action": "EXECUTE_GRAB",
                    "grip_angle": grip_target,
                    "message": "Initiated VLM-planned servo arm grasp sequence per robot config."
                })
            elif plan.action == "STOP":
                if comm:
                    comm.send_stop()
                self._send_json({"status": "ok", "executed": "STOP", "message": "Motors halted."})
            else:
                action = plan.action
                pwm = max(vlm_planner.min_overcoming_pwm, min(vlm_planner.max_speed, plan.recommended_pwm))
                dur = plan.duration_ms
                if action == "NUDGE_RIGHT":
                    if comm: comm.send_nudge("R", dur, pwm)
                elif action == "NUDGE_LEFT":
                    if comm: comm.send_nudge("L", dur, pwm)
                elif action == "NUDGE_FORWARD":
                    if comm: comm.send_nudge("F", dur, pwm)
                elif action == "NUDGE_BACK":
                    if comm: comm.send_nudge("B", dur, pwm)
                elif action == "PIVOT_RIGHT":
                    if comm: comm.send_drive(pwm, -pwm)
                elif action == "PIVOT_LEFT":
                    if comm: comm.send_drive(-pwm, pwm)
                elif action == "DRIVE_FORWARD":
                    if comm: comm.send_drive(pwm, pwm)
                self._send_json({"status": "ok", "executed": action, "pwm": pwm, "duration_ms": dur})

        elif parsed.path == "/api/vlm/grab":
            plan = vlm_planner.get_last_plan()
            grip_target = plan.arm_plan.get("calibrated_angles", {}).get("grip_target", None) if plan else None
            if arm:
                def _run_grab_manual():
                    arm.open_gripper(wait=True)
                    time.sleep(0.15)
                    arm.move_arm_simultaneous(arm.s1_down, arm.s2_down, arm.s3_open, wait_for_s1=True)
                    time.sleep(0.25)
                    arm.close_gripper(wait=True, angle=grip_target)
                    time.sleep(0.25)
                    arm.move_arm_simultaneous(arm.s1_stow, arm.s2_stow, arm.cur_s3, wait_for_s1=True)
                threading.Thread(target=_run_grab_manual, daemon=True).start()
            self._send_json({
                "status": "ok",
                "message": "Triggered VLM planned arm grasp sequence.",
                "grip_angle": grip_target
            })

        elif parsed.path == "/api/vlm/auto":
            data = self._read_json()
            global vlm_auto_advisory
            vlm_auto_advisory = bool(data.get("auto_advisory", False))
            self._send_json({"status": "ok", "auto_advisory": vlm_auto_advisory})

        elif parsed.path == "/api/wifi/connect":
            data = self._read_json()
            ssid = data.get("ssid")
            password = data.get("password")
            res = wifi_manager.connect_to_wifi(ssid, password)
            self._send_json(res)

        elif parsed.path == "/api/wifi/hotspot":
            data = self._read_json()
            ssid = data.get("ssid")
            password = data.get("password")
            res = wifi_manager.activate_hotspot(ssid, password)
            self._send_json(res)

        elif parsed.path == "/api/wifi/startup_mode":
            data = self._read_json()
            mode = data.get("mode", "ap")
            res = wifi_manager.set_startup_mode(mode)
            self._send_json(res)

        elif parsed.path == "/api/restart_service":
            self._send_json({"status": "ok", "message": "Restarting web service..."})
            def _delayed_exit():
                time.sleep(0.5)
                os._exit(0)
            threading.Thread(target=_delayed_exit, daemon=True).start()

        else:
            self.send_error(404, "Not Found")


def _start_auto_reloader():
    """Background thread that monitors web interface script and configuration files for changes.
    When HTML, CSS, JS, or config files change on disk, it cleanly exits the process (os._exit(0))
    so systemd automatically restarts egrabbot-web.service, and client browsers auto-reload.
    """
    watched_paths = [os.path.abspath(__file__)]
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cfg in ["config/robot_config.yaml", "config/vision_config.yaml"]:
        p = os.path.join(proj_root, cfg)
        if os.path.exists(p):
            watched_paths.append(p)

    mtimes = {}
    for p in watched_paths:
        try:
            mtimes[p] = os.path.getmtime(p)
        except Exception:
            pass

    def _watch_loop():
        time.sleep(2.0)
        while True:
            time.sleep(1.0)
            for p in list(watched_paths):
                try:
                    if not os.path.exists(p):
                        continue
                    cur_m = os.path.getmtime(p)
                    prev_m = mtimes.get(p)
                    if prev_m is not None and cur_m > prev_m:
                        print(f"\n[AutoReloader] Detected change in {p} (mtime {prev_m} -> {cur_m}).")
                        print("[AutoReloader] Automatically restarting web service for fresh HTML/CSS/JS...")
                        try:
                            if camera:
                                camera.stop()
                        except Exception:
                            pass
                        try:
                            if comm:
                                comm.disconnect()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        os._exit(0)
                    mtimes[p] = cur_m
                except Exception:
                    pass

    t = threading.Thread(target=_watch_loop, daemon=True, name="AutoReloaderThread")
    t.start()


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    global camera, comm, arm
    parser = argparse.ArgumentParser(description="ErovoutikaGrab Local Web Dashboard")
    parser.add_argument("--port", type=int, default=5001, help="Web server port (default: 5001)")
    parser.add_argument("--mock", action="store_true", help="Force simulated hardware adapters")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding interface (default: 0.0.0.0)")
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS and serve plain HTTP")
    args = parser.parse_args()

    load_all_configs()

    # 1. Initialize Camera
    if args.mock:
        print("[Web] Using MockCameraAdapter")
        camera = MockCameraAdapter(width=640, height=480)
        camera.start()
    else:
        vcfg = {}
        if os.path.exists(VISION_CONFIG_PATH):
            with open(VISION_CONFIG_PATH, "r") as f:
                vcfg = yaml.safe_load(f) or {}
        cam_cfg = vcfg.get("camera", {})
        idx = cam_cfg.get("device_index", 0)
        w = cam_cfg.get("width", 640)
        h = cam_cfg.get("height", 480)
        fps = cam_cfg.get("fps", 30)
        fourcc = cam_cfg.get("fourcc", "MJPG")

        camera = V4L2CameraAdapter(device_index=idx, width=w, height=h, target_fps=fps, fourcc=fourcc)
        if not camera.start():
            print("[Warning] Could not initialize V4L2Camera. Falling back to MockCameraAdapter.")
            camera = MockCameraAdapter(width=640, height=480)
            camera.start()

    # 2. Initialize Comm & Arm
    connect_comm(force_mock=args.mock)
    if not args.mock:
        start_bt_watchdog()

    # 3. Start Background Vision Inference Worker
    start_vision_worker()

    # 4. Start Web Server on DualStack (IPv6 and IPv4 simultaneously)
    bind_host = "::" if args.host == "0.0.0.0" else args.host
    try:
        httpd = DualStackServer((bind_host, args.port), WebHandler)
        print(f"[Web] Bound DualStack (IPv6 & IPv4) on port {args.port}")
    except Exception as e:
        print(f"[Web] DualStack binding failed ({e}), using IPv4 on {args.host}:{args.port}")
        httpd = ThreadingHTTPServer((args.host, args.port), WebHandler)

    # SSL / HTTPS Configuration
    use_ssl = False
    if not args.no_ssl and os.path.exists(SSL_CERT_PATH) and os.path.exists(SSL_KEY_PATH):
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=SSL_CERT_PATH, keyfile=SSL_KEY_PATH)
            httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
            use_ssl = True
            print(f"[Web] 🔒 HTTPS (TLS) active with certificate: {SSL_CERT_PATH}")
        except Exception as e:
            print(f"[Web] ⚠️ Failed to initialize SSL ({e}). Falling back to HTTP.")
            use_ssl = False

    proto = "https" if use_ssl else "http"
    Port80RedirectHandler.protocol = proto
    Port80RedirectHandler.target_port = args.port

    # Optional port 80 redirector (if port 80 is available)
    redirect_server = None
    if args.port != 80:
        try:
            redirect_server = DualStackServer(("::", 80), Port80RedirectHandler)
            threading.Thread(target=redirect_server.serve_forever, daemon=True).start()
            print(f"[Web] Port 80 forwarder active -> auto-redirects http://egrabbot.local to {proto}://egrabbot.local:{args.port}")
        except Exception as e:
            pass

    local_ip = get_local_ip()
    hostname = socket.gethostname()
    mdns_domain = f"{hostname}.local:{args.port}"

    print("\n" + "=" * 68)
    print(f"   🔒 EROVOUTIKAGRAB UNIFIED SERVER ACTIVE: {proto}://{mdns_domain}   ")
    print("=" * 68)
    print("Open in any browser:")
    print(f"   👉 {proto}://{mdns_domain}")
    print(f"   👉 {proto}://localhost:{args.port}")
    print(f"   👉 {proto}://{local_ip}:{args.port}")
    print(f"   👉 http://{hostname}.local (via Port 80 auto-forward to {proto})")
    print("Robot LCD (View-Only Mode):")
    print(f"   👉 {proto}://localhost:{args.port}/?device=lcd")
    print("=" * 68)
    print("Press Ctrl+C to stop.\n")

    _start_auto_reloader()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web dashboard...")
    finally:
        stop_vision_worker()
        stop_autonomous_mode()
        if comm:
            comm.send_stop()
            comm.disconnect()
        if camera:
            camera.stop()
        if redirect_server:
            redirect_server.server_close()
        httpd.server_close()
        print("Done.")


if __name__ == "__main__":
    main()
