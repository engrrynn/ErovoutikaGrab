"""Autonomous Activities Manager for ErovoutikaGrab.

Provides mutually-exclusive interactive robot activities that can be activated
one at a time from the Web Cockpit:
1. Person Follower: Tracks and pursues detected persons using on-device LiteRT.
2. Fast Color Tracking: Locks onto and pursues calibrated HSV color blobs (Red, Green, Blue, Yellow, Orange, Purple).
3. Generic Object Tracking: Tracks and pursues any of the 80 COCO classes using LiteRT AI.
4. Color Tracking + Object Classification: Fast HSV blob pursuit fused with LiteRT neural object classification.
5. Intelligent Object Sizing: Calibrated dimensional analysis (W, H, Area, Category, Gripper Fit).
6. Simple Obstacle Avoidance (Cam Version): Monocular ground corridor spatial division (L, C, R) and collision avoidance.
"""

import os
import sys
import time
import math
import threading
from typing import Optional, Dict, Any, Tuple, List
import cv2
import numpy as np
from src.adapters.vision.yoloe_adapter import DetectionResult

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Calibrated HSV Ranges for Color Tracking [Hue 0-180, Sat 0-255, Val 0-255]
HSV_COLOR_RANGES = {
    "Red": [
        (np.array([0, 50, 45]), np.array([12, 255, 255])),
        (np.array([168, 50, 45]), np.array([180, 255, 255])),
    ],
    "Green": [
        (np.array([35, 45, 40]), np.array([85, 255, 255])),
    ],
    "Blue": [
        (np.array([90, 45, 40]), np.array([135, 255, 255])),
    ],
    "Yellow": [
        (np.array([18, 50, 50]), np.array([36, 255, 255])),
    ],
    "Orange": [
        (np.array([10, 55, 50]), np.array([24, 255, 255])),
    ],
    "Purple": [
        (np.array([130, 45, 40]), np.array([168, 255, 255])),
    ],
    "Cyan": [
        (np.array([80, 45, 40]), np.array([102, 255, 255])),
    ],
}


def get_hsv_ranges(color_name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Case-insensitive HSV range lookup with safe fallback to Red."""
    if not color_name:
        return HSV_COLOR_RANGES["Red"]
    clean = str(color_name).strip().capitalize()
    if clean in HSV_COLOR_RANGES:
        return HSV_COLOR_RANGES[clean]
    for k, v in HSV_COLOR_RANGES.items():
        if k.lower() == clean.lower():
            return v
    return HSV_COLOR_RANGES["Red"]


# Standard metric conversion baseline calibrated to gripper landing sweet spot
# Reference: At standard ground landing distance, 256px sweet spot width = ~80mm gripper span.
MM_PER_PIXEL_SWEET_SPOT = 80.0 / 256.0  # ~0.3125 mm/pixel


class ActivityManager:
    """Manages mutually-exclusive autonomous activities with safe chassis control."""

    def __init__(self, camera=None, comm=None, vision=None, vlm_planner=None, arm=None):
        self.camera = camera
        self.comm = comm
        self.vision = vision
        self.vlm_planner = vlm_planner
        self.arm = arm

        self.lock = threading.Lock()
        self.active_activity: Optional[str] = None
        self.running: bool = False
        self.worker_thread: Optional[threading.Thread] = None

        # Temporal smoothing for sizing measurements (EMA filter)
        self._sizing_ema_w: Optional[float] = None
        self._sizing_ema_h: Optional[float] = None

        # Temporal smoothing & target locking for Person Follower (Gait EMA filter)
        self._pf_ema_x: Optional[float] = None
        self._pf_ema_ymax: Optional[float] = None
        self._pf_last_target_center: Optional[Tuple[int, int]] = None

        # Temporal smoothing for Color & Object Tracking (Anti-wobble filter)
        self._ct_ema_x: Optional[float] = None
        self._ct_ema_y: Optional[float] = None
        self._ot_ema_x: Optional[float] = None

        # Target spatial locking & persistence (Camera locks on what it sees first)
        self._locked_target_center: Optional[Tuple[int, int]] = None
        self._locked_target_label: Optional[str] = None
        self._locked_target_lost_frames: int = 0
        self._locked_target_frames: int = 0

        # Arm tracking state & latency synchronization
        self._arm_last_s1: Optional[int] = None
        self._arm_last_s2: Optional[int] = None
        self._arm_last_update_time: float = 0.0
        self._arm_holding: bool = False
        self._ik_settled_frames: int = 0
        self._is_grabbing: bool = False
        self._arm_settling: bool = False
        self._arm_settle_until: float = 0.0
        self._arm_deploy_event: threading.Event = threading.Event()
        self._arm_deploy_event.set()  # starts as ready (no blocking)

        # Obstacle avoidance Arm IK Environmental Scan state
        self._oa_scan_state: str = "IDLE"
        self._oa_scan_start_t: float = 0.0

        # Target dynamics tracking (Moving vs Stationary Target Detection & Slew Rate Smoothing)
        self._target_dyn_history: List[Tuple[float, float, float]] = []
        self._target_is_moving: bool = False
        self._last_cmd_l: int = 0
        self._last_cmd_r: int = 0

        # Configuration dictionary
        self.config: Dict[str, Any] = {
            "motion_enabled": True,             # Motion toggle: True=Move towards target, False=Just detect (stationary)
            "target_color": "Red",
            "target_object": "bottle",
            "sizing_target_object": "any",
            "follow_speed": 230,
            "turn_speed": 235,
            "obstacle_cruise_speed": 205,
            "obstacle_turn_speed": 225,
            "obstacle_threshold": 0.22,         # Collision score threshold (0.15 sensitive .. 0.35 relaxed)
            "person_target_height_ratio": 0.45,  # Person bbox height ~45% of image height (fallback)
            "person_target_ground_y": 0.83,     # Person foot contact point ~83% of image height
            "person_ground_tolerance": 0.06,    # Depth tolerance band (+/- 6%)
            "target_distance_cm": 55.0,         # Calibrated standoff distance for person follower (cm)
            "object_target_distance_cm": 13.5,  # Calibrated standoff distance for object tracking (cm)
            "color_target_distance_cm": 13.5,   # Calibrated standoff distance for color tracking (cm)
            "object_target_height_ratio": 0.35,  # Object bbox height ~35% of image height
            "sizing_target_height_ratio": 0.35,  # Ideal inspection height ratio
            "color_target_area_ratio": 0.08,    # Color blob area ~8% of image area
            "deadband_x": 30,                   # Pixels deadband for horizontal centering
            "trim_offset": 6,
        }

        # Real-time telemetry metrics
        self.telemetry: Dict[str, Any] = {
            "activity": "none",
            "running": False,
            "status": "IDLE",
            "motion_enabled": True,
            "target_found": False,
            "target_locked": False,
            "target_is_moving": False,
            "movement_mode": "SMOOTH_FAST",
            "distance_cm": 0.0,
            "error_x": 0,
            "error_y": 0,
            "target_size": 0.0,
            "ground_y": 0.0,
            "target_type": "none",
            "action": "STOPPED",
            "details": "Ready to activate activity.",
            "target_color": "Red",
            "target_object": "bottle",
            "classified_object": "none",
            "sizing": {
                "object": "none",
                "width_px": 0,
                "height_px": 0,
                "width_mm": 0.0,
                "height_mm": 0.0,
                "distance_cm": 0.0,
                "area_cm2": 0.0,
                "aspect_ratio": 0.0,
                "size_category": "NONE",
                "gripper_fit": "UNKNOWN",
                "gripper_msg": "--",
            },
            "obstacle": {
                "left_zone": "CLEAR",
                "center_zone": "CLEAR",
                "right_zone": "CLEAR",
                "left_score": 0.0,
                "center_score": 0.0,
                "right_score": 0.0,
                "left_dist_cm": 200.0,
                "center_dist_cm": 200.0,
                "right_dist_cm": 200.0,
                "action": "STOPPED",
                "clear_path": "CENTER",
            },
            "arm_ik": {
                "s1": 93,
                "s2": 45,
                "error_y": 0,
                "vertical_status": "CENTERED",
                "active": False,
                "latency_ms": 0.0,
            },
            "target_box": [],
            "leg_box": [],
            "target_polygon": [],
            "leg_polygon": [],
            "fps": 0.0,
            "vlm_plan": None,
            "motion_plan": None,
        }

        # Asynchronous perception pipeline for high-framerate real-time tracking
        self._perception_thread: Optional[threading.Thread] = None
        self._latest_detections: List[Any] = []
        self._latest_detection_time: float = 0.0
        self._vision_lock = threading.Lock()
        self._active_plan: Optional[Dict[str, Any]] = None

    def set_dependencies(self, camera=None, comm=None, vision=None, vlm_planner=None, arm=None):
        """Update live peripheral adapters."""
        with self.lock:
            if camera is not None:
                self.camera = camera
            if comm is not None:
                self.comm = comm
            if vision is not None:
                self.vision = vision
            if vlm_planner is not None:
                self.vlm_planner = vlm_planner
            if arm is not None:
                self.arm = arm

    def _get_turn_speed(self) -> int:
        """Reads configured baseline turn speed floor from robot_config.yaml or self.config.
        
        The turn speed set in robot_config.yaml is the hard baseline floor constraint (never less, but can scale higher).
        """
        baseline = int(self.config.get("turn_speed", 0))
        rc_path = os.path.join(BASE_DIR, "config/robot_config.yaml")
        if os.path.exists(rc_path):
            try:
                import yaml
                with open(rc_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                mcfg = cfg.get("motors", {})
                baseline = max(baseline, int(mcfg.get("turn_speed", 235)))
            except Exception:
                baseline = max(baseline, 235)
        else:
            baseline = max(baseline, 235)
        return max(235, baseline)

    def _get_base_speed(self) -> int:
        """Reads configured baseline drive speed floor from robot_config.yaml or self.config.
        
        The base speed set in robot_config.yaml is the hard baseline floor constraint (never less, even in forward and backward).
        """
        baseline = int(self.config.get("base_speed", 0))
        rc_path = os.path.join(BASE_DIR, "config/robot_config.yaml")
        if os.path.exists(rc_path):
            try:
                import yaml
                with open(rc_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                mcfg = cfg.get("motors", {})
                baseline = max(baseline, int(mcfg.get("base_speed", 235)))
            except Exception:
                baseline = max(baseline, 235)
        else:
            baseline = max(baseline, 235)
        return max(235, baseline)

    def _get_sweet_spot(self, w: int = 640, h: int = 480) -> Tuple[int, int]:
        """Returns calibrated (center_x, center_y) grasp sweet spot crosshair, scaled to frame resolution."""
        sx = int(self.config.get("sweet_spot_x", 0))
        sy = int(self.config.get("sweet_spot_y", 0))
        if sx <= 0 or sy <= 0:
            vc_path = os.path.join(BASE_DIR, "config/vision_config.yaml")
            if os.path.exists(vc_path):
                try:
                    import yaml
                    with open(vc_path, "r") as f:
                        vcfg = yaml.safe_load(f) or {}
                    gss = vcfg.get("grasp_sweet_spot", {})
                    sx = int(gss.get("center_x", 324))
                    sy = int(gss.get("center_y", 247))
                except Exception:
                    sx, sy = 324, 247
            else:
                sx, sy = 324, 247
        if w != 640 and w > 0:
            sx = int(round(sx * (w / 640.0)))
        if h != 480 and h > 0:
            sy = int(round(sy * (h / 480.0)))
        return sx, sy

    def _get_arm_limits(self) -> Dict[str, int]:
        """Reads calibrated servo angle bounds and stow baselines from arm controller or config."""
        if self.arm:
            return {
                "s1_min": self.arm.s1_min,
                "s1_max": self.arm.s1_max,
                "s1_stow": self.arm.s1_stow,
                "s1_down": self.arm.s1_down,
                "s2_min": self.arm.s2_min,
                "s2_max": self.arm.s2_max,
                "s2_stow": self.arm.s2_stow,
                "s2_down": self.arm.s2_down,
                "s3_open": getattr(self.arm, "s3_open", 170),
                "s3_close": getattr(self.arm, "s3_close", 40),
                "cur_s1": self.arm.cur_s1,
                "cur_s2": self.arm.cur_s2,
                "cur_s3": self.arm.cur_s3,
            }
        rc_path = os.path.join(BASE_DIR, "config/robot_config.yaml")
        if os.path.exists(rc_path):
            try:
                import yaml
                with open(rc_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                scfg = cfg.get("servos", {})
                s1 = scfg.get("servo1_shoulder", {})
                s2 = scfg.get("servo2_elbow", {})
                s3 = scfg.get("servo3_gripper", {})
                return {
                    "s1_min": int(s1.get("min_angle", 60)),
                    "s1_max": int(s1.get("max_angle", 170)),
                    "s1_stow": int(s1.get("up_angle", s1.get("stow_angle", 93))),
                    "s1_down": int(s1.get("down_angle", 170)),
                    "s2_min": int(s2.get("min_angle", 0)),
                    "s2_max": int(s2.get("max_angle", 45)),
                    "s2_stow": int(s2.get("up_angle", s2.get("stow_angle", 45))),
                    "s2_down": int(s2.get("down_angle", 0)),
                    "s3_open": int(s3.get("open_angle", 170)),
                    "s3_close": int(s3.get("close_angle", 40)),
                    "cur_s1": self._arm_last_s1 or int(s1.get("stow_angle", 93)),
                    "cur_s2": self._arm_last_s2 or int(s2.get("stow_angle", 45)),
                    "cur_s3": int(s3.get("open_angle", 170)),
                }
            except Exception:
                pass
        return {
            "s1_min": 60, "s1_max": 170, "s1_stow": 93, "s1_down": 170,
            "s2_min": 0, "s2_max": 45, "s2_stow": 45, "s2_down": 0,
            "s3_open": 170, "s3_close": 40,
            "cur_s1": self._arm_last_s1 or 93, "cur_s2": self._arm_last_s2 or 45, "cur_s3": 170,
        }

    def set_arm_down(self, open_gripper: bool = True, force: bool = False):
        """Moves servo arm into lowered ground pick reach pose (S1=down, S2=down).

        Used when starting and searching for objects or colors to prepare for ground grasping.
        """
        limits = self._get_arm_limits()
        s1_down = limits["s1_down"]
        s2_down = limits["s2_down"]
        s3 = limits.get("s3_open", 170) if open_gripper else limits.get("cur_s3", 170)

        if not force and self._arm_last_s1 == s1_down and self._arm_last_s2 == s2_down:
            return

        if self.arm:
            try:
                self.arm.cur_s1 = s1_down
                self.arm.cur_s2 = s2_down
                if open_gripper:
                    self.arm.cur_s3 = s3
                if hasattr(self.arm, "comm") and self.arm.comm:
                    self.arm.comm.send_servos(s1_down, s2_down, s3)
                if hasattr(self.arm, "_persist_state"):
                    self.arm._persist_state()
            except Exception as e:
                print(f"[Activities] Error setting arm down: {e}")
        elif self.comm:
            try:
                self.comm.send_servos(s1_down, s2_down, s3)
            except Exception as e:
                print(f"[Activities] Error sending servos down: {e}")

        self._arm_last_s1 = s1_down
        self._arm_last_s2 = s2_down
        self._arm_last_update_time = time.time()
        self.telemetry["arm_ik"] = {
            "s1": s1_down,
            "s2": s2_down,
            "error_y": 0,
            "vertical_status": "SETTLED_AT_LIMIT",
            "active": True,
            "status": "SETTLED",
            "latency_ms": 0.0,
        }

    def set_arm_up(self, force: bool = False):
        """Moves servo arm into upright stowed pose (S1=stow, S2=stow).

        Used when starting and searching during human following to maintain walking clearance.
        """
        limits = self._get_arm_limits()
        s1_stow = limits["s1_stow"]
        s2_stow = limits["s2_stow"]
        s3 = limits.get("cur_s3", 170)

        if not force and self._arm_last_s1 == s1_stow and self._arm_last_s2 == s2_stow:
            return

        if self.arm:
            try:
                self.arm.cur_s1 = s1_stow
                self.arm.cur_s2 = s2_stow
                if hasattr(self.arm, "comm") and self.arm.comm:
                    self.arm.comm.send_servos(s1_stow, s2_stow, s3)
                if hasattr(self.arm, "_persist_state"):
                    self.arm._persist_state()
            except Exception as e:
                print(f"[Activities] Error setting arm up: {e}")
        elif self.comm:
            try:
                self.comm.send_servos(s1_stow, s2_stow, s3)
            except Exception as e:
                print(f"[Activities] Error sending servos up: {e}")

        self._arm_last_s1 = s1_stow
        self._arm_last_s2 = s2_stow
        self._arm_last_update_time = time.time()
        self.telemetry["arm_ik"] = {
            "s1": s1_stow,
            "s2": s2_stow,
            "error_y": 0,
            "vertical_status": "STOWED",
            "active": True,
            "status": "SETTLED",
            "latency_ms": 0.0,
        }

    def _track_arm_elevation(self, cy: int, h: int, latency_s: float = 0.05, ymax: Optional[int] = None) -> Tuple[int, int]:
        """Decisive, non-oscillating inverse-kinematics arm elevation tracking to vertically center target in camera view.

        Features:
        - Rigidly locks servos in place once within the sweet spot deadband (holding state) with hysteresis.
        - Synchronizes with the calibrated grasp sweet spot (cy - sweet_spot_y).
        - Executes decisive quantized steps (min. 3 degrees) instead of 1-degree micro-hunting.
        - For ground pick activities (object/color tracking), holds arm DOWN in pick pose and verifies ground grasp envelope (ymax >= 445px).
        """
        now = time.time()
        limits = self._get_arm_limits()
        s1_min, s1_max = limits["s1_min"], limits["s1_max"]
        s1_stow, s1_down = limits["s1_stow"], limits["s1_down"]
        s2_min, s2_max = limits["s2_min"], limits["s2_max"]
        s2_stow, s2_down = limits["s2_stow"], limits["s2_down"]
        cur_s1 = self._arm_last_s1 or limits["cur_s1"]
        cur_s2 = self._arm_last_s2 or limits["cur_s2"]
        cur_s3 = limits["cur_s3"]

        # For ground pick activities (object/color tracking), the arm is already deployed DOWN in pick pose.
        # It must NOT pitch up towards horizon (sy=247), which falsely settles at 32-35cm distance!
        is_ground_activity = self.active_activity in ("object_tracking", "color_tracking", "color_track_and_classify", "object_sizing")
        if is_ground_activity and (cur_s1 >= s1_down - 15 and cur_s2 <= s2_down + 15):
            check_y = ymax if ymax is not None else cy
            in_grasp_reach = (check_y >= 445)
            status = "SETTLED_AT_LIMIT" if in_grasp_reach else "APPROACHING_GROUND"
            err_y = 0 if in_grasp_reach else max(1, 445 - check_y)

            self._arm_holding = True
            self.telemetry["arm_ik"] = {
                "s1": s1_down,
                "s2": s2_down,
                "error_y": err_y,
                "vertical_status": status,
                "active": True,
                "status": "SETTLED" if in_grasp_reach else "APPROACHING",
                "latency_ms": round(latency_s * 1000, 1),
            }
            return s1_down, s2_down

        # Calibrated vertical sweet spot from vision config / scaling
        _, sy = self._get_sweet_spot(640, h)
        ey = int(cy - sy)  # Positive: target is low (near floor). Negative: target is high.

        # Hysteresis Deadband Guard:
        # If holding steady, require error > 32px before moving.
        # If adjusting, stop and lock firmly when error <= 20px.
        deadband_y = 20
        breakout_y = 32
        is_centered = abs(ey) <= breakout_y if self._arm_holding else abs(ey) <= deadband_y

        if is_centered:
            self._arm_holding = True
            self.telemetry["arm_ik"] = {
                "s1": cur_s1,
                "s2": cur_s2,
                "error_y": ey,
                "vertical_status": "CENTERED",
                "active": True,
                "status": "SETTLED",
                "latency_ms": round(latency_s * 1000, 1),
            }
            return cur_s1, cur_s2

        self._arm_holding = False

        # Latency synchronization guard:
        min_interval = max(0.06, min(0.25, latency_s))
        if (now - self._arm_last_update_time) < min_interval:
            return cur_s1, cur_s2

        # Decisive Inverse Kinematics Mapping:
        # Object is lower in frame (ey > 0) -> Arm reaches DOWN towards floor
        # Object is higher in frame (ey < 0) -> Arm pitches UP towards stow pose
        norm_ey = max(-1.0, min(1.0, float(ey) / float(h * 0.38)))
        if norm_ey > 0:
            target_s1 = s1_stow + norm_ey * (s1_down - s1_stow)
            target_s2 = s2_stow - norm_ey * (s2_stow - s2_down)
        else:
            target_s1 = s1_stow - abs(norm_ey) * (s1_stow - s1_min)
            target_s2 = s2_stow

        # Quantize target into decisive 3-degree steps to eliminate single-degree hunting
        raw_s1 = int(round(target_s1))
        raw_s2 = int(round(target_s2))
        target_s1 = max(s1_min, min(s1_max, int(round(raw_s1 / 3.0) * 3)))
        target_s2 = max(s2_min, min(s2_max, int(round(raw_s2 / 3.0) * 3)))

        # Only dispatch if angular difference is decisive (>= 3 degrees)
        last_s1 = self._arm_last_s1 if self._arm_last_s1 is not None else cur_s1
        last_s2 = self._arm_last_s2 if self._arm_last_s2 is not None else cur_s2

        if (abs(target_s1 - last_s1) >= 3 or abs(target_s2 - last_s2) >= 3):
            if self.arm:
                try:
                    self.arm.cur_s1 = target_s1
                    self.arm.cur_s2 = target_s2
                    if hasattr(self.arm, "comm") and self.arm.comm:
                        self.arm.comm.send_servos(target_s1, target_s2, cur_s3)
                except Exception as e:
                    print(f"[Activities] Arm tracking move error: {e}")
            elif self.comm:
                try:
                    self.comm.send_servos(target_s1, target_s2, cur_s3)
                except Exception as e:
                    print(f"[Activities] Comm send_servos error: {e}")

            self._arm_last_s1 = target_s1
            self._arm_last_s2 = target_s2
            self._arm_last_update_time = now

        at_ground_limit = (target_s1 >= s1_down - 5) and (target_s2 <= s2_down + 5)
        if at_ground_limit:
            status = "SETTLED_AT_LIMIT"
        else:
            status = "PITCH_DOWN" if ey > 0 else "PITCH_UP"

        self.telemetry["arm_ik"] = {
            "s1": target_s1,
            "s2": target_s2,
            "error_y": ey,
            "vertical_status": status,
            "active": True,
            "status": "TRACKING",
            "latency_ms": round(latency_s * 1000, 1),
        }
        return target_s1, target_s2

    def _trigger_autonomous_grab(self, target_name: str, clamp_angle: Optional[int] = None) -> bool:
        """Initiates autonomous grab sequence when target has settled in the sweet spot."""
        if self._is_grabbing or not self.config.get("motion_enabled", True):
            return False

        self._is_grabbing = True
        self.telemetry["status"] = "GRABBING"
        self.telemetry["action"] = "INITIATE_GRAB"
        self.telemetry["details"] = f"IK settled in sweet spot! Executing autonomous grab sequence for {target_name}."

        if self.comm:
            self.comm.send_stop()

        def _grab_worker():
            try:
                print(f"[Activities] Autonomous grab initiated for '{target_name}'! Halting chassis and executing arm sequence...")
                time.sleep(0.2)
                if self.arm:
                    self.arm.reload_config()
                    self.arm.execute_pick_sequence(wait_completion=True, timeout_s=6.5, clamp_angle=clamp_angle)
                    self.telemetry["status"] = "OBJECT_SECURED"
                    self.telemetry["details"] = f"Object '{target_name}' successfully grasped and lifted to stow position!"
                    print(f"[Activities] Autonomous pick complete for '{target_name}'.")
                elif self.comm:
                    # Direct hardware servo sequence fallback
                    limits = self._get_arm_limits()
                    s1_stow, s1_down = limits["s1_stow"], limits["s1_down"]
                    s2_stow, s2_down = limits["s2_stow"], limits["s2_down"]
                    s3_open = limits["s3_open"]
                    target_s3_close = clamp_angle if clamp_angle is not None else limits["s3_close"]

                    # Step 1: Open gripper
                    self.comm.send_servos(s1_stow, s2_stow, s3_open)
                    time.sleep(0.4)
                    # Step 2: Extend down
                    self.comm.send_servos(s1_down, s2_down, s3_open)
                    time.sleep(0.7)
                    # Step 3: Clamp jaws
                    self.comm.send_servos(s1_down, s2_down, target_s3_close)
                    time.sleep(0.6)
                    # Step 4: Stow arm
                    self.comm.send_servos(s1_stow, s2_stow, target_s3_close)
                    time.sleep(0.7)
                    self.telemetry["status"] = "OBJECT_SECURED"
                    self.telemetry["details"] = f"Object '{target_name}' secured in gripper!"
            except Exception as e:
                print(f"[Activities] Autonomous grab sequence error: {e}")
                self.telemetry["details"] = f"Grab sequence error: {e}"
            finally:
                time.sleep(1.2)
                self._is_grabbing = False
                self._ik_settled_frames = 0

        threading.Thread(target=_grab_worker, daemon=True).start()
        return True

    def _plan_linkage_trajectory(
        self,
        target_name: str,
        cx: int,
        cy: int,
        w: int,
        h: int,
        dist_cm: float,
        target_dist_cm: float,
        is_moving: bool,
        target_poly: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Maps out the complete multi-step motion and arm IK plan to reach the target point or linkage.
        
        Calculates:
        1. Linkage Error Vector: dx (horizontal) and ey (vertical) from calibrated camera sweet spot crosshair (sx, sy).
        2. Arm Elevation IK Pose: Exact joint angles (S1*, S2*) pointing directly at target centroid.
        3. Multi-Step Trajectory Stages:
           - Stage 1: Heading Alignment (Pivot or Arc Turn to center linkage vector)
           - Stage 2: Approach Cruise (Continuous smooth forward drive to standoff)
           - Stage 3: Standoff Hold (Stable position hold within safe clearance)
        """
        sx, sy = self._get_sweet_spot(w, h)
        dx = cx - sx
        ey = cy - sy
        dist_err = dist_cm - target_dist_cm
        deadband_x = int(self.config.get("deadband_x", 30))
        b_speed = self._get_base_speed()
        f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
        t_speed = self._get_turn_speed()

        # 1. Arm IK Target Pose Calculation
        limits = self._get_arm_limits()
        s1_stow, s1_down = limits["s1_stow"], limits["s1_down"]
        s2_stow, s2_down = limits["s2_stow"], limits["s2_down"]
        s1_min, s1_max = limits["s1_min"], limits["s1_max"]
        s2_min, s2_max = limits["s2_min"], limits["s2_max"]

        norm_ey = max(-1.0, min(1.0, float(ey) / float(h * 0.40)))
        if norm_ey > 0:
            target_s1 = s1_stow + norm_ey * (s1_down - s1_stow)
            target_s2 = s2_stow - norm_ey * (s2_stow - s2_down)
        else:
            target_s1 = s1_stow - abs(norm_ey) * (s1_stow - s1_min)
            target_s2 = s2_stow

        target_s1 = max(s1_min, min(s1_max, int(round(target_s1))))
        target_s2 = max(s2_min, min(s2_max, int(round(target_s2))))

        # 2. Multi-Step Trajectory Stages Mapping
        stages = []
        turn_dir = "RIGHT" if dx > 0 else "LEFT"

        err_mag = abs(dx)
        err_ratio = min(1.0, max(0.0, (err_mag - deadband_x) / float(max(1, w // 3))))
        turn_pwm = max(t_speed, int(t_speed + err_ratio * (255 - t_speed)))

        # Heading Turn Stage
        if err_mag > deadband_x:
            est_turn_dur = max(0.10, min(0.35, 0.08 + (err_mag / float(w // 2)) * 0.20))
            if dist_err > 12.0:
                # Continuous Arc Steering Stage
                steer_bias = int(min(25, (err_mag / float(w // 2)) * 25))
                if dx > 0:
                    l_arc, r_arc = min(255, max(t_speed, f_speed + steer_bias)), max(b_speed, f_speed - steer_bias * 2)
                else:
                    l_arc, r_arc = max(b_speed, f_speed - steer_bias * 2), min(255, max(t_speed, f_speed + steer_bias))
                stages.append({
                    "stage": 1,
                    "name": f"ARC_ALIGN_{turn_dir}",
                    "action": f"TURN_{turn_dir}",
                    "left_pwm": l_arc,
                    "right_pwm": r_arc,
                    "duration_s": est_turn_dur,
                    "continuous": True,
                    "desc": f"Continuous arc steering {turn_dir.lower()} along linkage vector (dx={dx:+d}px)."
                })
            else:
                lpwm = turn_pwm if dx > 0 else -turn_pwm
                rpwm = -turn_pwm if dx > 0 else turn_pwm
                stages.append({
                    "stage": 1,
                    "name": f"PIVOT_ALIGN_{turn_dir}",
                    "action": f"TURN_{turn_dir}",
                    "left_pwm": lpwm,
                    "right_pwm": rpwm,
                    "duration_s": est_turn_dur,
                    "continuous": False,
                    "desc": f"Pivot alignment {turn_dir.lower()} to center linkage vector (dx={dx:+d}px)."
                })

        # Approach or Standoff Stage
        approach_band = 1.5 if target_dist_cm <= 15.0 else 8.0
        reverse_band = -3.5 if target_dist_cm <= 15.0 else -6.0

        if dist_err > approach_band:
            est_drive_dur = max(0.15, min(1.2, (dist_err / 80.0) * 0.6))
            stages.append({
                "stage": len(stages) + 1,
                "name": "APPROACH_CRUISE",
                "action": "APPROACH",
                "left_pwm": f_speed,
                "right_pwm": f_speed,
                "duration_s": est_drive_dur,
                "continuous": True,
                "desc": f"Continuous cruise along planned linkage to target standoff {target_dist_cm:.0f}cm."
            })
        elif dist_err < reverse_band:
            stages.append({
                "stage": len(stages) + 1,
                "name": "REVERSE_CLEAR",
                "action": "BACK_UP",
                "left_pwm": -f_speed,
                "right_pwm": -f_speed,
                "duration_s": 0.15,
                "continuous": False,
                "desc": f"Clearance reverse: Target {dist_cm:.0f}cm closer than standoff {target_dist_cm:.0f}cm."
            })
        else:
            stages.append({
                "stage": len(stages) + 1,
                "name": "STANDOFF_HOLD",
                "action": "ALIGNED_HOLD",
                "left_pwm": 0,
                "right_pwm": 0,
                "duration_s": 0.15,
                "continuous": False,
                "desc": f"Aligned at standoff {dist_cm:.0f}cm. Holding position."
            })

        active_stage = stages[0]
        return {
            "target": target_name,
            "target_point": [int(cx), int(cy)],
            "sweet_spot_point": [int(sx), int(sy)],
            "dist_cm": round(dist_cm, 1),
            "target_dist_cm": round(target_dist_cm, 1),
            "dx_px": int(dx),
            "ey_px": int(ey),
            "arm_target": {"s1": target_s1, "s2": target_s2},
            "active_stage": active_stage,
            "stages": stages,
            "is_moving": is_moving,
            "total_duration_s": round(sum(s["duration_s"] for s in stages), 2)
        }

    def toggle_motion(self, enabled: Optional[bool] = None) -> bool:
        """Toggles or sets the chassis motion mode (Autonomous Move vs Just Detect)."""
        with self.lock:
            if enabled is None:
                self.config["motion_enabled"] = not bool(self.config.get("motion_enabled", True))
            else:
                self.config["motion_enabled"] = bool(enabled)
            val = bool(self.config["motion_enabled"])
            self.telemetry["motion_enabled"] = val
            if not val and self.comm:
                self.comm.send_stop()
            return val

    @property
    def motion_enabled(self) -> bool:
        return bool(self.config.get("motion_enabled", True))

    def estimate_ground_distance(self, bottom_y: int, frame_h: int = 480) -> float:
        """Calculates physically grounded distance (in cm) from camera to floor contact point.
        
        Calibrated with camera mount height (14cm), downward tilt (16 deg = 0.28 rad),
        and 55 deg vertical FOV (fy ~ 460px on 480p).
        """
        if bottom_y <= 0:
            return 250.0
        dy = float(bottom_y - (frame_h / 2.0))
        angle_alpha = math.atan(dy / 460.0)
        total_angle = 0.28 + angle_alpha
        if total_angle <= 0.08:
            return 250.0
        dist_cm = 14.0 / math.tan(total_angle)
        return float(round(max(10.0, min(250.0, dist_cm)), 1))

    def _evaluate_target_dynamics(self, cx: float, cy: float, now: float) -> bool:
        """Determines if the target object is moving or stationary based on temporal tracking history.
        
        Applies ego-motion compensation:
        - When the robot is actively driving forward, vertical motion in the camera is due to vehicle approach.
          Genuine physical target motion is detected via lateral velocity deviation.
        - When the robot is stationary, full 2D Euclidean velocity is evaluated.
        """
        self._target_dyn_history.append((now, float(cx), float(cy)))
        cutoff = now - 0.70
        self._target_dyn_history = [pt for pt in self._target_dyn_history if pt[0] >= cutoff]

        if len(self._target_dyn_history) >= 3:
            t0, x0, y0 = self._target_dyn_history[0]
            t1, x1, y1 = self._target_dyn_history[-1]
            dt = t1 - t0
            if dt >= 0.12:
                is_chassis_driving = (self._last_cmd_l != 0 or self._last_cmd_r != 0)
                if is_chassis_driving:
                    # When robot is moving, ignore vertical looming motion; inspect lateral drift
                    speed = abs(x1 - x0) / dt
                    moving_threshold = 40.0
                    stationary_threshold = 25.0
                else:
                    speed = math.hypot(x1 - x0, y1 - y0) / dt
                    moving_threshold = 30.0
                    stationary_threshold = 18.0

                if self._target_is_moving:
                    if speed < stationary_threshold:
                        self._target_is_moving = False
                else:
                    if speed > moving_threshold:
                        self._target_is_moving = True

        self.telemetry["target_is_moving"] = self._target_is_moving
        self.telemetry["movement_mode"] = "SMOOTH_PULSE" if self._target_is_moving else "SMOOTH_FAST"
        return self._target_is_moving

    def _execute_gated_pulse(
        self,
        lpwm: int,
        rpwm: int,
        duration_s: float = 0.12,
        settle_s: float = 0.04,
        is_target_moving: Optional[bool] = None,
        continuous_drive: bool = False
    ):
        """Executes VLM-planned motion actuation:
        
        - In Stationary Mode (motion_enabled=False): Inhibits forward/reverse translation, executes smooth stationary yaw turns.
        - In Autonomous Mode (motion_enabled=True):
          * When continuous_drive=True: Commands continuous smooth drive along the efficient path without frame-to-frame sleep or stall pauses.
          * When continuous_drive=False: Executes finite micro-adjustment pulse without motor chatter.
        """
        is_stationary = not bool(self.config.get("motion_enabled", True))
        if is_target_moving is None:
            is_target_moving = self._target_is_moving

        if is_stationary:
            # In Stationary Mode: Forward/reverse translation is inhibited.
            # Sideways yaw turning to center the target horizontally is permitted and executed!
            if lpwm == rpwm or (lpwm == 0 and rpwm == 0):
                if self.comm and (self._last_cmd_l != 0 or self._last_cmd_r != 0):
                    self.comm.send_drive(0, 0)
                    self._last_cmd_l = 0
                    self._last_cmd_r = 0
                return

            # Convert any forward/reverse-biased arc turn to pure stationary pivot turn in place
            turn_speed = self._get_turn_speed()
            if lpwm > rpwm:
                pivot_l, pivot_r = turn_speed, -turn_speed
            else:
                pivot_l, pivot_r = -turn_speed, turn_speed

            duration_s = max(0.06, min(0.12, duration_s))
            if self.comm:
                p_l, p_r = self._apply_trim_direct(pivot_l, pivot_r)
                self.comm.send_drive(p_l, p_r)
                time.sleep(duration_s)
                self.comm.send_drive(0, 0)
                self._last_cmd_l = 0
                self._last_cmd_r = 0
            return

        if not self.comm:
            return

        # Target reached or halted
        if lpwm == 0 and rpwm == 0:
            if self._last_cmd_l != 0 or self._last_cmd_r != 0:
                self.comm.send_drive(0, 0)
                self._last_cmd_l = 0
                self._last_cmd_r = 0
            return

        # Hard baseline speed floor constraints from robot_config.yaml:
        # Never less than base_speed for forward/backward and never less than turn_speed for turning!
        b_speed = self._get_base_speed()
        t_speed = self._get_turn_speed()

        if lpwm > 0 and rpwm > 0:
            # Forward drive: both wheels must be at least base_speed
            lpwm = max(b_speed, lpwm)
            rpwm = max(b_speed, rpwm)
        elif lpwm < 0 and rpwm < 0:
            # Backward drive: both wheels must be at least base_speed in reverse magnitude
            lpwm = min(-b_speed, lpwm)
            rpwm = min(-b_speed, rpwm)
        elif (lpwm > 0 and rpwm < 0) or (lpwm < 0 and rpwm > 0):
            # Pivot turn: both wheels must be at least turn_speed in magnitude
            if lpwm > 0:
                lpwm = max(t_speed, lpwm)
                rpwm = min(-t_speed, rpwm)
            else:
                lpwm = min(-t_speed, lpwm)
                rpwm = max(t_speed, rpwm)
        elif lpwm != 0 or rpwm != 0:
            # Single-wheel / differential forward/backward drive:
            if lpwm > 0:
                lpwm = max(b_speed, lpwm)
            elif lpwm < 0:
                lpwm = min(-b_speed, lpwm)
            if rpwm > 0:
                rpwm = max(b_speed, rpwm)
            elif rpwm < 0:
                rpwm = min(-b_speed, rpwm)

        # Slew-rate transition: when starting from stop, jump to target to overcome stiction; while moving, smooth transitions
        max_slew = 35
        def _slew(cur: int, target: int) -> int:
            if cur == 0:
                return target
            delta = target - cur
            if delta > max_slew:
                return cur + max_slew
            elif delta < -max_slew:
                return cur - max_slew
            return target

        adj_l, adj_r = self._apply_trim_direct(lpwm, rpwm)
        cmd_l = _slew(self._last_cmd_l, adj_l)
        cmd_r = _slew(self._last_cmd_r, adj_r)
        self._last_cmd_l = cmd_l
        self._last_cmd_r = cmd_r

        # CONTINUOUS EFFICIENT PATH MOVEMENT:
        # Drives along the planned trajectory smoothly and continuously without zero-brake or 40% PWM stall pauses!
        self.comm.send_drive(cmd_l, cmd_r)
        if not continuous_drive and duration_s > 0:
            time.sleep(min(0.08, duration_s))

    def start_activity(self, name: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Starts a specific activity, ensuring only one runs at a time."""
        clean_name = name.strip().lower()
        if clean_name in ("color_track_and_object_classification", "color_tracking_classification"):
            clean_name = "color_track_and_classify"
        elif clean_name in ("sizing", "measure", "object_measure", "measurement", "dimensioning"):
            clean_name = "object_sizing"
        elif clean_name in (
            "obstacle_avoidance",
            "obstacle",
            "avoidance",
            "cam_avoidance",
            "simple_obstacle_avoidance",
            "simple_obstacle_avoidance_cam_version",
            "cam_obstacle_avoidance"
        ):
            clean_name = "obstacle_avoidance"

        valid_activities = (
            "person_follower",
            "color_tracking",
            "object_tracking",
            "color_track_and_classify",
            "object_sizing",
            "obstacle_avoidance",
        )
        if clean_name not in valid_activities:
            print(f"[Activities] Unknown activity requested: {name}")
            return False

        with self.lock:
            # Stop any existing activity first (strictly one at a time)
            self._stop_internal_unsafe()

            # Reset sizing temporal filter
            self._sizing_ema_w = None
            self._sizing_ema_h = None

            # Reset target spatial locking & persistence filter
            self._locked_target_center = None
            self._locked_target_label = None
            self._locked_target_lost_frames = 0
            self._locked_target_frames = 0

            if params:
                for k, v in params.items():
                    if k in self.config:
                        self.config[k] = v
                    elif k == "target_area_ratio":
                        self.config["color_target_area_ratio"] = v
                    elif k == "target_dist_ratio":
                        self.config["person_target_height_ratio"] = v
                    elif k == "object_dist_ratio":
                        self.config["object_target_height_ratio"] = v
                    elif k == "sizing_dist_ratio":
                        self.config["sizing_target_height_ratio"] = v
                    elif k == "sensitivity":
                        # Map sensitivity slider (0.1..0.5) to threshold
                        self.config["obstacle_threshold"] = float(v)

            # Ensure normalized target_color and target_object
            if "target_color" in self.config:
                c_norm = str(self.config["target_color"]).strip().capitalize()
                self.config["target_color"] = c_norm
                self.telemetry["target_color"] = c_norm
            if "target_object" in self.config:
                self.telemetry["target_object"] = str(self.config["target_object"]).strip().lower()

            self.active_activity = clean_name
            self.running = True
            self.telemetry["activity"] = clean_name
            self.telemetry["running"] = True
            self.telemetry["status"] = "STARTING"
            self.telemetry["details"] = f"Initializing {clean_name.replace('_', ' ').title()}..."

            # Position servo arm at start:
            # - Object & color tracking: arm DOWN pose FIRST, hard-block detection until settled
            # - Human following: arm UP at start (stowed upright pose for walking clearance)
            if clean_name in ("object_tracking", "color_tracking", "color_track_and_classify", "object_sizing"):
                already_down = (self._arm_last_s1 == 170 and self._arm_last_s2 == 0)
                # Hard-block: clear the event so both perception and activity threads wait
                self._arm_deploy_event.clear()
                self._arm_settling = True
                self.set_arm_down(open_gripper=True, force=True)
                # Arduino firmware moves 1°/20ms: S1 77° + S2 45° ≈ 1.54s. Use 1.8s for safety.
                settle_dur = 0.25 if already_down else 1.8
                self._arm_settle_until = time.time() + settle_dur
                self.telemetry["status"] = "DEPLOYING_ARM"
                self.telemetry["action"] = "LOWERING_ARM"
                self.telemetry["details"] = f"Lowering servo arm to ground grasp pose ({settle_dur:.1f}s)... Detection blocked until complete."

                # Deferred release: a short-lived thread that sleeps for the full settle duration,
                # then fires the event so both _perception_worker and _activity_loop unblock atomically.
                def _arm_settle_release(dur: float):
                    time.sleep(dur)
                    self._arm_settling = False
                    self._arm_deploy_event.set()
                    print(f"[Activities] Arm settled after {dur:.1f}s — detection unblocked.")
                threading.Thread(target=_arm_settle_release, args=(settle_dur,), daemon=True).start()
            elif clean_name in ("person_follower", "obstacle_avoidance"):
                self.set_arm_up(force=True)
                self._arm_settling = False
                self._arm_settle_until = 0.0
                self._arm_deploy_event.set()  # ensure no block

            # Launch asynchronous perception worker for high-throughput tracking
            if clean_name in ("person_follower", "object_tracking", "color_track_and_classify") and self.camera and self.vision:
                self._perception_thread = threading.Thread(target=self._perception_worker, daemon=True)
                self._perception_thread.start()

            self.worker_thread = threading.Thread(target=self._activity_loop, daemon=True)
            self.worker_thread.start()
            print(f"[Activities] Started activity: {clean_name} (params={params})")
            return True

    def _perception_worker(self):
        """Asynchronous vision perception worker that runs detection in background to sustain high framerate."""
        # Hard-block: wait for arm deployment to complete before ANY detection
        self._arm_deploy_event.wait()

        while self.running:

            act = self.active_activity
            if act not in ("person_follower", "object_tracking", "color_track_and_classify"):
                time.sleep(0.04)
                continue

            frame = None
            if self.camera and self.camera.is_opened():
                try:
                    frame = self.camera.get_frame()
                except Exception:
                    frame = None

            if frame is None or not self.vision:
                time.sleep(0.03)
                continue

            try:
                dets = self.vision.detect_all(frame)
                with self._vision_lock:
                    self._latest_detections = dets
                    self._latest_detection_time = time.time()
            except Exception as e:
                pass

            time.sleep(0.01)

    def stop_activity(self) -> bool:
        """Stops the active activity and halts robot motion safely."""
        with self.lock:
            return self._stop_internal_unsafe()

    def _stop_internal_unsafe(self) -> bool:
        """Internal helper to halt execution. Assumes lock is held."""
        self.running = False
        self.active_activity = None
        self._sizing_ema_w = None
        self._sizing_ema_h = None
        self._pf_ema_x = None
        self._pf_ema_ymax = None
        self._pf_last_target_center = None
        self._ct_ema_x = None
        self._ct_ema_y = None
        self._ot_ema_x = None
        self._locked_target_center = None
        self._locked_target_label = None
        self._locked_target_lost_frames = 0
        self._locked_target_frames = 0
        self._target_dyn_history.clear()
        self._target_is_moving = False
        self._last_cmd_l = 0
        self._last_cmd_r = 0
        self._arm_settling = False
        self._arm_settle_until = 0.0
        self._arm_deploy_event.set()  # release any threads blocked on arm deploy wait
        self._perception_thread = None
        with self._vision_lock:
            self._latest_detections = []
            self._latest_detection_time = 0.0
        self._active_plan = None
        self.telemetry["activity"] = "none"
        self.telemetry["running"] = False
        self.telemetry["status"] = "IDLE"
        self.telemetry["target_found"] = False
        self.telemetry["target_locked"] = False
        self.telemetry["target_is_moving"] = False
        self.telemetry["movement_mode"] = "SMOOTH_FAST"
        self.telemetry["target_box"] = []
        self.telemetry["leg_box"] = []
        self.telemetry["target_polygon"] = []
        self.telemetry["leg_polygon"] = []
        self.telemetry["motion_plan"] = None
        self.telemetry["target_type"] = "none"
        self.telemetry["ground_y"] = 0.0
        self.telemetry["action"] = "STOPPED"
        self.telemetry["details"] = "Activity stopped. Standing by."
        self.telemetry["obstacle"] = {
            "left_zone": "CLEAR",
            "center_zone": "CLEAR",
            "right_zone": "CLEAR",
            "left_score": 0.0,
            "center_score": 0.0,
            "right_score": 0.0,
            "action": "STOPPED",
            "clear_path": "CENTER",
        }
        self.telemetry["arm_ik"]["active"] = False
        self.telemetry["arm_ik"]["vertical_status"] = "IDLE"

        if self.comm:
            try:
                self.comm.send_stop()
            except Exception as e:
                print(f"[Activities] Error sending stop to comm: {e}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns real-time status and telemetry dictionary."""
        with self.lock:
            res = dict(self.telemetry)
            res["active_activity"] = self.active_activity or "none"
            res["running"] = self.running
            res["config"] = dict(self.config)
            return res

    def _apply_trim(self, speed: int) -> Tuple[int, int]:
        """Calculates left and right motor PWM applying chassis trim offset."""
        trim = int(self.config.get("trim_offset", 6))
        lpwm = speed - trim
        rpwm = speed + trim
        return int(lpwm), int(rpwm)

    def _apply_trim_direct(self, lpwm: int, rpwm: int) -> Tuple[int, int]:
        """Applies trim offset to arbitrary left and right PWM values, keeping in valid bounds."""
        trim = int(self.config.get("trim_offset", 6))
        return int(max(-255, min(255, lpwm - trim))), int(max(-255, min(255, rpwm + trim)))

    def _compute_diminishing_turn_pwm(self, ex: int, w: int) -> int:
        """Computes diminishing turn PWM scaling down to baseline turn speed floor as error decreases."""
        deadband_x = int(self.config.get("deadband_x", 25))
        min_overcome = self._get_turn_speed()
        max_turn = max(min_overcome, 248)
        err_mag = abs(ex)
        if err_mag <= deadband_x:
            return 0
        ratio = min(1.0, max(0.0, (err_mag - deadband_x) / float(max(1, w // 3))))
        return int(min_overcome + ratio * (max_turn - min_overcome))

    def _activity_loop(self):
        """Background control loop running at ~15-20 Hz."""
        last_seen_time = 0.0

        # Hard-block: wait for arm deployment to complete before ANY detection or driving.
        # Motors are held stopped. The _arm_settle_release thread fires the event when done.
        if self._arm_settling:
            self.telemetry["status"] = "DEPLOYING_ARM"
            self.telemetry["action"] = "LOWERING_ARM"
            self.telemetry["details"] = "Arm deploying to ground pose — detection blocked until settled."
            if self.comm:
                try:
                    self.comm.send_stop()
                except Exception:
                    pass
            self._arm_deploy_event.wait()

        while self.running:

            t0 = time.time()
            frame = None
            if self.camera and self.camera.is_opened():
                try:
                    frame = self.camera.get_frame()
                except Exception:
                    frame = None

            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            cx_img = w // 2

            act = self.active_activity
            if act == "person_follower":
                last_seen_time = self._step_person_follower(frame, w, h, cx_img, last_seen_time)
            elif act == "color_tracking":
                last_seen_time = self._step_color_tracking(frame, w, h, cx_img, last_seen_time)
            elif act == "object_tracking":
                last_seen_time = self._step_object_tracking(frame, w, h, cx_img, last_seen_time)
            elif act == "color_track_and_classify":
                last_seen_time = self._step_color_track_and_classify(frame, w, h, cx_img, last_seen_time)
            elif act == "object_sizing":
                last_seen_time = self._step_object_sizing(frame, w, h, cx_img, last_seen_time)
            elif act == "obstacle_avoidance":
                last_seen_time = self._step_obstacle_avoidance(frame, w, h, cx_img, last_seen_time)
            else:
                break

            dt = time.time() - t0
            if dt > 0:
                self.telemetry["fps"] = round(1.0 / dt, 1)

            # Regulate loop to ~15-20 Hz
            time.sleep(max(0.01, 0.05 - dt))

        # Ensure motors stop when loop exits
        if self.comm:
            try:
                self.comm.send_stop()
            except Exception:
                pass
        with self.lock:
            if not self.running:
                self.telemetry["status"] = "IDLE"
                self.telemetry["action"] = "STOPPED"
                self.telemetry["target_found"] = False
                self.telemetry["target_box"] = []
                self.telemetry["details"] = "Activity stopped. Standing by."

    def _step_person_follower(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Processes frame for human tracking with high-priority leg/lower-body grounding and upper-body fallback."""
        now = time.time()

        # Semantic class pools
        LEG_CLASSES = {
            "leg", "legs", "pants", "jeans", "trousers", "shorts",
            "shoe", "shoes", "boot", "boots", "sneaker", "sneakers", "foot", "feet"
        }
        BODY_CLASSES = {"person", "human", "man", "woman", "boy", "girl"}
        UPPER_BODY_CLASSES = {"torso", "shirt", "jacket", "coat", "hoodie", "sweater"}

        leg_candidates = []
        body_candidates = []
        upper_candidates = []

        detections = []
        with self._vision_lock:
            if self._latest_detections and (now - self._latest_detection_time) < 1.2:
                detections = list(self._latest_detections)
        # Direct synchronous detection if perception thread not running (e.g. tests or startup)
        if not detections and self.vision:
            try:
                detections = self.vision.detect_all(frame)
            except Exception as e:
                print(f"[Activities] Vision inference exception: {e}")

        for d in detections:
            cat = d.category.lower().strip()
            bb = d.bounding_box
            if not bb or len(bb) != 4:
                continue
            ymin, xmin, ymax, xmax = [int(v) for v in bb]
            if xmax <= xmin or ymax <= ymin:
                continue
            poly = getattr(d, "mask_polygon", None)

            # 1. Direct Leg / Lower-body detection (Priority 1)
            if any(c in cat for c in LEG_CLASSES) and d.confidence >= 0.18:
                leg_candidates.append((ymin, xmin, ymax, xmax, d.confidence, cat, poly))
            # 2. Whole body / person detection (Priority 2)
            elif any(c == cat or c in cat for c in BODY_CLASSES) and d.confidence >= 0.20:
                body_candidates.append((ymin, xmin, ymax, xmax, d.confidence, cat, poly))
            # 3. Upper body detection (Priority 3 fallback)
            elif any(c in cat for c in UPPER_BODY_CLASSES) and d.confidence >= 0.22:
                upper_candidates.append((ymin, xmin, ymax, xmax, d.confidence, cat, poly))

        # Target selection with LEGS AS TOP PRIORITY
        target_box = None
        leg_box = None
        target_type = "NONE"
        chosen_poly = None
        px_center = cx_img
        px_ground_y = int(h * 0.83)
        height_ratio = 0.0

        # Helper: Bipedal fusion (merges left + right leg boxes into single bipedal stance)
        def _fuse_legs(legs):
            if not legs:
                return None
            if len(legs) == 1:
                return legs[0][:4]
            legs_sorted = sorted(legs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            l1 = legs_sorted[0]
            c1_x = (l1[1] + l1[3]) // 2
            for l2 in legs_sorted[1:]:
                c2_x = (l2[1] + l2[3]) // 2
                if abs(c1_x - c2_x) < int(w * 0.35) and abs(l1[2] - l2[2]) < int(h * 0.25):
                    return (
                        min(l1[0], l2[0]),
                        min(l1[1], l2[1]),
                        max(l1[2], l2[2]),
                        max(l1[3], l2[3]),
                    )
            return l1[:4]

        # Scenario A: Both Body and Legs Detected -> Fused Person with Leg Priority
        if body_candidates and leg_candidates:
            body_candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            primary_body = body_candidates[0]
            by0, bx0, by1, bx1 = primary_body[:4]

            # Find legs associated with this body
            assoc_legs = [l for l in leg_candidates if not (l[3] < bx0 - 30 or l[1] > bx1 + 30)]
            fused_leg = _fuse_legs(assoc_legs) if assoc_legs else _fuse_legs(leg_candidates)

            target_box = (by0, bx0, by1, bx1)
            leg_box = fused_leg
            target_type = "PERSON_WITH_LEGS"
            chosen_poly = primary_body[6] or (leg_candidates[0][6] if leg_candidates else None)
            px_center = (fused_leg[1] + fused_leg[3]) // 2
            px_ground_y = fused_leg[2]
            height_ratio = (by1 - by0) / float(h)

        # Scenario B: ONLY Legs Detected -> Top Priority
        elif leg_candidates:
            fused_leg = _fuse_legs(leg_candidates)
            target_box = fused_leg
            leg_box = fused_leg
            target_type = "BIPEDAL_LEGS" if len(leg_candidates) > 1 else "LEGS"
            chosen_poly = leg_candidates[0][6]
            px_center = (fused_leg[1] + fused_leg[3]) // 2
            px_ground_y = fused_leg[2]
            height_ratio = (fused_leg[2] - fused_leg[0]) / float(h)

        # Scenario C: ONLY Whole Body Detected -> Use body with ground-contact ymax
        elif body_candidates:
            if self._pf_last_target_center is not None:
                lx, ly = self._pf_last_target_center
                body_candidates.sort(key=lambda b: ((b[1] + b[3]) // 2 - lx) ** 2 + ((b[0] + b[2]) // 2 - ly) ** 2)
            else:
                body_candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

            b = body_candidates[0]
            by0, bx0, by1, bx1 = b[:4]
            target_box = (by0, bx0, by1, bx1)
            target_type = "WHOLE_BODY"
            chosen_poly = b[6]
            px_center = (bx0 + bx1) // 2
            px_ground_y = by1
            height_ratio = (by1 - by0) / float(h)
            # Synthesize leg zone in bottom 40% of box
            leg_y_top = int(by0 + (by1 - by0) * 0.60)
            leg_box = (leg_y_top, bx0, by1, bx1)

        # Scenario D: ONLY Upper Body Detected -> Fallback to upper body centroid/height
        elif upper_candidates:
            upper_candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            u = upper_candidates[0]
            uy0, ux0, uy1, ux1 = u[:4]
            target_box = (uy0, ux0, uy1, ux1)
            leg_box = []
            target_type = "UPPER_BODY"
            chosen_poly = u[6]
            px_center = (ux0 + ux1) // 2
            px_ground_y = uy1
            height_ratio = (uy1 - uy0) / float(h)

        # Process actuation if target was found
        if target_box is not None:
            last_seen = now
            ymin, xmin, ymax, xmax = target_box
            self._pf_last_target_center = (px_center, (ymin + ymax) // 2)

            # Store segmentation polygon contour in telemetry for live HUD/stream display
            if chosen_poly and len(chosen_poly) >= 3:
                self.telemetry["target_polygon"] = [[int(pt[0]), int(pt[1])] for pt in chosen_poly]
            else:
                self.telemetry["target_polygon"] = [
                    [int(xmin), int(ymin)], [int(xmax), int(ymin)],
                    [int(xmax), int(ymax)], [int(xmin), int(ymax)]
                ]

            # Temporal EMA Smoothing for horizontal center and ground depth
            if self._pf_ema_x is None:
                self._pf_ema_x = float(px_center)
                self._pf_ema_ymax = float(px_ground_y)
            else:
                self._pf_ema_x = 0.35 * float(px_center) + 0.65 * self._pf_ema_x
                self._pf_ema_ymax = 0.40 * float(px_ground_y) + 0.60 * self._pf_ema_ymax

            smooth_cx = int(self._pf_ema_x)
            smooth_gy = int(self._pf_ema_ymax)

            sx, sy = self._get_sweet_spot(w, h)
            ex = smooth_cx - sx
            ey = smooth_gy - sy
            ground_ratio = smooth_gy / float(h)

            # Calibrated Physical Distance in Centimeters
            dist_cm = self.estimate_ground_distance(smooth_gy, h)
            self.telemetry["distance_cm"] = dist_cm

            target_dist = float(self.config.get("target_distance_cm", 55.0))
            dist_tolerance = 10.0  # +/- 10cm band

            if dist_cm > (target_dist + dist_tolerance):
                dist_action = "APPROACH"
            elif dist_cm < (target_dist - dist_tolerance):
                dist_action = "BACK_UP"
            else:
                dist_action = "ALIGNED"

            deadband_x = int(self.config.get("deadband_x", 30))
            b_speed = self._get_base_speed()
            f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
            t_speed = self._get_turn_speed()
            turn_pwm = self._compute_diminishing_turn_pwm(ex, w)
            pulse_dur = 0.12

            # Evaluate target dynamics (Moving vs Stationary Target)
            target_cy = (ymin + ymax) // 2
            is_moving = self._evaluate_target_dynamics(px_center, target_cy, now)

            # 1. MAP OUT ALL MOVEMENT & KINEMATICS FIRST
            motion_plan = self._plan_linkage_trajectory(
                target_name=target_type,
                cx=smooth_cx,
                cy=target_cy,
                w=w,
                h=h,
                dist_cm=dist_cm,
                target_dist_cm=target_dist,
                is_moving=is_moving,
                target_poly=self.telemetry.get("target_polygon")
            )
            self.telemetry["motion_plan"] = motion_plan
            active_stage = motion_plan["active_stage"]
            left_pwm = active_stage["left_pwm"]
            right_pwm = active_stage["right_pwm"]
            action = active_stage["action"]
            continuous_drive = active_stage.get("continuous", False)
            pulse_dur = active_stage.get("duration_s", 0.12)

            # Optional VLM planner integration (deterministic, fast with use_vlm_llm=False)
            if self.vlm_planner:
                sweet_spot = {"center_x": sx, "center_y": sy, "box_width": int(w * 0.35), "box_height": int(h * 0.35)}
                mock_det = DetectionResult(
                    detected=True,
                    category=target_type.lower(),
                    confidence=0.90,
                    bounding_box=(ymin, xmin, ymax, xmax),
                    material_color="Dynamic",
                    mask_polygon=self.telemetry.get("target_polygon")
                )
                vlm_plan = self.vlm_planner.plan_movement(
                    mock_det,
                    sweet_spot,
                    use_vlm_llm=False,
                    dist_cm=dist_cm,
                    target_dist_cm=float(target_dist),
                    target_is_moving=is_moving
                )
                self.telemetry["vlm_plan"] = vlm_plan.to_dict()
                if vlm_plan.continuous_drive:
                    left_pwm = vlm_plan.left_pwm
                    right_pwm = vlm_plan.right_pwm
                    action = vlm_plan.action
                    continuous_drive = True
                    pulse_dur = max(0.08, min(0.20, vlm_plan.duration_ms / 1000.0))

                # Hard turn and base speed floor guarantees from robot config
                if "FORWARD" in action or "APPROACH" in action or "CRUISE" in action:
                    if left_pwm > 0: left_pwm = max(b_speed, left_pwm)
                    if right_pwm > 0: right_pwm = max(b_speed, right_pwm)
                elif "BACK" in action or "REVERSE" in action:
                    if left_pwm < 0: left_pwm = min(-b_speed, left_pwm)
                    if right_pwm < 0: right_pwm = min(-b_speed, right_pwm)

                if "TURN" in action or "PIVOT" in action or "NUDGE" in action:
                    if left_pwm > 0 and left_pwm < t_speed:
                        left_pwm = t_speed
                    elif left_pwm < 0 and left_pwm > -t_speed:
                        left_pwm = -t_speed
                    if right_pwm > 0 and right_pwm < t_speed:
                        right_pwm = t_speed
                    elif right_pwm < 0 and right_pwm > -t_speed:
                        right_pwm = -t_speed

                if (left_pwm != 0 or right_pwm != 0):
                    self.vlm_planner.record_feedback(
                        action_taken=action,
                        pwm=max(abs(left_pwm), abs(right_pwm)),
                        duration_ms=int(pulse_dur * 1000),
                        prev_dx=ex,
                        new_dx=ex,
                        target_name=target_type,
                        area_ratio=ground_ratio
                    )
            else:
                # Fallback if no vlm_planner configured
                if abs(ex) > deadband_x:
                    pulse_dur = max(0.08, min(0.18, 0.08 + (abs(ex) / float(w // 2)) * 0.10))
                    if ex > 0:
                        action = "TURN_RIGHT"
                        if dist_action == "APPROACH":
                            steer_bias = int(min(25, (ex / float(w // 2)) * 25))
                            left_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            right_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = turn_pwm, -turn_pwm
                    else:
                        action = "TURN_LEFT"
                        if dist_action == "APPROACH":
                            steer_bias = int(min(25, (abs(ex) / float(w // 2)) * 25))
                            right_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            left_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = -turn_pwm, turn_pwm
                elif dist_action == "APPROACH":
                    action = "APPROACH"
                    pulse_dur = max(0.10, min(0.20, (dist_cm - target_dist) / 80.0 * 0.15))
                    left_pwm, right_pwm = f_speed, f_speed
                    continuous_drive = True
                elif dist_action == "BACK_UP":
                    action = "BACK_UP"
                    pulse_dur = 0.10
                    left_pwm, right_pwm = -f_speed, -f_speed
                else:
                    action = "ALIGNED_HOLD"
                    left_pwm, right_pwm = 0, 0

            # Arm elevation tracking (smooth hardware interpolation)
            dt_lat = max(0.04, time.time() - now)
            self._track_arm_elevation(target_cy, h, latency_s=dt_lat)

            is_motion = self.config.get("motion_enabled", True)
            if not is_motion:
                if abs(ex) > deadband_x:
                    action = f"{action} [STATIONARY TURN]"
                else:
                    action = f"{action} [STATIONARY TRACK]"

            # Telemetry update
            self.telemetry["target_found"] = True
            self.telemetry["target_box"] = [int(ymin), int(xmin), int(ymax), int(xmax)]
            self.telemetry["leg_box"] = [int(v) for v in leg_box] if leg_box else []
            self.telemetry["target_type"] = target_type
            self.telemetry["error_x"] = int(ex)
            self.telemetry["ground_y"] = round(ground_ratio * 100, 1)
            self.telemetry["target_size"] = round(dist_cm, 1)
            self.telemetry["status"] = "FOLLOWING"
            self.telemetry["action"] = action
            dyn_label = "Moving" if is_moving else "Stationary"
            self.telemetry["details"] = f"{target_type.replace('_', ' ').title()} ({dyn_label}) @ {dist_cm:.0f}cm | Action: {action}"

            # Execute Sense-Think-Act Motion (Continuous smooth cruise along planned path)
            self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, is_target_moving=is_moving, continuous_drive=continuous_drive)
        else:
            self.telemetry["target_found"] = False
            self.telemetry["target_box"] = []
            self.telemetry["leg_box"] = []
            self.telemetry["target_polygon"] = []
            self.telemetry["leg_polygon"] = []
            self.telemetry["motion_plan"] = None
            self.telemetry["target_type"] = "NONE"
            self._target_dyn_history.clear()
            self._target_is_moving = False
            lost_duration = now - last_seen
            if lost_duration > 1.0:
                self._pf_ema_x = None
                self._pf_ema_ymax = None
                self._pf_last_target_center = None
                self.telemetry["status"] = "SEARCHING"
                self.telemetry["action"] = "STOPPED"
                self.telemetry["details"] = "Searching for legs or person in view..."
                if self.comm:
                    self.comm.send_stop()
                self.set_arm_up()

        return last_seen

    def _step_color_tracking(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Processes frame for high-speed HSV color blob segmentation and tracking."""
        target_color = str(self.config.get("target_color", "Red")).strip().capitalize()
        self.telemetry["target_color"] = target_color
        ranges = get_hsv_ranges(target_color)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_blobs = []
        min_detect_area = 200  # Lowered to detect smaller/distant colored targets
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a >= min_detect_area:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cand_cx = int(m["m10"] / m["m00"])
                    cand_cy = int(m["m01"] / m["m00"])
                else:
                    bx, by, bbw, bbh = cv2.boundingRect(cnt)
                    cand_cx, cand_cy = bx + bbw // 2, by + bbh // 2
                candidate_blobs.append((a, cnt, cand_cx, cand_cy))

        best_cnt = None
        max_area = 0
        if candidate_blobs:
            # Spatial Lock-In: Lock onto first seen target and keep focus on it!
            if self._locked_target_center is not None and self._locked_target_lost_frames < 25:
                lx, ly = self._locked_target_center
                gate_radius = max(240.0, float(w) * 0.55)
                candidate_blobs.sort(key=lambda b: math.hypot(b[2] - lx, b[3] - ly))
                closest = candidate_blobs[0]
                dist_to_lock = math.hypot(closest[2] - lx, closest[3] - ly)
                if dist_to_lock <= gate_radius or len(candidate_blobs) == 1:
                    max_area = closest[0]
                    best_cnt = closest[1]
                    self._locked_target_center = (closest[2], closest[3])
                    self._locked_target_lost_frames = 0
                    self._locked_target_frames += 1
                else:
                    self._locked_target_lost_frames += 1
            else:
                # Lock onto the first detected target (largest area)
                candidate_blobs.sort(key=lambda b: b[0], reverse=True)
                chosen = candidate_blobs[0]
                max_area = chosen[0]
                best_cnt = chosen[1]
                self._locked_target_center = (chosen[2], chosen[3])
                self._locked_target_label = target_color
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 1
        elif self._locked_target_center is not None:
            self._locked_target_lost_frames += 1
            if self._locked_target_lost_frames > 25:
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_frames = 0

        now = time.time()
        if best_cnt is not None:
            last_seen = now
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            m = cv2.moments(best_cnt)
            if m["m00"] > 0:
                raw_cx = int(m["m10"] / m["m00"])
                raw_cy = int(m["m01"] / m["m00"])
            else:
                raw_cx, raw_cy = x + bw // 2, y + bh // 2

            # Store segmentation polygon contour in telemetry for live HUD/stream display
            approx = cv2.approxPolyDP(best_cnt, epsilon=2.0, closed=True)
            self.telemetry["target_polygon"] = [[int(p[0][0]), int(p[0][1])] for p in approx]

            # Temporal EMA smoothing to eliminate single-pixel noise wobble
            if self._ct_ema_x is None:
                self._ct_ema_x = float(raw_cx)
                self._ct_ema_y = float(raw_cy)
            else:
                self._ct_ema_x = 0.35 * float(raw_cx) + 0.65 * self._ct_ema_x
                self._ct_ema_y = 0.35 * float(raw_cy) + 0.65 * self._ct_ema_y

            cx = int(self._ct_ema_x)
            cy = int(self._ct_ema_y)

            sx, sy = self._get_sweet_spot(w, h)
            ex = cx - sx
            ey = cy - sy

            total_area = float(w * h)
            area_ratio = max_area / total_area

            # Calibrated Physical Distance in Centimeters
            dist_cm = self.estimate_ground_distance(y + bh, h)
            self.telemetry["distance_cm"] = dist_cm

            target_area_ratio = float(self.config.get("color_target_area_ratio", 0.08))
            deadband_x = int(self.config.get("deadband_x", 25))

            self.telemetry["target_found"] = True
            self.telemetry["target_locked"] = (self._locked_target_center is not None)
            self.telemetry["target_box"] = [int(y), int(x), int(y + bh), int(x + bw)]
            self.telemetry["error_x"] = int(ex)
            self.telemetry["error_y"] = int(ey)
            self.telemetry["target_size"] = round(dist_cm, 1)

            t_speed = self._get_turn_speed()
            b_speed = self._get_base_speed()
            f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
            turn_pwm = self._compute_diminishing_turn_pwm(ex, w)
            target_dist = float(self.config.get("color_target_distance_cm", self.config.get("target_distance_cm", self.config.get("target_dist", 13.5))))

            # Evaluate target dynamics (Moving vs Stationary Target)
            is_moving = self._evaluate_target_dynamics(cx, cy, now)

            # 1. MAP OUT ALL MOVEMENT & KINEMATICS FIRST
            motion_plan = self._plan_linkage_trajectory(
                target_name=target_color,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                dist_cm=dist_cm,
                target_dist_cm=target_dist,
                is_moving=is_moving,
                target_poly=self.telemetry.get("target_polygon")
            )
            self.telemetry["motion_plan"] = motion_plan
            active_stage = motion_plan["active_stage"]
            left_pwm = active_stage["left_pwm"]
            right_pwm = active_stage["right_pwm"]
            action = active_stage["action"]
            continuous_drive = active_stage.get("continuous", False)
            pulse_dur = active_stage.get("duration_s", 0.12)

            if self.vlm_planner:
                sweet_spot = {"center_x": sx, "center_y": sy, "box_width": int(w * 0.35), "box_height": int(h * 0.35)}
                mock_det = DetectionResult(
                    detected=True,
                    category=f"{target_color}_object",
                    confidence=0.92,
                    bounding_box=(y, x, y + bh, x + bw),
                    material_color=target_color,
                    mask_polygon=self.telemetry.get("target_polygon")
                )
                vlm_plan = self.vlm_planner.plan_movement(
                    mock_det,
                    sweet_spot,
                    use_vlm_llm=False,
                    dist_cm=dist_cm,
                    target_dist_cm=float(target_dist),
                    target_is_moving=is_moving
                )
                self.telemetry["vlm_plan"] = vlm_plan.to_dict()
                if vlm_plan.continuous_drive:
                    left_pwm = vlm_plan.left_pwm
                    right_pwm = vlm_plan.right_pwm
                    action = vlm_plan.action
                    continuous_drive = True
                    pulse_dur = max(0.08, min(0.20, vlm_plan.duration_ms / 1000.0))

                # Hard turn and base speed floor guarantees from robot config
                if "FORWARD" in action or "APPROACH" in action or "CRUISE" in action:
                    if left_pwm > 0: left_pwm = max(b_speed, left_pwm)
                    if right_pwm > 0: right_pwm = max(b_speed, right_pwm)
                elif "BACK" in action or "REVERSE" in action:
                    if left_pwm < 0: left_pwm = min(-b_speed, left_pwm)
                    if right_pwm < 0: right_pwm = min(-b_speed, right_pwm)

                if "TURN" in action or "PIVOT" in action or "NUDGE" in action:
                    if left_pwm > 0 and left_pwm < t_speed:
                        left_pwm = t_speed
                    elif left_pwm < 0 and left_pwm > -t_speed:
                        left_pwm = -t_speed
                    if right_pwm > 0 and right_pwm < t_speed:
                        right_pwm = t_speed
                    elif right_pwm < 0 and right_pwm > -t_speed:
                        right_pwm = -t_speed

                if (left_pwm != 0 or right_pwm != 0):
                    self.vlm_planner.record_feedback(
                        action_taken=action,
                        pwm=max(abs(left_pwm), abs(right_pwm)),
                        duration_ms=int(pulse_dur * 1000),
                        prev_dx=ex,
                        new_dx=ex,
                        target_name=target_color,
                        area_ratio=area_ratio
                    )
            else:
                # Fallback if no vlm_planner configured: drive until true grasp reach
                needs_approach = (dist_cm > (target_dist + 1.5)) and (int(y + bh) < 445)
                if abs(ex) > deadband_x:
                    pulse_dur = max(0.08, min(0.18, 0.08 + (abs(ex) / float(w // 2)) * 0.10))
                    if ex > 0:
                        action = "TURN_RIGHT"
                        if needs_approach:
                            steer_bias = int(min(25, (ex / float(w // 2)) * 25))
                            left_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            right_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = turn_pwm, -turn_pwm
                    else:
                        action = "TURN_LEFT"
                        if needs_approach:
                            steer_bias = int(min(25, (abs(ex) / float(w // 2)) * 25))
                            right_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            left_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = -turn_pwm, turn_pwm
                elif needs_approach:
                    action = "APPROACH"
                    pulse_dur = max(0.10, min(0.20, (dist_cm - target_dist) / 80.0 * 0.25))
                    left_pwm, right_pwm = f_speed, f_speed
                    continuous_drive = True
                else:
                    action = "ALIGNED_AT_TARGET"
                    left_pwm, right_pwm = 0, 0

            # Arm elevation tracking (smooth hardware interpolation)
            dt_lat = max(0.04, time.time() - now)
            self._track_arm_elevation(cy, h, latency_s=dt_lat, ymax=int(y + bh))

            # Settle & Autonomous Grab Trigger:
            # If target is centered horizontally and vertically, within grasp range, and steady:
            is_x_centered = abs(ex) <= deadband_x
            is_y_centered = self.telemetry.get("arm_ik", {}).get("vertical_status") in ("CENTERED", "SETTLED_AT_LIMIT")
            is_near = (dist_cm <= target_dist + 1.5) or (int(y + bh) >= 445)

            vplan = self.telemetry.get("vlm_plan") or {}
            vlm_action = vplan.get("action", "")
            is_vlm_aligned = (vlm_action == "ALIGNED_GRASP")
            vlm_clamp_angle = (vplan.get("arm_plan") or {}).get("calibrated_angles", {}).get("grip_target")

            if (is_x_centered and is_y_centered and is_near and not is_moving) or (is_vlm_aligned and is_near):
                self._ik_settled_frames += (2 if is_vlm_aligned else 1)
                if self._ik_settled_frames >= 4 and not self._is_grabbing:
                    self._trigger_autonomous_grab(target_color, clamp_angle=vlm_clamp_angle)
            else:
                self._ik_settled_frames = max(0, self._ik_settled_frames - 1)

            if self._is_grabbing:
                return last_seen

            is_motion = self.config.get("motion_enabled", True)
            if not is_motion:
                if abs(ex) > deadband_x:
                    action = f"{action} [STATIONARY TURN]"
                else:
                    action = f"{action} [STATIONARY TRACK]"

            self.telemetry["status"] = "TRACKING"
            self.telemetry["action"] = action
            dyn_label = "Moving" if is_moving else "Stationary"
            self.telemetry["details"] = f"Locked on {target_color} ({dyn_label}) @ {dist_cm:.0f}cm | Action: {action}"

            # Execute Sense-Think-Act Motion (Continuous smooth cruise along planned path)
            self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, is_target_moving=is_moving, continuous_drive=continuous_drive)
        else:
            self.telemetry["target_found"] = False
            self.telemetry["target_box"] = []
            self.telemetry["target_polygon"] = []
            self.telemetry["motion_plan"] = None
            self._target_dyn_history.clear()
            self._target_is_moving = False
            lost_duration = now - last_seen
            if lost_duration > 1.2:
                self._ct_ema_x = None
                self._ct_ema_y = None
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 0
                self.telemetry["target_locked"] = False
                self.telemetry["status"] = "SEARCHING"
                self.telemetry["action"] = "STOPPED"
                self.telemetry["details"] = f"Searching for {target_color} target in view..."
                if self.comm:
                    self.comm.send_stop()
                self.set_arm_down(open_gripper=True)

        return last_seen

    def _step_object_tracking(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Processes frame for generic object pursuit across all COCO & YOLOE classes."""
        target_obj = self.config.get("target_object", "bottle").strip().lower()
        best_box = None
        best_conf = 0.0

        # Synonym dictionary for fuzzy class matching
        SYNONYM_MAP = {
            "bottle": ["bottle", "water bottle", "plastic bottle", "wine bottle", "beer bottle", "can", "cup", "flask", "jar", "container", "spray bottle", "spray_bottle", "traffic cone", "cone", "dispenser"],
            "spray bottle": ["spray bottle", "spray_bottle", "spray", "cleaning spray", "bottle", "traffic cone", "cone", "dispenser", "container"],
            "spray_bottle": ["spray bottle", "spray_bottle", "spray", "cleaning spray", "bottle", "traffic cone", "cone", "dispenser", "container"],
            "cup": ["cup", "mug", "coffee cup", "glass", "bottle", "can"],
            "phone": ["cell phone", "phone", "mobile phone", "telephone", "remote", "control"],
            "person": ["person", "human", "man", "woman", "boy", "girl", "leg", "legs", "pants"],
            "box": ["box", "carton", "package", "book", "container", "suitcase"],
            "toy": ["teddy bear", "toy", "sports ball", "frisbee"],
        }
        search_terms = SYNONYM_MAP.get(target_obj, [target_obj])

        detections = []
        with self._vision_lock:
            if self._latest_detections and (now - self._latest_detection_time) < 1.2:
                detections = list(self._latest_detections)
        if not detections and self.vision:
            try:
                detections = self.vision.detect_all(frame)
            except Exception as e:
                print(f"[Activities] Vision detection error in object tracking: {e}")

        matching_candidates = []
        for d in detections:
            cat = d.category.lower().strip()
            is_match = (target_obj in ("any", "all", "*", "auto", "")) or any(st in cat or cat in st for st in search_terms)
            if is_match and d.confidence >= 0.18:
                bb = d.bounding_box
                if bb and len(bb) == 4:
                    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    score = area + (d.confidence * 1000.0)
                    cand_cx = (bb[1] + bb[3]) // 2
                    cand_cy = (bb[0] + bb[2]) // 2
                    poly = getattr(d, "mask_polygon", None)
                    matching_candidates.append({
                        "box": (bb[0], bb[1], bb[2], bb[3], d.confidence, d.category),
                        "poly": poly,
                        "cx": cand_cx,
                        "cy": cand_cy,
                        "area": area,
                        "score": score,
                        "cat": d.category
                    })

        best_box = None
        best_poly = None
        if matching_candidates:
            # Spatial Lock-In: Lock onto first seen target and keep focus on it!
            if self._locked_target_center is not None and self._locked_target_lost_frames < 25:
                lx, ly = self._locked_target_center
                gate_radius = max(240.0, float(w) * 0.55)
                matching_candidates.sort(key=lambda c: math.hypot(c["cx"] - lx, c["cy"] - ly))
                closest = matching_candidates[0]
                dist_to_lock = math.hypot(closest["cx"] - lx, closest["cy"] - ly)
                if dist_to_lock <= gate_radius or len(matching_candidates) == 1:
                    best_box = closest["box"]
                    best_poly = closest["poly"]
                    self._locked_target_center = (closest["cx"], closest["cy"])
                    self._locked_target_lost_frames = 0
                    self._locked_target_frames += 1
                else:
                    self._locked_target_lost_frames += 1
            else:
                # Lock onto the first detected target (highest score)
                matching_candidates.sort(key=lambda c: c["score"], reverse=True)
                chosen = matching_candidates[0]
                best_box = chosen["box"]
                best_poly = chosen["poly"]
                self._locked_target_center = (chosen["cx"], chosen["cy"])
                self._locked_target_label = chosen["cat"]
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 1
        elif self._locked_target_center is not None:
            self._locked_target_lost_frames += 1
            if self._locked_target_lost_frames > 25:
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_frames = 0

        now = time.time()
        if best_box is not None:
            last_seen = now
            ymin, xmin, ymax, xmax, conf, cat_name = best_box
            raw_cx = (xmin + xmax) // 2
            px_height = ymax - ymin
            height_ratio = px_height / float(h)

            # Store segmentation polygon contour in telemetry for live HUD/stream display
            if best_poly and len(best_poly) >= 3:
                self.telemetry["target_polygon"] = [[int(pt[0]), int(pt[1])] for pt in best_poly]
            else:
                self.telemetry["target_polygon"] = [
                    [int(xmin), int(ymin)], [int(xmax), int(ymin)],
                    [int(xmax), int(ymax)], [int(xmin), int(ymax)]
                ]

            # Temporal EMA smoothing for tracking center
            if self._ot_ema_x is None:
                self._ot_ema_x = float(raw_cx)
            else:
                self._ot_ema_x = 0.35 * float(raw_cx) + 0.65 * self._ot_ema_x

            px_center = int(self._ot_ema_x)
            sx, sy = self._get_sweet_spot(w, h)
            ex = px_center - sx
            target_h = float(self.config.get("object_target_height_ratio", 0.35))
            deadband_x = int(self.config.get("deadband_x", 25))
            h_tol = 0.07

            # Calibrated Physical Distance in Centimeters
            dist_cm = self.estimate_ground_distance(ymax, h)
            self.telemetry["distance_cm"] = dist_cm

            target_cy = (ymin + ymax) // 2
            ey = target_cy - sy

            self.telemetry["target_found"] = True
            self.telemetry["target_locked"] = (self._locked_target_center is not None)
            self.telemetry["target_box"] = [int(ymin), int(xmin), int(ymax), int(xmax)]
            self.telemetry["error_x"] = int(ex)
            self.telemetry["error_y"] = int(ey)
            self.telemetry["target_size"] = round(dist_cm, 1)

            t_speed = self._get_turn_speed()
            b_speed = self._get_base_speed()
            f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
            turn_pwm = self._compute_diminishing_turn_pwm(ex, w)
            target_dist = float(self.config.get("object_target_distance_cm", self.config.get("target_distance_cm", self.config.get("target_dist", 13.5))))

            # Evaluate target dynamics (Moving vs Stationary Target)
            is_moving = self._evaluate_target_dynamics(px_center, target_cy, now)

            # 1. MAP OUT ALL MOVEMENT & KINEMATICS FIRST
            motion_plan = self._plan_linkage_trajectory(
                target_name=cat_name,
                cx=px_center,
                cy=target_cy,
                w=w,
                h=h,
                dist_cm=dist_cm,
                target_dist_cm=target_dist,
                is_moving=is_moving,
                target_poly=self.telemetry.get("target_polygon")
            )
            self.telemetry["motion_plan"] = motion_plan
            active_stage = motion_plan["active_stage"]
            left_pwm = active_stage["left_pwm"]
            right_pwm = active_stage["right_pwm"]
            action = active_stage["action"]
            continuous_drive = active_stage.get("continuous", False)
            pulse_dur = active_stage.get("duration_s", 0.12)

            if self.vlm_planner:
                sweet_spot = {"center_x": sx, "center_y": sy, "box_width": int(w * 0.35), "box_height": int(h * 0.35)}
                mock_det = DetectionResult(
                    detected=True,
                    category=cat_name,
                    confidence=conf,
                    bounding_box=(ymin, xmin, ymax, xmax),
                    material_color="Dynamic",
                    mask_polygon=self.telemetry.get("target_polygon")
                )
                vlm_plan = self.vlm_planner.plan_movement(
                    mock_det,
                    sweet_spot,
                    use_vlm_llm=False,
                    dist_cm=dist_cm,
                    target_dist_cm=float(target_dist),
                    target_is_moving=is_moving
                )
                self.telemetry["vlm_plan"] = vlm_plan.to_dict()
                if vlm_plan.continuous_drive:
                    left_pwm = vlm_plan.left_pwm
                    right_pwm = vlm_plan.right_pwm
                    action = vlm_plan.action
                    continuous_drive = True
                    pulse_dur = max(0.08, min(0.20, vlm_plan.duration_ms / 1000.0))

                # Hard turn and base speed floor guarantees from robot config
                if "FORWARD" in action or "APPROACH" in action or "CRUISE" in action:
                    if left_pwm > 0: left_pwm = max(b_speed, left_pwm)
                    if right_pwm > 0: right_pwm = max(b_speed, right_pwm)
                elif "BACK" in action or "REVERSE" in action:
                    if left_pwm < 0: left_pwm = min(-b_speed, left_pwm)
                    if right_pwm < 0: right_pwm = min(-b_speed, right_pwm)

                if "TURN" in action or "PIVOT" in action or "NUDGE" in action:
                    if left_pwm > 0 and left_pwm < t_speed:
                        left_pwm = t_speed
                    elif left_pwm < 0 and left_pwm > -t_speed:
                        left_pwm = -t_speed
                    if right_pwm > 0 and right_pwm < t_speed:
                        right_pwm = t_speed
                    elif right_pwm < 0 and right_pwm > -t_speed:
                        right_pwm = -t_speed

                if (left_pwm != 0 or right_pwm != 0):
                    self.vlm_planner.record_feedback(
                        action_taken=action,
                        pwm=max(abs(left_pwm), abs(right_pwm)),
                        duration_ms=int(pulse_dur * 1000),
                        prev_dx=ex,
                        new_dx=ex,
                        target_name=cat_name,
                        area_ratio=height_ratio
                    )
            else:
                # Fallback if no vlm_planner configured: drive until true grasp reach
                needs_approach = (dist_cm > (target_dist + 1.5)) and (ymax < 445)
                if abs(ex) > deadband_x:
                    pulse_dur = max(0.08, min(0.18, 0.08 + (abs(ex) / float(w // 2)) * 0.10))
                    if ex > 0:
                        action = "TURN_RIGHT"
                        if needs_approach:
                            steer_bias = int(min(25, (ex / float(w // 2)) * 25))
                            left_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            right_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = turn_pwm, -turn_pwm
                    else:
                        action = "TURN_LEFT"
                        if needs_approach:
                            steer_bias = int(min(25, (abs(ex) / float(w // 2)) * 25))
                            right_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            left_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = -turn_pwm, turn_pwm
                elif needs_approach:
                    action = "APPROACH"
                    pulse_dur = max(0.10, min(0.20, (dist_cm - target_dist) / 80.0 * 0.25))
                    left_pwm, right_pwm = f_speed, f_speed
                    continuous_drive = True
                elif dist_cm < (target_dist - 3.0) and ymax > 465:
                    action = "BACK_UP"
                    pulse_dur = 0.10
                    left_pwm, right_pwm = -f_speed, -f_speed
                else:
                    action = "ALIGNED_HOLD"
                    left_pwm, right_pwm = 0, 0

            # Arm elevation tracking (smooth hardware interpolation)
            dt_lat = max(0.04, time.time() - now)
            self._track_arm_elevation(target_cy, h, latency_s=dt_lat, ymax=ymax)

            # Settle & Autonomous Grab Trigger:
            # If target object is centered horizontally and vertically, within grasp range, and steady:
            is_x_centered = abs(ex) <= deadband_x
            is_y_centered = self.telemetry.get("arm_ik", {}).get("vertical_status") in ("CENTERED", "SETTLED_AT_LIMIT")
            is_near = (dist_cm <= target_dist + 1.5) or (ymax >= 445)

            vplan = self.telemetry.get("vlm_plan") or {}
            vlm_action = vplan.get("action", "")
            is_vlm_aligned = (vlm_action == "ALIGNED_GRASP")
            vlm_clamp_angle = (vplan.get("arm_plan") or {}).get("calibrated_angles", {}).get("grip_target")

            if (is_x_centered and is_y_centered and is_near and not is_moving) or (is_vlm_aligned and is_near):
                self._ik_settled_frames += (2 if is_vlm_aligned else 1)
                if self._ik_settled_frames >= 4 and not self._is_grabbing:
                    self._trigger_autonomous_grab(cat_name, clamp_angle=vlm_clamp_angle)
            else:
                self._ik_settled_frames = max(0, self._ik_settled_frames - 1)

            if self._is_grabbing:
                return last_seen

            is_motion = self.config.get("motion_enabled", True)
            if not is_motion:
                if abs(ex) > deadband_x:
                    action = f"{action} [STATIONARY TURN]"
                else:
                    action = f"{action} [STATIONARY TRACK]"

            self.telemetry["status"] = "TRACKING OBJECT"
            self.telemetry["action"] = action
            dyn_label = "Moving" if is_moving else "Stationary"
            self.telemetry["details"] = f"Tracking '{cat_name}' ({conf*100:.0f}%, {dyn_label}) @ {dist_cm:.0f}cm | Action: {action}"

            # Execute Sense-Think-Act Motion (Continuous smooth cruise along planned path)
            self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, is_target_moving=is_moving, continuous_drive=continuous_drive)
        else:
            self.telemetry["target_found"] = False
            self.telemetry["target_box"] = []
            self.telemetry["target_polygon"] = []
            self.telemetry["motion_plan"] = None
            self._target_dyn_history.clear()
            self._target_is_moving = False
            lost_duration = now - last_seen
            if lost_duration > 1.2:
                self._ot_ema_x = None
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 0
                self.telemetry["target_locked"] = False
                self.telemetry["status"] = "SEARCHING"
                self.telemetry["action"] = "STOPPED"
                self.telemetry["details"] = f"Searching for '{target_obj}' in view..."
                if self.comm:
                    self.comm.send_stop()
                self.set_arm_down(open_gripper=True)

        return last_seen

    def _step_color_track_and_classify(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Pursues target color blob while LiteRT neural net identifies what the object is."""
        target_color = str(self.config.get("target_color", "Red")).strip().capitalize()
        self.telemetry["target_color"] = target_color
        ranges = get_hsv_ranges(target_color)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_blobs = []
        min_detect_area = 200  # Lowered to detect smaller colored objects
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a >= min_detect_area:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cand_cx = int(m["m10"] / m["m00"])
                    cand_cy = int(m["m01"] / m["m00"])
                else:
                    bx, by, bbw, bbh = cv2.boundingRect(cnt)
                    cand_cx, cand_cy = bx + bbw // 2, by + bbh // 2
                candidate_blobs.append((a, cnt, cand_cx, cand_cy))

        best_cnt = None
        max_area = 0
        if candidate_blobs:
            # Spatial Lock-In: Lock onto first seen target and keep focus on it!
            if self._locked_target_center is not None and self._locked_target_lost_frames < 25:
                lx, ly = self._locked_target_center
                gate_radius = max(240.0, float(w) * 0.55)
                candidate_blobs.sort(key=lambda b: math.hypot(b[2] - lx, b[3] - ly))
                closest = candidate_blobs[0]
                dist_to_lock = math.hypot(closest[2] - lx, closest[3] - ly)
                if dist_to_lock <= gate_radius or len(candidate_blobs) == 1:
                    max_area = closest[0]
                    best_cnt = closest[1]
                    self._locked_target_center = (closest[2], closest[3])
                    self._locked_target_lost_frames = 0
                    self._locked_target_frames += 1
                else:
                    self._locked_target_lost_frames += 1
            else:
                # Lock onto the first detected target (largest area)
                candidate_blobs.sort(key=lambda b: b[0], reverse=True)
                chosen = candidate_blobs[0]
                max_area = chosen[0]
                best_cnt = chosen[1]
                self._locked_target_center = (chosen[2], chosen[3])
                self._locked_target_label = target_color
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 1
        elif self._locked_target_center is not None:
            self._locked_target_lost_frames += 1
            if self._locked_target_lost_frames > 25:
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_frames = 0

        now = time.time()
        if best_cnt is not None:
            last_seen = now
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            # Store segmentation polygon contour in telemetry for live HUD/stream display
            approx = cv2.approxPolyDP(best_cnt, epsilon=2.0, closed=True)
            self.telemetry["target_polygon"] = [[int(p[0][0]), int(p[0][1])] for p in approx]
            m = cv2.moments(best_cnt)
            if m["m00"] > 0:
                raw_cx = int(m["m10"] / m["m00"])
                raw_cy = int(m["m01"] / m["m00"])
            else:
                raw_cx, raw_cy = x + bw // 2, y + bh // 2

            if self._ct_ema_x is None:
                self._ct_ema_x = float(raw_cx)
                self._ct_ema_y = float(raw_cy)
            else:
                self._ct_ema_x = 0.35 * float(raw_cx) + 0.65 * self._ct_ema_x
                self._ct_ema_y = 0.35 * float(raw_cy) + 0.65 * self._ct_ema_y

            cx = int(self._ct_ema_x)
            cy = int(self._ct_ema_y)

            sx, sy = self._get_sweet_spot(w, h)
            ex = cx - sx
            ey = cy - sy

            total_area = float(w * h)
            area_ratio = max_area / total_area

            # Neural object classification on the detected color blob
            classified_label = "Object"
            class_conf = 0.0

            if self.vision:
                try:
                    detections = self.vision.detect_all(frame)
                    best_overlap_area = 0
                    for d in detections:
                        bb = d.bounding_box
                        if bb and len(bb) == 4:
                            ymin, xmin, ymax, xmax = bb
                            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                                if d.confidence > class_conf:
                                    class_conf = d.confidence
                                    classified_label = d.category
                            else:
                                ix1 = max(x, xmin)
                                iy1 = max(y, ymin)
                                ix2 = min(x + bw, xmax)
                                iy2 = min(y + bh, ymax)
                                if ix2 > ix1 and iy2 > iy1:
                                    overlap = (ix2 - ix1) * (iy2 - iy1)
                                    if overlap > best_overlap_area and d.confidence > 0.18:
                                        best_overlap_area = overlap
                                        class_conf = d.confidence
                                        classified_label = d.category
                except Exception:
                    pass

            self.telemetry["classified_object"] = classified_label
            full_title = f"{target_color} {classified_label}"
            target_area_ratio = float(self.config.get("color_target_area_ratio", 0.08))
            deadband_x = int(self.config.get("deadband_x", 25))

            # Calibrated Physical Distance in Centimeters
            dist_cm = self.estimate_ground_distance(y + bh, h)
            self.telemetry["distance_cm"] = dist_cm

            self.telemetry["target_found"] = True
            self.telemetry["target_locked"] = (self._locked_target_center is not None)
            self.telemetry["target_box"] = [int(y), int(x), int(y + bh), int(x + bw)]
            self.telemetry["error_x"] = int(ex)
            self.telemetry["error_y"] = int(ey)
            self.telemetry["target_size"] = round(dist_cm, 1)

            t_speed = self._get_turn_speed()
            b_speed = self._get_base_speed()
            f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
            turn_pwm = self._compute_diminishing_turn_pwm(ex, w)
            target_dist = float(self.config.get("color_target_distance_cm", self.config.get("target_distance_cm", self.config.get("target_dist", 13.5))))

            # Evaluate target dynamics (Moving vs Stationary Target)
            is_moving = self._evaluate_target_dynamics(cx, cy, now)

            # 1. MAP OUT ALL MOVEMENT & KINEMATICS FIRST
            motion_plan = self._plan_linkage_trajectory(
                target_name=full_title,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                dist_cm=dist_cm,
                target_dist_cm=target_dist,
                is_moving=is_moving,
                target_poly=self.telemetry.get("target_polygon")
            )
            self.telemetry["motion_plan"] = motion_plan
            active_stage = motion_plan["active_stage"]
            left_pwm = active_stage["left_pwm"]
            right_pwm = active_stage["right_pwm"]
            action = active_stage["action"]
            continuous_drive = active_stage.get("continuous", False)
            pulse_dur = active_stage.get("duration_s", 0.12)

            if self.vlm_planner:
                sweet_spot = {"center_x": sx, "center_y": sy, "box_width": int(w * 0.35), "box_height": int(h * 0.35)}
                mock_det = DetectionResult(
                    detected=True,
                    category=classified_label.lower(),
                    confidence=max(0.70, class_conf),
                    bounding_box=(y, x, y + bh, x + bw),
                    material_color=target_color,
                    mask_polygon=self.telemetry.get("target_polygon")
                )
                vlm_plan = self.vlm_planner.plan_movement(
                    mock_det,
                    sweet_spot,
                    use_vlm_llm=False,
                    dist_cm=dist_cm,
                    target_dist_cm=float(target_dist),
                    target_is_moving=is_moving
                )
                self.telemetry["vlm_plan"] = vlm_plan.to_dict()
                if vlm_plan.continuous_drive:
                    left_pwm = vlm_plan.left_pwm
                    right_pwm = vlm_plan.right_pwm
                    action = vlm_plan.action
                    continuous_drive = True
                    pulse_dur = max(0.08, min(0.20, vlm_plan.duration_ms / 1000.0))

                # Hard turn and base speed floor guarantees from robot config
                if "FORWARD" in action or "APPROACH" in action or "CRUISE" in action:
                    if left_pwm > 0: left_pwm = max(b_speed, left_pwm)
                    if right_pwm > 0: right_pwm = max(b_speed, right_pwm)
                elif "BACK" in action or "REVERSE" in action:
                    if left_pwm < 0: left_pwm = min(-b_speed, left_pwm)
                    if right_pwm < 0: right_pwm = min(-b_speed, right_pwm)

                if "TURN" in action or "PIVOT" in action or "NUDGE" in action:
                    if left_pwm > 0 and left_pwm < t_speed:
                        left_pwm = t_speed
                    elif left_pwm < 0 and left_pwm > -t_speed:
                        left_pwm = -t_speed
                    if right_pwm > 0 and right_pwm < t_speed:
                        right_pwm = t_speed
                    elif right_pwm < 0 and right_pwm > -t_speed:
                        right_pwm = -t_speed

                if (left_pwm != 0 or right_pwm != 0):
                    self.vlm_planner.record_feedback(
                        action_taken=action,
                        pwm=max(abs(left_pwm), abs(right_pwm)),
                        duration_ms=int(pulse_dur * 1000),
                        prev_dx=ex,
                        new_dx=ex,
                        target_name=full_title,
                        area_ratio=area_ratio
                    )
            else:
                # Fallback if no vlm_planner configured: drive until true grasp reach
                needs_approach = (dist_cm > (target_dist + 1.5)) and (int(y + bh) < 445)
                if abs(ex) > deadband_x:
                    pulse_dur = max(0.08, min(0.18, 0.08 + (abs(ex) / float(w // 2)) * 0.10))
                    if ex > 0:
                        action = "TURN_RIGHT"
                        if needs_approach:
                            steer_bias = int(min(25, (ex / float(w // 2)) * 25))
                            left_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            right_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = turn_pwm, -turn_pwm
                    else:
                        action = "TURN_LEFT"
                        if needs_approach:
                            steer_bias = int(min(25, (abs(ex) / float(w // 2)) * 25))
                            right_pwm = min(255, max(t_speed, f_speed + steer_bias))
                            left_pwm = max(b_speed, f_speed - steer_bias * 2)
                            continuous_drive = True
                        else:
                            left_pwm, right_pwm = -turn_pwm, turn_pwm
                elif needs_approach:
                    action = "APPROACH"
                    pulse_dur = max(0.10, min(0.20, (dist_cm - target_dist) / 80.0 * 0.25))
                    left_pwm, right_pwm = f_speed, f_speed
                    continuous_drive = True
                else:
                    action = "ALIGNED_AT_TARGET"
                    left_pwm, right_pwm = 0, 0

            # Arm elevation tracking (smooth hardware interpolation)
            dt_lat = max(0.04, time.time() - now)
            self._track_arm_elevation(cy, h, latency_s=dt_lat, ymax=int(y + bh))

            # Settle & Autonomous Grab Trigger:
            # If target is centered horizontally and vertically, within grasp range, and steady:
            is_x_centered = abs(ex) <= deadband_x
            is_y_centered = self.telemetry.get("arm_ik", {}).get("vertical_status") in ("CENTERED", "SETTLED_AT_LIMIT")
            is_near = (dist_cm <= target_dist + 1.5) or (int(y + bh) >= 445)

            vplan = self.telemetry.get("vlm_plan") or {}
            vlm_action = vplan.get("action", "")
            is_vlm_aligned = (vlm_action == "ALIGNED_GRASP")
            vlm_clamp_angle = (vplan.get("arm_plan") or {}).get("calibrated_angles", {}).get("grip_target")

            if (is_x_centered and is_y_centered and is_near and not is_moving) or (is_vlm_aligned and is_near):
                self._ik_settled_frames += (2 if is_vlm_aligned else 1)
                if self._ik_settled_frames >= 4 and not self._is_grabbing:
                    self._trigger_autonomous_grab(full_title, clamp_angle=vlm_clamp_angle)
            else:
                self._ik_settled_frames = max(0, self._ik_settled_frames - 1)

            if self._is_grabbing:
                return last_seen

            is_motion = self.config.get("motion_enabled", True)
            if not is_motion:
                if abs(ex) > deadband_x:
                    action = f"{action} [STATIONARY TURN]"
                else:
                    action = f"{action} [STATIONARY TRACK]"

            self.telemetry["status"] = "TRACKING & CLASSIFYING"
            self.telemetry["action"] = action
            conf_str = f" ({class_conf*100:.0f}%)" if class_conf > 0 else ""
            dyn_label = "Moving" if is_moving else "Stationary"
            self.telemetry["details"] = f"Identified: {full_title}{conf_str} ({dyn_label}) @ {dist_cm:.0f}cm | Action: {action}"

            # Execute Sense-Think-Act Motion (Continuous smooth cruise along planned path)
            self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, is_target_moving=is_moving, continuous_drive=continuous_drive)
        else:
            self.telemetry["target_found"] = False
            self.telemetry["target_box"] = []
            self.telemetry["target_polygon"] = []
            self.telemetry["motion_plan"] = None
            self._target_dyn_history.clear()
            self._target_is_moving = False
            lost_duration = now - last_seen
            if lost_duration > 1.2:
                self._ct_ema_x = None
                self._ct_ema_y = None
                self._locked_target_center = None
                self._locked_target_label = None
                self._locked_target_lost_frames = 0
                self._locked_target_frames = 0
                self.telemetry["target_locked"] = False
                self.telemetry["status"] = "SEARCHING"
                self.telemetry["action"] = "STOPPED"
                self.telemetry["details"] = f"Searching for {target_color} objects in view..."
                if self.comm:
                    self.comm.send_stop()
                self.set_arm_down(open_gripper=True)

        return last_seen

    def _step_object_sizing(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Processes frame for real-time object sizing, dimensional analysis, and gripper fit."""
        filter_target = self.config.get("sizing_target_object", "any").strip().lower()
        best_cand = None
        best_score = 0.0

        # Classes of large objects, scenery, architecture, vehicles, and furniture to exclude
        LARGE_AND_SCENERY_CLASSES = {
            "wall", "floor", "ground", "ceiling", "sky", "scenery", "background",
            "room", "window", "door", "building", "person", "man", "woman", "boy", "girl",
            "chair", "table", "desk", "bed", "couch", "sofa", "bench", "tree", "plant",
            "tv", "refrigerator", "oven", "microwave", "sink", "car", "truck", "bus", "train",
            "motorcycle", "bicycle", "cabinet", "wardrobe", "shelf", "dining table"
        }

        if self.vision:
            try:
                detections = self.vision.detect_all(frame)
                for d in detections:
                    cat = d.category.lower().strip()
                    if filter_target != "any" and filter_target not in cat:
                        continue
                    # Ignore large scenery, furniture, and vehicles
                    if filter_target == "any" and any(sc in cat for sc in LARGE_AND_SCENERY_CLASSES):
                        continue
                    if d.confidence < 0.20:
                        continue

                    bb = d.bounding_box
                    if bb and len(bb) == 4:
                        ymin, xmin, ymax, xmax = [int(v) for v in bb]
                        bw = xmax - xmin
                        bh = ymax - ymin
                        if bw <= 10 or bh <= 10:
                            continue
                        area = bw * bh
                        # STRICT SMALL-OBJECT FILTER:
                        # Avoid scanning for large objects! Reject any object spanning > 22% of frame area, > 38% width, or > 48% height
                        if area > (w * h * 0.22) or bw > (w * 0.38) or bh > (h * 0.48):
                            continue

                        # Prioritize objects centered in front of camera (sweet spot alignment)
                        cand_cx = (xmin + xmax) // 2
                        cand_cy = (ymin + ymax) // 2
                        dx_norm = abs(cand_cx - cx_img) / float(w // 2)
                        dy_norm = abs(cand_cy - int(h * 0.515)) / float(h // 2)
                        center_proximity = max(0.0, 1.0 - (dx_norm * 0.6 + dy_norm * 0.4))

                        # Strongly favor small, compact tabletop graspable objects (bonus for small footprint)
                        is_compact_small = (bw < int(w * 0.24) and bh < int(h * 0.35))
                        small_bonus = 35.0 if is_compact_small else 10.0
                        cand_score = (d.confidence * 35.0) + (center_proximity * 35.0) + small_bonus

                        if cand_score > best_score:
                            best_score = cand_score
                            poly = getattr(d, "mask_polygon", None)
                            best_cand = (ymin, xmin, ymax, xmax, d.confidence, d.category, poly)
            except Exception as e:
                print(f"[Activities] Vision error during object sizing: {e}")

        now = time.time()
        if best_cand is not None:
            last_seen = now
            ymin, xmin, ymax, xmax, conf, cat_name, poly = best_cand
            raw_w = float(xmax - xmin)
            raw_h = float(ymax - ymin)
            px_center = (xmin + xmax) // 2
            sx, sy = self._get_sweet_spot(w, h)
            ex = px_center - sx

            # Temporal exponential moving average (EMA) filter to prevent jitter
            if self._sizing_ema_w is None:
                self._sizing_ema_w = raw_w
                self._sizing_ema_h = raw_h
            else:
                self._sizing_ema_w = 0.70 * self._sizing_ema_w + 0.30 * raw_w
                self._sizing_ema_h = 0.70 * self._sizing_ema_h + 0.30 * raw_h

            smooth_w_px = int(round(self._sizing_ema_w))
            smooth_h_px = int(round(self._sizing_ema_h))

            height_ratio = float(smooth_h_px) / float(h)
            target_h = float(self.config.get("sizing_target_height_ratio", 0.35))
            deadband_x = int(self.config.get("deadband_x", 30))

            # Perspective depth correction:
            # Baseline MM_PER_PIXEL_SWEET_SPOT is calibrated at inspection distance (target_h ~ 0.35).
            # When object is further away (smaller height_ratio), apply inverse scale correction
            dist_factor = max(0.5, min(2.2, target_h / max(0.08, height_ratio)))
            width_mm = round(smooth_w_px * MM_PER_PIXEL_SWEET_SPOT * dist_factor, 1)
            height_mm = round(smooth_h_px * MM_PER_PIXEL_SWEET_SPOT * dist_factor, 1)
            area_cm2 = round((width_mm * height_mm) / 100.0, 1)
            aspect_ratio = round(smooth_w_px / max(1.0, float(smooth_h_px)), 2)

            # Categorical Size Classification
            if width_mm < 35 or height_mm < 35:
                size_cat = "TINY"
            elif width_mm <= 65 and height_mm <= 85:
                size_cat = "SMALL"
            elif width_mm <= 110 and height_mm <= 155:
                size_cat = "MEDIUM"
            elif width_mm <= 180 and height_mm <= 240:
                size_cat = "LARGE"
            else:
                size_cat = "OVERSIZED"

            # Gripper Fit Evaluation (Gripper max opening = 85mm)
            if 15.0 <= width_mm <= 85.0:
                gripper_fit = "GRASPABLE"
                gripper_msg = "Fits Gripper Jaws (<= 85mm)"
            elif width_mm > 85.0:
                gripper_fit = "TOO_LARGE"
                gripper_msg = "Exceeds Max Gripper Span (> 85mm)"
            else:
                gripper_fit = "TOO_SMALL"
                gripper_msg = "Too Small / Thin (< 15mm)"

            # Update real-time sizing telemetry
            # Calibrated Physical Distance in Centimeters
            dist_cm = self.estimate_ground_distance(ymax, h)
            self.telemetry["distance_cm"] = dist_cm
            self.telemetry["target_found"] = True
            self.telemetry["target_box"] = [int(ymin), int(xmin), int(ymax), int(xmax)]
            self.telemetry["target_polygon"] = [[int(pt[0]), int(pt[1])] for pt in poly] if poly else []
            self.telemetry["error_x"] = int(ex)
            self.telemetry["target_size"] = round(dist_cm, 1)
            self.telemetry["sizing"] = {
                "object": cat_name,
                "width_px": smooth_w_px,
                "height_px": smooth_h_px,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "distance_cm": dist_cm,
                "area_cm2": area_cm2,
                "aspect_ratio": aspect_ratio,
                "size_category": size_cat,
                "gripper_fit": gripper_fit,
                "gripper_msg": gripper_msg,
            }

            t_speed = self._get_turn_speed()
            b_speed = self._get_base_speed()
            f_speed = max(b_speed, int(self.config.get("follow_speed", 235)))
            turn_pwm = self._compute_diminishing_turn_pwm(ex, w)
            target_dist = float(self.config.get("sizing_target_distance_cm", 28.0))

            # Evaluate target dynamics (Moving vs Stationary Target)
            target_cy = (ymin + ymax) // 2
            px_center = (xmin + xmax) // 2
            is_moving = self._evaluate_target_dynamics(px_center, target_cy, now)

            # 1. PLAN MOTION WITH VLM FIRST (Computes continuous path and anti-overextension guidance)
            continuous_drive = False
            pulse_dur = 0.12
            if self.vlm_planner:
                sweet_spot = {"center_x": sx, "center_y": sy, "box_width": int(w * 0.35), "box_height": int(h * 0.35)}
                mock_det = DetectionResult(
                    detected=True,
                    category=cat_name,
                    confidence=conf,
                    bounding_box=(ymin, xmin, ymax, xmax),
                    material_color="Dynamic"
                )
                vlm_plan = self.vlm_planner.plan_movement(
                    mock_det,
                    sweet_spot,
                    dist_cm=dist_cm,
                    target_dist_cm=float(target_dist),
                    target_is_moving=is_moving
                )
                self.telemetry["vlm_plan"] = vlm_plan.to_dict()
                left_pwm = vlm_plan.left_pwm
                right_pwm = vlm_plan.right_pwm
                action = vlm_plan.action
                continuous_drive = vlm_plan.continuous_drive
                pulse_dur = max(0.08, min(0.20, vlm_plan.duration_ms / 1000.0))

                # Hard turn and base speed floor guarantees from robot config
                if "FORWARD" in action or "APPROACH" in action or "CRUISE" in action or "MEASURE" in action:
                    if left_pwm > 0: left_pwm = max(b_speed, left_pwm)
                    if right_pwm > 0: right_pwm = max(b_speed, right_pwm)
                elif "BACK" in action or "REVERSE" in action:
                    if left_pwm < 0: left_pwm = min(-b_speed, left_pwm)
                    if right_pwm < 0: right_pwm = min(-b_speed, right_pwm)

                if "TURN" in action or "CENTER" in action or "PIVOT" in action:
                    if left_pwm > 0 and left_pwm < t_speed:
                        left_pwm = t_speed
                    elif left_pwm < 0 and left_pwm > -t_speed:
                        left_pwm = -t_speed
                    if right_pwm > 0 and right_pwm < t_speed:
                        right_pwm = t_speed
                    elif right_pwm < 0 and right_pwm > -t_speed:
                        right_pwm = -t_speed

                if (left_pwm != 0 or right_pwm != 0):
                    self.vlm_planner.record_feedback(
                        action_taken=action,
                        pwm=max(abs(left_pwm), abs(right_pwm)),
                        duration_ms=int(pulse_dur * 1000),
                        prev_dx=ex,
                        new_dx=ex,
                        target_name=cat_name,
                        area_ratio=height_ratio
                    )
            else:
                # Fallback if no vlm_planner configured
                if abs(ex) > deadband_x:
                    pulse_dur = max(0.08, min(0.18, 0.08 + (abs(ex) / float(w // 2)) * 0.10))
                    if ex > 0:
                        action = "CENTER_RIGHT"
                        left_pwm, right_pwm = turn_pwm, -turn_pwm
                    else:
                        action = "CENTER_LEFT"
                        left_pwm, right_pwm = -turn_pwm, turn_pwm
                elif height_ratio < (target_h - 0.08):
                    action = "APPROACH_MEASURE"
                    pulse_dur = 0.12
                    left_pwm, right_pwm = f_speed, f_speed
                    continuous_drive = True
                elif height_ratio > (target_h + 0.10):
                    action = "BACK_UP"
                    pulse_dur = 0.10
                    left_pwm, right_pwm = -f_speed, -f_speed
                else:
                    action = "ALIGNED_MEASURING"
                    left_pwm, right_pwm = 0, 0

            # Arm elevation tracking (IK-like pitch up/down within servo bounds synced to latency)
            dt_lat = max(0.04, time.time() - now)
            self._track_arm_elevation(target_cy, h, latency_s=dt_lat, ymax=ymax)

            is_motion = self.config.get("motion_enabled", True)
            if not is_motion:
                if abs(ex) > deadband_x:
                    action = f"{action} [STATIONARY TURN]"
                else:
                    action = f"{action} [STATIONARY TRACK]"

            self.telemetry["status"] = "MEASURING"
            self.telemetry["action"] = action
            dyn_label = "Moving" if is_moving else "Stationary"
            self.telemetry["details"] = (
                f"{cat_name.title()}: {size_cat} ({width_mm:.0f}x{height_mm:.0f}mm, {dyn_label} @ {dist_cm:.0f}cm) | "
                f"{gripper_fit} | Action: {action}"
            )

            # Execute Sense-Think-Act Motion (Continuous smooth cruise along planned path)
            self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, is_target_moving=is_moving, continuous_drive=continuous_drive)
        else:
            self._target_dyn_history.clear()
            self._target_is_moving = False
            lost_duration = now - last_seen
            if lost_duration > 1.2:
                self.telemetry["target_found"] = False
                self.telemetry["target_box"] = []
                self.telemetry["status"] = "SEARCHING"
                self.telemetry["action"] = "STOPPED"
                target_str = filter_target if filter_target != "any" else "objects"
                self.telemetry["details"] = f"Searching for {target_str} to size..."
                if self.comm:
                    self.comm.send_stop()
                self.set_arm_down(open_gripper=True)

        return last_seen

    def _step_obstacle_avoidance(self, frame: np.ndarray, w: int, h: int, cx_img: int, last_seen: float) -> float:
        """Processes frame for monocular ground-plane corridor obstacle avoidance.

        Filters out floor grout lines, floor wood grains, and shadows with Gaussian smoothing.
        Only solid 3D obstacle clusters or AI object detections in the lower ground plane trigger collision avoidance.
        """
        now = time.time()
        col_w = w // 3
        y_top = int(h * 0.45)   # y ~ 216 px
        y_bot = int(h * 0.95)   # y ~ 456 px

        # 1. Optical Gradient & Edge Density Analysis on Ground Corridor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y_top:y_bot, :]

        # Gaussian blur suppresses high-frequency tile grout lines, floor wood grains, and reflections
        blurred = cv2.GaussianBlur(roi, (7, 7), 1.8)
        # Elevated thresholds (80, 200) ensure only high-contrast solid object boundaries trigger
        edges = cv2.Canny(blurred, 80, 200)

        # Morphological opening cleans up isolated line artifacts and shadow boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        # Calculate edge density in each corridor sector
        left_edges = clean_edges[:, :col_w]
        center_edges = clean_edges[:, col_w:2 * col_w]
        right_edges = clean_edges[:, 2 * col_w:]

        dens_l = float(cv2.countNonZero(left_edges)) / max(1.0, float(left_edges.size))
        dens_c = float(cv2.countNonZero(center_edges)) / max(1.0, float(center_edges.size))
        dens_r = float(cv2.countNonZero(right_edges)) / max(1.0, float(right_edges.size))

        # Real floor baseline: clean floor <= 0.015, tile/wood <= 0.035. Real 3D obstacles >= 0.08.
        # Subtract floor noise floor (0.035) so floor patterns never trigger false positive
        score_l = min(1.0, max(0.0, (dens_l - 0.035) * 6.5))
        score_c = min(1.0, max(0.0, (dens_c - 0.035) * 6.5))
        score_r = min(1.0, max(0.0, (dens_r - 0.035) * 6.5))

        # 2. Fuse AI Object Detections in Lower Ground Plane
        active_boxes = []
        if self.vision:
            try:
                detections = self.vision.detect_all(frame)
                for d in detections:
                    if d.confidence < 0.22:
                        continue
                    bb = d.bounding_box
                    if bb and len(bb) == 4:
                        ymin, xmin, ymax, xmax = bb
                        # Obstacle must enter immediate lower ground plane (y > 0.58 H)
                        if ymax > int(h * 0.58):
                            active_boxes.append((ymin, xmin, ymax, xmax, d.category, d.confidence))
                            obj_cx = (xmin + xmax) // 2
                            # Proximity weighted by distance to bottom of frame (closer = higher threat)
                            prox = ((float(ymax) - y_top) / float(y_bot - y_top)) ** 2
                            obj_w = float(xmax - xmin)
                            obj_h = float(ymax - ymin)
                            area_wt = (obj_w * obj_h) / float(w * h)
                            obj_score = min(1.0, prox * 0.65 + area_wt * 2.5)

                            if obj_cx < col_w:
                                score_l = max(score_l, obj_score)
                            elif obj_cx < 2 * col_w:
                                score_c = max(score_c, obj_score)
                            else:
                                score_r = max(score_r, obj_score)
            except Exception as e:
                print(f"[Activities] Vision error during obstacle check: {e}")

        # Compute lowest obstacle contact row in each corridor sector
        lowest_l = 0
        lowest_c = 0
        lowest_r = 0
        nz_l = np.where(left_edges > 0)[0]
        if len(nz_l) > 0: lowest_l = int(np.max(nz_l)) + y_top
        nz_c = np.where(center_edges > 0)[0]
        if len(nz_c) > 0: lowest_c = int(np.max(nz_c)) + y_top
        nz_r = np.where(right_edges > 0)[0]
        if len(nz_r) > 0: lowest_r = int(np.max(nz_r)) + y_top

        for bb_ymin, bb_xmin, bb_ymax, bb_xmax, _, _ in active_boxes:
            b_cx = (bb_xmin + bb_xmax) // 2
            if b_cx < col_w:
                lowest_l = max(lowest_l, bb_ymax)
            elif b_cx < 2 * col_w:
                lowest_c = max(lowest_c, bb_ymax)
            else:
                lowest_r = max(lowest_r, bb_ymax)

        dist_l = self.estimate_ground_distance(lowest_l, h) if lowest_l > y_top else 200.0
        dist_c = self.estimate_ground_distance(lowest_c, h) if lowest_c > y_top else 200.0
        dist_r = self.estimate_ground_distance(lowest_r, h) if lowest_r > y_top else 200.0

        # 3. Collision Risk Evaluation & Corridor Thresholding
        thresh = float(self.config.get("obstacle_threshold", 0.25))
        # Blocked if score exceeds threshold OR if physically closer than 28cm
        blocked_l = (score_l >= thresh) or (dist_l < 28.0)
        blocked_c = (score_c >= thresh) or (dist_c < 28.0)
        blocked_r = (score_r >= thresh) or (dist_r < 28.0)

        b_speed = self._get_base_speed()
        c_speed = max(b_speed, int(self.config.get("obstacle_cruise_speed", 235)))
        t_speed = max(self._get_turn_speed(), int(self.config.get("obstacle_turn_speed", 235)))

        # 4. Reactive Avoidance State Machine with Arm IK Environmental Scan
        if not blocked_c and not blocked_l and not blocked_r:
            self._oa_scan_state = "IDLE"
            action = "CRUISING_FORWARD"
            clear_path = "FORWARD"
            left_pwm, right_pwm = self._apply_trim(c_speed)
        elif blocked_c:
            # Front obstruction suspected: Use Arm IK to inspect environment before concluding obstruction!
            if self._oa_scan_state == "IDLE":
                self._oa_scan_state = "SCAN_ELEVATE"
                self._oa_scan_start_t = now
                action = "IK_ENVIRONMENT_SCAN"
                clear_path = "SCANNING"
                left_pwm, right_pwm = 0, 0
                if self.comm:
                    self.comm.send_stop()
                # Elevate arm via IK to Horizon Scan pose to check true 3D horizon
                self._track_arm_elevation(int(h * 0.38), h, latency_s=0.08)
                self.telemetry["details"] = "Obstruction suspected in front. Scanning 3D environment with Arm IK..."
            elif self._oa_scan_state == "SCAN_ELEVATE":
                # Give servos 0.25s to reach elevated scan pose and let camera stabilize
                if (now - self._oa_scan_start_t) >= 0.25:
                    self._oa_scan_state = "SCAN_VERIFY"
                action = "IK_ENVIRONMENT_SCAN"
                clear_path = "SCANNING"
                left_pwm, right_pwm = 0, 0
            elif self._oa_scan_state == "SCAN_VERIFY":
                # Verify whether the elevated view still detects solid obstacles in front
                has_ai_obstacle_in_center = any(
                    ((bb_xmin + bb_xmax) // 2) >= col_w and ((bb_xmin + bb_xmax) // 2) < 2 * col_w
                    for _, bb_xmin, _, bb_xmax, _, _ in active_boxes
                )
                is_solid_3d_obstacle = (score_c >= (thresh * 1.25)) or (dist_c < 24.0) or has_ai_obstacle_in_center

                if is_solid_3d_obstacle:
                    # Confirmed genuine 3D obstacle in front!
                    self._oa_scan_state = "AVOID"
                    self.telemetry["details"] = "Arm IK scan CONFIRMED physical obstacle! Executing avoidance maneuver."
                else:
                    # Elevated scan revealed center is clear (floor shadow/grout line dismissed)
                    self._oa_scan_state = "IDLE"
                    self.telemetry["details"] = "Arm IK scan cleared center path (floor artifact discarded). Resuming cruise."
                    action = "CRUISING_FORWARD"
                    clear_path = "FORWARD"
                    left_pwm, right_pwm = self._apply_trim(c_speed)

            if self._oa_scan_state == "AVOID":
                if not blocked_r and (dist_r > dist_l or score_r <= score_l):
                    action = "AVOID_RIGHT"
                    clear_path = "RIGHT"
                    left_pwm, right_pwm = t_speed, -t_speed
                elif not blocked_l:
                    action = "AVOID_LEFT"
                    clear_path = "LEFT"
                    left_pwm, right_pwm = -t_speed, t_speed
                else:
                    action = "ESCAPE_BACKUP"
                    clear_path = "REVERSE"
                    lp, rp = self._apply_trim(c_speed)
                    left_pwm, right_pwm = -lp, -rp
        elif blocked_l and not blocked_r:
            action = "VEER_RIGHT"
            clear_path = "RIGHT_FORWARD"
            left_pwm = min(255, c_speed + 15)
            right_pwm = max(b_speed, c_speed - 25)
            left_pwm, right_pwm = self._apply_trim_direct(left_pwm, right_pwm)
        elif blocked_r and not blocked_l:
            action = "VEER_LEFT"
            clear_path = "LEFT_FORWARD"
            left_pwm = max(b_speed, c_speed - 25)
            right_pwm = min(255, c_speed + 15)
            left_pwm, right_pwm = self._apply_trim_direct(left_pwm, right_pwm)
        else:
            action = "CAUTIOUS_FORWARD"
            clear_path = "NARROW_CENTER"
            left_pwm, right_pwm = self._apply_trim(c_speed)

        is_motion = self.config.get("motion_enabled", True)
        if not is_motion:
            action = f"{action} [DETECT ONLY]"

        # Update telemetry with calibrated distances in centimeters
        self.telemetry["distance_cm"] = dist_c
        self.telemetry["target_found"] = (blocked_c or blocked_l or blocked_r)
        self.telemetry["status"] = "AVOIDING" if (blocked_c or action.startswith("AVOID") or action.startswith("ESCAPE")) else "CRUISING"
        self.telemetry["action"] = action
        self.telemetry["target_size"] = round(dist_c, 1)
        self.telemetry["target_box"] = [y_top, col_w, y_bot, 2 * col_w] if blocked_c else []
        # Extract solid obstacle contours in corridor ROI with frame coordinate offset
        obs_contours = []
        raw_cnts, _ = cv2.findContours(clean_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in raw_cnts:
            if cv2.contourArea(cnt) >= 60:
                cnt_copy = cnt.copy()
                cnt_copy[:, :, 1] += y_top
                approx = cv2.approxPolyDP(cnt_copy, 2.0, True)
                if len(approx) >= 3:
                    obs_contours.append(approx.reshape(-1, 2).tolist())

        self.telemetry["obstacle"] = {
            "left_zone": "BLOCKED" if blocked_l else "CLEAR",
            "center_zone": "BLOCKED" if blocked_c else "CLEAR",
            "right_zone": "BLOCKED" if blocked_r else "CLEAR",
            "left_score": round(score_l * 100, 1),
            "center_score": round(score_c * 100, 1),
            "right_score": round(score_r * 100, 1),
            "left_dist_cm": dist_l,
            "center_dist_cm": dist_c,
            "right_dist_cm": dist_r,
            "action": action,
            "clear_path": clear_path,
            "contours": obs_contours,
        }
        self.telemetry["details"] = (
            f"Obstacle Nav [{action}] | L:{dist_l:.0f}cm [{'X' if blocked_l else 'O'}] "
            f"C:{dist_c:.0f}cm [{'X' if blocked_c else 'O'}] R:{dist_r:.0f}cm [{'X' if blocked_r else 'O'}] | Path: {clear_path}"
        )

        pulse_dur = 0.15 if (blocked_c or blocked_l or blocked_r) else 0.18
        continuous_drive = not (blocked_c or blocked_l or blocked_r)
        self._execute_gated_pulse(left_pwm, right_pwm, duration_s=pulse_dur, settle_s=0.04, continuous_drive=continuous_drive)

        return now
