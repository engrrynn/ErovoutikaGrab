#!/usr/bin/env python3
"""Main Application Entrypoint for ErovoutikaGrab.

Launches the Autonomous Mobile Manipulator system and the LCD HDMI HUD Dashboard.
Usage:
    python3 scripts/run_robot.py                      # Normal run with HDMI HUD
    python3 scripts/run_robot.py --headless           # Headless / Terminal console mode
    python3 scripts/run_robot.py --mock-comm          # Real camera, simulated Arduino
    python3 scripts/run_robot.py --mock-all           # Full simulation mode (no hardware required)
"""

import argparse
import os
import sys
import time
import yaml

# Setup import path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.camera.v4l2_cam import V4L2CameraAdapter
from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.adapters.vision.mock_vision import MockVisionAdapter
from src.adapters.vision.tracker_adapter import VisualServoingTracker
from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter
from src.core.context import RobotContext
from src.core.state_machine import AutonomousStateMachine
from src.manipulation.arm_controller import ArmController
from src.navigation.visual_servoing import VisualServoingController


def load_yaml(rel_path: str):
    full_path = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_headless_loop(context: RobotContext, sm: AutonomousStateMachine):
    """Console loop for headless execution without X11/Wayland display."""
    print("\n[Headless Mode] Starting autonomous loop. Press Ctrl+C to stop.")
    sm.start_autonomous_run()
    last_print = 0.0

    try:
        while True:
            sm.step()
            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                state = context.get_state()
                telem = context.get_telemetry()
                det = context.get_detection()
                det_name = det.category if (det and det.detected) else "None"
                roi = context.get_tracked_roi()

                print(
                    f"\r[STATE: {state:<16}] "
                    f"Target: {det_name:<12} | "
                    f"Tracked ROI: {str(roi):<18} | "
                    f"Servos: {telem.s1_shoulder}°, {telem.s2_elbow}°, {telem.s3_gripper}° | "
                    f"Motors: L={telem.left_pwm} R={telem.right_pwm} | "
                    f"Ping: {telem.ping_ms}ms",
                    end="",
                    flush=True
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[Headless Mode] Keyboard interrupt received.")


def main():
    parser = argparse.ArgumentParser(description="ErovoutikaGrab Autonomous Object Categorizer & Picker")
    parser.add_argument("--mock-comm", action="store_true", help="Simulate Arduino communication")
    parser.add_argument("--mock-cam", action="store_true", help="Simulate camera video stream")
    parser.add_argument("--mock-vision", action="store_true", help="Simulate vision detections")
    parser.add_argument("--mock-all", action="store_true", help="Simulate all hardware (dry run)")
    parser.add_argument("--headless", action="store_true", help="Run in terminal without GUI display")
    parser.add_argument("--fullscreen", action="store_true", default=True, help="Launch HUD in fullscreen mode on display (default: True)")
    parser.add_argument("--no-fullscreen", dest="fullscreen", action="store_false", help="Disable default fullscreen mode")
    parser.add_argument("--windowed", action="store_true", help="Launch in legacy windowed mode instead of fullscreen")
    args = parser.parse_args()

    if args.mock_all:
        args.mock_comm = True
        args.mock_cam = True
        args.mock_vision = True

    # Auto-bind HDMI / Wayland display if available
    if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"
    if not os.environ.get("XDG_RUNTIME_DIR") and os.path.exists("/run/user/1000"):
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
    if not os.environ.get("WAYLAND_DISPLAY") and os.path.exists("/run/user/1000/wayland-0"):
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if not args.headless and not args.windowed and args.fullscreen:
        if has_display:
            print("\n[Launch] Starting ErovoutikaGrab in default Fullscreen Robot HUD mode...")
            launcher = os.path.join(BASE_DIR, "scripts/launch_desktop_gui.py")
            cmd = [sys.executable, launcher]
            if args.mock_all or args.mock_comm:
                cmd.append("--mock")
            import subprocess
            subprocess.run(cmd)
            return
        else:
            print("[Info] No graphical display detected. Falling back to --headless mode.")
            args.headless = True

    # 1. Load Configurations
    robot_cfg = load_yaml("config/robot_config.yaml")
    vision_cfg = load_yaml("config/vision_config.yaml")

    print("\n==========================================")
    print("   ErovoutikaGrab Autonomous System 2.0   ")
    print("==========================================\n")

    # 2. Instantiate Adapters
    # Communication Adapter
    comm_cfg = robot_cfg.get("communication", {})
    if args.mock_comm:
        print("[Setup] Using MockCommAdapter (Simulated Arduino)")
        comm_adapter = MockCommAdapter()
    else:
        print(f"[Setup] Using BluetoothSerialAdapter ({comm_cfg.get('serial_port', '/dev/rfcomm0')})")
        comm_adapter = BluetoothSerialAdapter(
            port=comm_cfg.get("serial_port", "/dev/rfcomm0"),
            baud_rate=comm_cfg.get("baud_rate", 9600),
            bluetooth_mac=comm_cfg.get("bluetooth_mac", "20:25:08:00:46:FB"),
            rfcomm_channel=comm_cfg.get("rfcomm_channel", 1)
        )
    comm_adapter.connect()

    # Camera Adapter
    cam_cfg = vision_cfg.get("camera", {})
    if args.mock_cam:
        print("[Setup] Using MockCameraAdapter (Synthetic Video)")
        cam_adapter = MockCameraAdapter(
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30)
        )
    else:
        print(f"[Setup] Using V4L2CameraAdapter (/dev/video{cam_cfg.get('device_index', 0)})")
        cam_adapter = V4L2CameraAdapter(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            target_fps=cam_cfg.get("fps", 30)
        )
    cam_adapter.start()

    # Vision Adapter (YOLOE Open-Vocabulary Object Detection & Categorization)
    if args.mock_vision:
        print("[Setup] Using MockVisionAdapter (Simulation)")
        vision_adapter = MockVisionAdapter()
    else:
        yoloe_cfg = vision_cfg.get("yoloe", {})
        model_path = yoloe_cfg.get("model_path", "models/yoloe-26n-seg.pt")
        conf_thresh = yoloe_cfg.get("confidence_threshold", 0.35)
        imgsz = yoloe_cfg.get("imgsz", 320)
        target_classes = yoloe_cfg.get("target_classes", None)
        try:
            print(f"[Setup] Initializing YOLOEVisionAdapter ({model_path})...")
            vision_adapter = YOLOEVisionAdapter(
                model_path=model_path,
                confidence_threshold=conf_thresh,
                imgsz=imgsz,
                target_classes=target_classes
            )
            if not vision_adapter.model_loaded:
                print("[Setup] Notice: YOLOE model not ready, falling back to MockVisionAdapter.")
                vision_adapter = MockVisionAdapter()
        except Exception as e:
            print(f"[Setup] Notice: YOLOE init exception ({e}), falling back to MockVisionAdapter.")
            vision_adapter = MockVisionAdapter()

    # 3. Core Context & Calibration
    context = RobotContext(config={"robot": robot_cfg, "vision": vision_cfg})
    spot = vision_cfg.get("grasp_sweet_spot", {})
    context.set_sweet_spot(
        x=spot.get("center_x", 320),
        y=spot.get("center_y", 380),
        w=spot.get("box_width", 120),
        h=spot.get("box_height", 90)
    )

    # 4. Navigation & Manipulation
    motors_cfg = robot_cfg.get("motors", {})
    vs_cfg = robot_cfg.get("visual_servoing", {})
    servoing_controller = VisualServoingController(
        center_x=context.sweet_spot_x,
        grasp_y=context.sweet_spot_y,
        tol_x=spot.get("align_tolerance_x", 20),
        tol_y=spot.get("align_tolerance_y", 25),
        base_speed=motors_cfg.get("base_speed", 210),
        turn_speed=motors_cfg.get("turn_speed", 200),
        min_overcoming_pwm=motors_cfg.get("min_overcoming_pwm", 190),
        pulse_threshold_px=vs_cfg.get("pulse_threshold_px", 45),
        nudge_duration_ms=motors_cfg.get("nudge_default_ms", 70),
        nudge_pwm=motors_cfg.get("nudge_pwm", 210)
    )

    arm_controller = ArmController(comm_adapter)
    tracker = VisualServoingTracker()

    # 5. Autonomous State Machine
    state_machine = AutonomousStateMachine(
        context=context,
        camera=cam_adapter,
        comm=comm_adapter,
        vision=vision_adapter,
        tracker=tracker,
        servoing=servoing_controller,
        arm=arm_controller
    )

    try:
        if args.headless:
            run_headless_loop(context, state_machine)
        else:
            from PyQt5.QtWidgets import QApplication
            from src.ui.hud_dashboard import HUDDashboard

            app = QApplication(sys.argv)
            dashboard = HUDDashboard(context, state_machine)
            if args.fullscreen:
                dashboard.showFullScreen()
            else:
                dashboard.show()

            print("[Ready] GUI active. Entering UI event loop.\n")
            app.exec_()

    finally:
        print("\n[Shutdown] Stopping robot safely...")
        state_machine.stop()
        comm_adapter.disconnect()
        cam_adapter.stop()
        print("[Shutdown] Complete.")


if __name__ == "__main__":
    main()
