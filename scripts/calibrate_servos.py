#!/usr/bin/env python3
"""Interactive Servo Angle Calibration & Constraint Discovery Tool.

Helps you safely center all 3 servos at 90° and step each servo individually
to identify the exact physical minimum and maximum angle constraints without stalling or binding.
Saves the discovered constraints to config/robot_config.yaml.
"""

import os
import sys
import time
import yaml

# Add parent directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.manipulation.arm_controller import ArmController

CONFIG_PATH = os.path.join(BASE_DIR, "config/robot_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\n[Saved] Updated servo constraints saved to {CONFIG_PATH}")


def main():
    print("=====================================================")
    print("   ErovoutikaGrab - Servo Calibration & Discovery    ")
    print("=====================================================\n")

    cfg = load_config()
    comm_cfg = cfg.get("communication", {})

    swap = cfg.get("motors", {}).get("swap_left_right", True)

    use_mock = "--mock" in sys.argv
    if use_mock:
        print("[Mode] Running with MockCommAdapter (Simulated Arduino)")
        comm = MockCommAdapter(swap_left_right=swap)
    else:
        port = comm_cfg.get("serial_port", "/dev/rfcomm0")
        baud = comm_cfg.get("baud_rate", 9600)
        mac = comm_cfg.get("bluetooth_mac", "20:25:08:00:46:FB")
        channel = comm_cfg.get("rfcomm_channel", 1)
        print(f"[Connecting] Bluetooth {mac} on {port} @ {baud} baud (swap_left_right={swap})...")
        comm = BluetoothSerialAdapter(port=port, baud_rate=baud, bluetooth_mac=mac, rfcomm_channel=channel, swap_left_right=swap)

    if not comm.connect():
        print("\n[Error] Connection failed.")
        print("Note: If 'egrabbot-web.service' is running, it exclusively holds /dev/rfcomm0.")
        print("  - To free the port: sudo systemctl stop egrabbot-web.service")
        print("  - To restart service: sudo systemctl start egrabbot-web.service")
        print("  - Or test offline: python3 scripts/calibrate_servos.py --mock")
        print("[Fallback] Falling back to MockCommAdapter for testing...")
        comm = MockCommAdapter(swap_left_right=swap)
        comm.connect()

    # Load recorded constraints directly from robot_config.yaml
    servos_cfg = cfg.get("servos", {})
    s1_cfg = servos_cfg.get("servo1_shoulder", {})
    s2_cfg = servos_cfg.get("servo2_elbow", {})
    s3_cfg = servos_cfg.get("servo3_gripper", {})

    s1_min = s1_cfg.get("min_angle", 60)
    s1_max = s1_cfg.get("max_angle", 180)
    s1_stow = s1_cfg.get("stow_angle", 60)
    s1_down = s1_cfg.get("down_angle", 180)
    s1_center = s1_cfg.get("center_angle", 90)

    s2_min = s2_cfg.get("min_angle", 0)
    s2_max = s2_cfg.get("max_angle", 80)
    s2_stow = s2_cfg.get("stow_angle", 80)
    s2_down = s2_cfg.get("down_angle", 0)
    s2_center = s2_cfg.get("center_angle", 80)

    s3_close = s3_cfg.get("close_angle", 70)
    s3_open = s3_cfg.get("open_angle", 110)
    s3_min = s3_cfg.get("min_angle", min(s3_close, s3_open))
    s3_max = s3_cfg.get("max_angle", max(s3_close, s3_open))
    s3_center = s3_cfg.get("center_angle", 90)

    arm = ArmController(comm, config=cfg)
    arm.cur_s1 = s1_center
    arm.cur_s2 = s2_center
    arm.cur_s3 = s3_center

    # Current angles (start at calibrated centers: S1=93°, S2=45°, S3=110°)
    s1 = s1_center
    s2 = s2_center
    s3 = s3_center

    # Safely command to calibrated neutral positions
    print(f"\n>>> Moving servos to neutral centers: S1={s1}°, S2={s2}°, S3={s3}°...")
    comm.send_servos(s1, s2, s3)
    time.sleep(0.5)

    selected_servo = 1  # 1 = S1, 2 = S2, 3 = S3

    print("\n-----------------------------------------------------")
    print("Controls:")
    print("  1 / 2 / 3      : Select Servo (1=Shoulder, 2=Elbow, 3=Gripper)")
    print("  c              : Command all 3 to calibrated neutral centers")
    print("  + / -          : Nudge selected servo by +1° / -1°")
    print("  ++ / --        : Nudge selected servo by +5° / -5°")
    print("  g <angle>      : Go directly to angle (e.g. 'g 45')")
    print("  stow / up      : Test UP/STOW pose (sequential: S2 lifts, S1 retracts)")
    print("  down           : Test DOWN/REACH pose (sequential: S1 extends, S2 lowers)")
    print("  open / close   : Test Gripper OPEN / CLOSE")
    print("  setmin         : Record current angle as MIN limit for selected servo")
    print("  setmax         : Record current angle as MAX limit for selected servo")
    print("  setup / setstow: Record current angle as UP / STOW position (S1/S2)")
    print("  setdown        : Record current angle as PICK REACH position (S1/S2)")
    print("  setcenter      : Record current angle as NEUTRAL CENTER position")
    print("  setopen        : Record current angle as Gripper OPEN position")
    print("  setclose       : Record current angle as Gripper CLOSE position")
    print("  save           : Save recorded limits to config/robot_config.yaml")
    print("  q              : Quit")
    print("-----------------------------------------------------")

    try:
        while True:
            servo_names = {1: "Servo 1 (Shoulder Base)", 2: "Servo 2 (Elbow Base)", 3: "Servo 3 (Gripper)"}
            cur_angle = s1 if selected_servo == 1 else (s2 if selected_servo == 2 else s3)
            print(f"\n[ACTIVE: {servo_names[selected_servo]}] Current Target: {cur_angle}° | State: (S1={s1}°, S2={s2}°, S3={s3}°)")
            print(f"Recorded Constraints -> S1: [{s1_min}°, {s1_max}°] | S2: [{s2_min}°, {s2_max}°] | S3: [{s3_min}°, {s3_max}°]")

            cmd = input("Command [1,2,3, c, +, -, ++, --, g <deg>, setmin, setmax, setstow, setdown, setcenter, save, q]: ").strip().lower()

            if cmd == "q":
                break
            elif cmd == "1":
                selected_servo = 1
            elif cmd == "2":
                selected_servo = 2
            elif cmd == "3":
                selected_servo = 3
            elif cmd == "c":
                print(f">>> Centering all servos to neutral: S1={s1_center}°, S2={s2_center}°, S3={s3_center}°...")
                arm.move_arm_sequential(s1_center, s2_center, s3_center, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd in ("stow", "up"):
                print(f">>> Moving arm to UP/STOW pose: S1={s1_stow}°, S2={s2_stow}°, S3={s3}° (sequential)...")
                arm.move_arm_sequential(s1_stow, s2_stow, s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd == "down":
                print(f">>> Moving arm to DOWN/REACH pose: S1={s1_down}°, S2={s2_down}°, S3={s3}° (sequential)...")
                arm.move_arm_sequential(s1_down, s2_down, s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd == "open":
                print(f">>> Opening gripper to {s3_open}°...")
                arm.move_gripper(s3_open, clamp_to_bounds=False)
                s3 = arm.cur_s3
            elif cmd == "close":
                print(f">>> Closing gripper to {s3_close}°...")
                arm.move_gripper(s3_close, clamp_to_bounds=False)
                s3 = arm.cur_s3
            elif cmd == "+":
                target_s1 = min(180, s1 + 1) if selected_servo == 1 else s1
                target_s2 = min(180, s2 + 1) if selected_servo == 2 else s2
                target_s3 = min(180, s3 + 1) if selected_servo == 3 else s3
                if selected_servo == 3:
                    arm.move_gripper(target_s3, clamp_to_bounds=False)
                else:
                    arm.move_arm_sequential(target_s1, target_s2, target_s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd == "-":
                target_s1 = max(0, s1 - 1) if selected_servo == 1 else s1
                target_s2 = max(0, s2 - 1) if selected_servo == 2 else s2
                target_s3 = max(0, s3 - 1) if selected_servo == 3 else s3
                if selected_servo == 3:
                    arm.move_gripper(target_s3, clamp_to_bounds=False)
                else:
                    arm.move_arm_sequential(target_s1, target_s2, target_s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd == "++":
                target_s1 = min(180, s1 + 5) if selected_servo == 1 else s1
                target_s2 = min(180, s2 + 5) if selected_servo == 2 else s2
                target_s3 = min(180, s3 + 5) if selected_servo == 3 else s3
                if selected_servo == 3:
                    arm.move_gripper(target_s3, clamp_to_bounds=False)
                else:
                    arm.move_arm_sequential(target_s1, target_s2, target_s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd == "--":
                target_s1 = max(0, s1 - 5) if selected_servo == 1 else s1
                target_s2 = max(0, s2 - 5) if selected_servo == 2 else s2
                target_s3 = max(0, s3 - 5) if selected_servo == 3 else s3
                if selected_servo == 3:
                    arm.move_gripper(target_s3, clamp_to_bounds=False)
                else:
                    arm.move_arm_sequential(target_s1, target_s2, target_s3, clamp_to_bounds=False)
                s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
            elif cmd.startswith("g "):
                try:
                    target_deg = int(cmd.split()[1])
                    target_s1 = max(0, min(180, target_deg)) if selected_servo == 1 else s1
                    target_s2 = max(0, min(180, target_deg)) if selected_servo == 2 else s2
                    target_s3 = max(0, min(180, target_deg)) if selected_servo == 3 else s3
                    if selected_servo == 3:
                        arm.move_gripper(target_s3, clamp_to_bounds=False)
                    else:
                        arm.move_arm_sequential(target_s1, target_s2, target_s3, clamp_to_bounds=False)
                    s1, s2, s3 = arm.cur_s1, arm.cur_s2, arm.cur_s3
                except Exception as e:
                    print(f"Invalid format or error: {e}. Use: g 90")
            elif cmd == "setmin":
                if selected_servo == 1: s1_min = s1
                elif selected_servo == 2: s2_min = s2
                elif selected_servo == 3: s3_min = s3
                print(f">>> Recorded MIN for Servo {selected_servo}: {cur_angle}°")
            elif cmd == "setmax":
                if selected_servo == 1: s1_max = s1
                elif selected_servo == 2: s2_max = s2
                elif selected_servo == 3: s3_max = s3
                print(f">>> Recorded MAX for Servo {selected_servo}: {cur_angle}°")
            elif cmd in ("setstow", "setup"):
                if selected_servo == 1: s1_stow = s1
                elif selected_servo == 2: s2_stow = s2
                print(f">>> Recorded UP/STOW angle for Servo {selected_servo}: {cur_angle}°")
            elif cmd == "setdown":
                if selected_servo == 1: s1_down = s1
                elif selected_servo == 2: s2_down = s2
                print(f">>> Recorded DOWN/REACH angle for Servo {selected_servo}: {cur_angle}°")
            elif cmd == "setcenter":
                if selected_servo == 1: s1_center = s1
                elif selected_servo == 2: s2_center = s2
                elif selected_servo == 3: s3_center = s3
                print(f">>> Recorded CENTER angle for Servo {selected_servo}: {cur_angle}°")
            elif cmd == "setopen":
                s3_open = s3
                print(f">>> Recorded OPEN angle for Gripper: {s3}°")
            elif cmd == "setclose":
                s3_close = s3
                print(f">>> Recorded CLOSE angle for Gripper: {s3}°")
            elif cmd == "save":
                cfg.setdefault("servos", {})
                cfg["servos"].setdefault("servo1_shoulder", {})
                cfg["servos"].setdefault("servo2_elbow", {})
                cfg["servos"].setdefault("servo3_gripper", {})

                # Ensure bounds fully encapsulate stow/up and down angles
                s1_max = max(s1_max, s1_down, s1_stow)
                s1_min = min(s1_min, s1_down, s1_stow)
                cfg["servos"]["servo1_shoulder"]["min_angle"] = s1_min
                cfg["servos"]["servo1_shoulder"]["max_angle"] = s1_max
                cfg["servos"]["servo1_shoulder"]["up_angle"] = s1_stow
                cfg["servos"]["servo1_shoulder"]["stow_angle"] = s1_stow
                cfg["servos"]["servo1_shoulder"]["down_angle"] = s1_down
                cfg["servos"]["servo1_shoulder"]["center_angle"] = s1_center

                s2_max = max(s2_max, s2_down, s2_stow)
                s2_min = min(s2_min, s2_down, s2_stow)
                cfg["servos"]["servo2_elbow"]["min_angle"] = s2_min
                cfg["servos"]["servo2_elbow"]["max_angle"] = s2_max
                cfg["servos"]["servo2_elbow"]["up_angle"] = s2_stow
                cfg["servos"]["servo2_elbow"]["stow_angle"] = s2_stow
                cfg["servos"]["servo2_elbow"]["down_angle"] = s2_down
                cfg["servos"]["servo2_elbow"]["center_angle"] = s2_center

                s3_min_bound = min(s3_close, s3_open)
                s3_max_bound = max(s3_close, s3_open)
                cfg["servos"]["servo3_gripper"]["min_angle"] = s3_min_bound
                cfg["servos"]["servo3_gripper"]["max_angle"] = s3_max_bound
                cfg["servos"]["servo3_gripper"]["close_angle"] = s3_close
                cfg["servos"]["servo3_gripper"]["open_angle"] = s3_open
                cfg["servos"]["servo3_gripper"]["center_angle"] = s3_center
                save_config(cfg)
                arm.reload_config(force_disk=True)
            else:
                print("Unknown command.")

            time.sleep(0.1)

    finally:
        comm.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
