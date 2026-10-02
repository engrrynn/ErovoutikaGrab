#!/usr/bin/env python3
"""Bluetooth connection diagnostics and interactive CLI for ErovoutikaGrab.
Applies calibrated servo angles loaded directly from config/robot_config.yaml.
"""

import os
import sys
import time
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.manipulation.arm_controller import ArmController

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/robot_config.yaml"))


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Saved] Configuration written to {CONFIG_PATH}")


def main():
    print("=== ErovoutikaGrab Bluetooth Diagnostics ===")

    cfg = load_config()
    comm_cfg = cfg.get("communication", {})
    port = comm_cfg.get("serial_port", "/dev/rfcomm0")
    baud = comm_cfg.get("baud_rate", 9600)
    mac = comm_cfg.get("bluetooth_mac", "20:25:08:00:46:FB")
    channel = comm_cfg.get("rfcomm_channel", 1)
    swap = cfg.get("motors", {}).get("swap_left_right", True)

    use_mock = "--mock" in sys.argv
    if use_mock:
        print("[Mode] Running with MockCommAdapter (Simulated)")
        adapter = MockCommAdapter(swap_left_right=swap)
    else:
        print(f"Target Bluetooth Device: {mac} on channel {channel}")
        print(f"Serial Port: {port} @ {baud} baud (swap_left_right={swap})\n")
        adapter = BluetoothSerialAdapter(
            port=port,
            baud_rate=baud,
            bluetooth_mac=mac,
            rfcomm_channel=channel,
            swap_left_right=swap
        )

    print("Attempting connection...")
    if not adapter.connect():
        print("[FAIL] Could not connect to Arduino via Bluetooth.")
        print("Note: If 'egrabbot-web.service' is running, it exclusively holds /dev/rfcomm0.")
        print("  - To free the port: sudo systemctl stop egrabbot-web.service")
        print("  - To restart service: sudo systemctl start egrabbot-web.service")
        print("  - Or test offline: python3 scripts/test_bluetooth.py --mock")
        print("Falling back to MockCommAdapter for testing...")
        adapter = MockCommAdapter(swap_left_right=swap)
        adapter.connect()


    print("[PASS] Connected! Measuring ping latency...")
    for i in range(3):
        adapter.ping()
        time.sleep(0.3)
        telem = adapter.get_latest_telemetry()
        print(f"  Ping attempt {i+1}: {telem.ping_ms} ms")

    arm = ArmController(adapter, config=cfg)

    print("\nReading telemetry...")
    time.sleep(0.5)
    telem = adapter.get_latest_telemetry()
    print(f"  Motors: L={telem.left_pwm} R={telem.right_pwm}")
    print(f"  Servos: S1={telem.s1_shoulder}° S2={telem.s2_elbow}° S3={telem.s3_gripper}°")
    print(f"  Macro Active: {telem.macro_active}")

    try:
        while True:
            # Reload to display any newly saved calibrations
            arm.reload_config()
            cfg = load_config()
            motors_cfg = cfg.get("motors", {})
            nudge_pwm = motors_cfg.get("nudge_pwm", 210)
            nudge_ms = motors_cfg.get("nudge_default_ms", 70)

            print("\n--- Interactive Command Menu (Using Calibrated Angles) ---")
            print(f" 1. Test Arm Stow        -> S1={arm.s1_stow}°, S2={arm.s2_stow}°, S3={arm.cur_s3}°")
            print(f" 2. Test Gripper Open    -> S3={arm.s3_open}°")
            print(f" 3. Test Gripper Close   -> S3={arm.s3_close}°")
            print(f" 4. Test Arm Down (Reach)-> S1={arm.s1_down}°, S2={arm.s2_down}°, S3={arm.cur_s3}°")
            print(f" 5. Full Pick Sequence   -> Calibrated Open -> Reach -> Close -> Lift")
            print(f" 6. Test Micro-Nudge Left (60ms @ {nudge_pwm} PWM)")
            print(f" 7. Test Micro-Nudge Right (60ms @ {nudge_pwm} PWM)")
            print(f" 8. Test Micro-Nudge Fwd  ({nudge_ms}ms @ {nudge_pwm} PWM)")
            print(f" 9. Center All Servos    -> S1={arm.s1_center}°, S2={arm.s2_center}°, S3={arm.s3_center}°")
            print(f"10. Adjust & Save Nudge Speed (Current: {nudge_pwm} PWM, {nudge_ms}ms)")
            print(f"11. Direct Move Gripper (Manual S3 Angle: e.g. 20°, 45°, 70°, 100°)")
            print(f"12. Test Arm Down Hover (S1=120°, S2=40°) [Off-Ground Ground Clearance Test]")
            print(f"13. Re-Attach All Servos (<ATTACH>) [Re-energize / Hold Torque]")
            print(f"14. Detach All Servos (<DETACH>) [Cut Torque / Free-Spin Hand Positioning]")
            print(" 0. Exit")

            choice = input("\nEnter choice [0-14]: ").strip()
            if choice == "0":
                break
            elif choice == "1":
                print(f"Sending Stow (S1={arm.s1_stow}°, S2={arm.s2_stow}°, S3={arm.cur_s3}°)...")
                arm.stow()
            elif choice == "2":
                print(f"Opening Gripper to calibrated {arm.s3_open}° (holding S1={arm.cur_s1}°, S2={arm.cur_s2}°)...")
                arm.open_gripper()
            elif choice == "3":
                print(f"Closing Gripper to calibrated {arm.s3_close}° (holding S1={arm.cur_s1}°, S2={arm.cur_s2}°)...")
                arm.close_gripper()
            elif choice == "4":
                print(f"Lowering Arm to calibrated pick reach (S1={arm.s1_down}°, S2={arm.s2_down}°)...")
                arm.arm_down()
                print(f"[Arm Down] Position reached: S1={arm.cur_s1}°, S2={arm.cur_s2}°, S3={arm.cur_s3}°")
                print("Tip: Use [2] (Open Gripper), [3] (Close Gripper), or [11] (Direct Move Gripper) to test jaw motion.")
            elif choice == "5":
                print("Executing full calibrated pick sequence...")
                arm.execute_pick_sequence()
            elif choice == "6":
                print(f"Sending Nudge Left (60ms @ {nudge_pwm} PWM)...")
                adapter.send_nudge("L", duration_ms=60, pwm=nudge_pwm)
            elif choice == "7":
                print(f"Sending Nudge Right (60ms @ {nudge_pwm} PWM)...")
                adapter.send_nudge("R", duration_ms=60, pwm=nudge_pwm)
            elif choice == "8":
                print(f"Sending Nudge Forward ({nudge_ms}ms @ {nudge_pwm} PWM)...")
                adapter.send_nudge("F", duration_ms=nudge_ms, pwm=nudge_pwm)
            elif choice == "9":
                print(f"Centering servos to calibrated neutrals (S1={arm.s1_center}°, S2={arm.s2_center}°, S3={arm.s3_center}°)...")
                arm.center()
            elif choice in ("10", "s", "S"):
                val_pwm = input(f"Enter new Nudge PWM [100-255, current: {nudge_pwm}]: ").strip()
                val_ms = input(f"Enter new Nudge Duration in ms [20-300, current: {nudge_ms}]: ").strip()
                new_pwm = int(val_pwm) if val_pwm.isdigit() else nudge_pwm
                new_ms = int(val_ms) if val_ms.isdigit() else nudge_ms
                print(f"Testing new nudge: Forward {new_ms}ms @ {new_pwm} PWM...")
                adapter.send_nudge("F", duration_ms=new_ms, pwm=new_pwm)
                time.sleep(0.5)
                save_ans = input(f"Save tested settings ({new_pwm} PWM, {new_ms}ms) to config/robot_config.yaml? (Y/n) [default: y]: ").strip().lower()
                if save_ans in ("", "y", "yes"):
                    cfg.setdefault("motors", {})
                    cfg["motors"]["nudge_pwm"] = new_pwm
                    cfg["motors"]["nudge_default_ms"] = new_ms
                    if new_pwm > cfg["motors"].get("base_speed", 210):
                        cfg["motors"]["base_speed"] = new_pwm
                    save_config(cfg)
            elif choice == "11":
                val_deg = input(f"Enter target Gripper (S3) angle [{arm.s3_min}-{arm.s3_max}°, current: {arm.cur_s3}°]: ").strip()
                if val_deg.isdigit():
                    deg = int(val_deg)
                    print(f"Directly commanding Gripper to {deg}° (holding S1={arm.cur_s1}°, S2={arm.cur_s2}°)...")
                    arm.move_gripper(deg)
                else:
                    print("Invalid angle.")
            elif choice == "12":
                print("Lowering Arm to Hover Position (S1=120°, S2=40°) - Gripper off ground...")
                arm.move_arm_sequential(120, 40, arm.cur_s3)
                print("Now try closing/opening gripper: [2] Open, [3] Close, [11] Direct Angle.")
            elif choice == "13":
                print("Sending <ATTACH> to re-attach and energize all servos (Pins 9, 10, 11)...")
                arm.attach()
                print("Command sent! Servos should now hold position with holding torque.")
            elif choice == "14":
                print("Sending <DETACH> to detach all servos (cut holding torque)...")
                arm.detach()
                print("Command sent! Servos can now be freely repositioned by hand.")
            else:
                print("Invalid option.")

            time.sleep(0.5)
            telem = adapter.get_latest_telemetry()
            print(f"Live State: S1={telem.s1_shoulder}° S2={telem.s2_elbow}° S3={telem.s3_gripper}° | Motors: L={telem.left_pwm} R={telem.right_pwm}")
    finally:
        adapter.send_stop()
        adapter.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
