#!/usr/bin/env python3
"""Interactive Motor & Speed Calibration Tool for ErovoutikaGrab.

Calibrates and fine-tunes:
1. Minimum Overcoming PWM (Static Friction / Stiction Floor Ramp Test)
2. Cruising Base Speed & Straight-Line L/R Wheel Balance Trim
3. Pivot Turn Speed
4. Anti-Stiction Micro-Nudge Torque (PWM) and Pulse Duration (ms)
5. Live WASD Teleoperation Test Drive with Emergency Stop

Saves all calibrated parameters directly to config/robot_config.yaml so they
are automatically used by autonomous visual servoing, test scripts, and state machines.
"""

import os
import select
import sys
import time
import yaml

# Add parent directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter

CONFIG_PATH = os.path.join(BASE_DIR, "config/robot_config.yaml")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\n[Saved] Motor calibration parameters successfully written to {CONFIG_PATH}")


class MotorCalibrationSession:
    """Manages motor calibration state, test runs, and YAML synchronization."""

    def __init__(self, comm_adapter, config):
        self.comm = comm_adapter
        self.config = config
        motors_cfg = config.get("motors", {})

        # Core motor parameters
        self.base_speed = int(motors_cfg.get("base_speed", 210))
        self.turn_speed = int(motors_cfg.get("turn_speed", 200))
        self.min_overcoming_pwm = int(motors_cfg.get("min_overcoming_pwm", 190))
        self.nudge_pwm = int(motors_cfg.get("nudge_pwm", 210))
        self.nudge_default_ms = int(motors_cfg.get("nudge_default_ms", 70))
        self.max_speed = int(motors_cfg.get("max_speed", 255))

        # Balance trim (Offset added to Left or Right to ensure straight-line driving)
        # positive trim_offset = right motor slightly faster; negative = left faster
        self.trim_offset = int(motors_cfg.get("trim_offset", 0))
        self.swap_left_right = bool(motors_cfg.get("swap_left_right", True))

    def get_drive_pwms(self, speed, direction=1):
        """Calculate left and right PWM applying balance trim offset."""
        # direction: 1 for forward, -1 for backward
        left = speed
        right = speed
        if self.trim_offset > 0:
            # Veers left, boost left or reduce right
            right = max(100, right - self.trim_offset)
        elif self.trim_offset < 0:
            # Veers right, reduce left
            left = max(100, left - abs(self.trim_offset))

        return int(left * direction), int(right * direction)

    def emergency_stop(self):
        """Immediately stop all motors."""
        self.comm.send_stop()

    def timed_drive(self, left_pwm, right_pwm, duration_sec):
        """Drive with specified PWMs for duration, then stop."""
        self.comm.send_drive(left_pwm, right_pwm)
        time.sleep(duration_sec)
        self.comm.send_stop()

    def save_to_config(self):
        """Sync current calibrated values back into robot_config.yaml preserving all other sections."""
        full_cfg = load_config()
        full_cfg.setdefault("motors", {})
        m = full_cfg["motors"]
        m["base_speed"] = self.base_speed
        m["turn_speed"] = self.turn_speed
        m["min_overcoming_pwm"] = self.min_overcoming_pwm
        m["nudge_pwm"] = self.nudge_pwm
        m["nudge_default_ms"] = self.nudge_default_ms
        m["max_speed"] = self.max_speed
        m["trim_offset"] = self.trim_offset
        m["swap_left_right"] = self.swap_left_right

        self.config.setdefault("motors", {})
        self.config["motors"].update(m)

        save_config(full_cfg)


def run_stiction_ramp_test(session):
    """Gradually ramp PWM up in short bursts until wheels overcome static friction."""
    print("\n=======================================================")
    print("   🔍 STATIC FRICTION (STICTION FLOOR) RAMP TEST       ")
    print("=======================================================")
    print("Instructions:")
    print("1. Place robot on your target surface (floor, mat, table).")
    print("2. The test will send short 300ms pulses starting at 140 PWM,")
    print("   increasing by +5 PWM every step.")
    print("3. Press [ENTER] as soon as BOTH wheels break stiction and turn!")
    print("4. Press [Q] to cancel.")
    print("=======================================================\n")

    input("Press Enter to begin ramp test...")

    pwm = 140
    detected_pwm = None

    # Enable non-blocking keyboard input
    try:
        while pwm <= 255:
            print(f"\r>>> Testing PWM: {pwm} (Pulse 300ms)... [Press Enter when wheels spin, Q to abort] ", end="", flush=True)

            # Send brief pulse
            session.comm.send_drive(pwm, pwm)
            pulse_start = time.time()
            user_input = None

            # Poll for keypress during pulse and rest window
            while time.time() - pulse_start < 0.3:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.readline().strip()
                    user_input = ch
                    break

            session.comm.send_stop()

            if user_input is not None:
                if user_input.lower() == "q":
                    print("\n[Aborted] Ramp test cancelled.")
                    return
                detected_pwm = pwm
                break

            # Rest interval between pulses
            rest_start = time.time()
            while time.time() - rest_start < 0.6:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.readline().strip()
                    user_input = ch
                    break

            if user_input is not None:
                if user_input.lower() == "q":
                    print("\n[Aborted] Ramp test cancelled.")
                    return
                detected_pwm = pwm
                break

            pwm += 5

    finally:
        session.emergency_stop()

    if detected_pwm is not None:
        print(f"\n\n[SUCCESS] Wheels broke static friction at: {detected_pwm} PWM!")
        print(f"Previous min_overcoming_pwm: {session.min_overcoming_pwm} PWM")
        confirm = input(f"Set and SAVE min_overcoming_pwm={detected_pwm} to config/robot_config.yaml? (Y/n) [default: y]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            session.min_overcoming_pwm = detected_pwm
            # Also adjust nudge_pwm and base_speed floor if they were lower than stiction
            if session.nudge_pwm < detected_pwm:
                session.nudge_pwm = detected_pwm
            if session.base_speed < detected_pwm:
                session.base_speed = detected_pwm
            session.save_to_config()
            print(f"[Updated & Saved] min_overcoming_pwm = {session.min_overcoming_pwm} PWM")
    else:
        print("\n[Warning] Reached max PWM (255) without confirmation.")


def run_straight_drive_test(session):
    """Test straight line driving and calibrate balance trim."""
    orig_speed = session.base_speed
    orig_trim = session.trim_offset
    while True:
        lpwm, rpwm = session.get_drive_pwms(session.base_speed, direction=1)
        print("\n--- Straight-Line Drive Test ---")
        print(f"Current Settings: Base Speed = {session.base_speed} PWM | Trim Offset = {session.trim_offset}")
        print(f"Effective Motor PWMs -> Left: {lpwm}, Right: {rpwm}")
        print("Options:")
        print("  [1] Drive Forward 0.8s burst")
        print("  [2] Drive Forward 1.5s burst")
        print("  [3] Drive Backward 0.8s burst")
        print("  [4] Increase Base Speed (+5 PWM)")
        print("  [5] Decrease Base Speed (-5 PWM)")
        print("  [6] Veers Left -> Trim +3 (boost left relative to right)")
        print("  [7] Veers Right -> Trim -3 (boost right relative to left)")
        print("  [8] Reset Trim Offset to 0")
        print("  [S] SAVE tested speed & trim to config/robot_config.yaml")
        print("  [0] Done / Back to Main Menu")

        choice = input("\nEnter choice: ").strip().upper()
        if choice == "0":
            if session.base_speed != orig_speed or session.trim_offset != orig_trim:
                ans = input(f"Save tested base speed ({session.base_speed} PWM) to YAML? (Y/n) [default: y]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    session.save_to_config()
            break
        elif choice == "S":
            session.save_to_config()
            orig_speed = session.base_speed
            orig_trim = session.trim_offset
        elif choice == "1":
            print(f"Driving Forward (0.8s @ L={lpwm}, R={rpwm})...")
            session.timed_drive(lpwm, rpwm, 0.8)
        elif choice == "2":
            print(f"Driving Forward (1.5s @ L={lpwm}, R={rpwm})...")
            session.timed_drive(lpwm, rpwm, 1.5)
        elif choice == "3":
            blpwm, brpwm = session.get_drive_pwms(session.base_speed, direction=-1)
            print(f"Driving Backward (0.8s @ L={blpwm}, R={brpwm})...")
            session.timed_drive(blpwm, brpwm, 0.8)
        elif choice == "4":
            session.base_speed = min(session.max_speed, session.base_speed + 5)
            print(f"Base speed increased to: {session.base_speed} PWM")
        elif choice == "5":
            session.base_speed = max(100, session.base_speed - 5)
            print(f"Base speed decreased to: {session.base_speed} PWM")
        elif choice == "6":
            session.trim_offset += 3
            print(f"Trim offset adjusted to: {session.trim_offset} (Veer Left compensated)")
        elif choice == "7":
            session.trim_offset -= 3
            print(f"Trim offset adjusted to: {session.trim_offset} (Veer Right compensated)")
        elif choice == "8":
            session.trim_offset = 0
            print("Trim offset reset to 0.")


def run_pivot_turn_test(session):
    """Test and calibrate pivot turning speed."""
    orig_turn = session.turn_speed
    while True:
        tspd = session.turn_speed
        print("\n--- Pivot Turn Speed Calibration ---")
        print(f"Current Turn Speed: {tspd} PWM")
        print("Options:")
        print("  [1] Pivot Left (0.5s burst)")
        print("  [2] Pivot Right (0.5s burst)")
        print("  [3] Pivot Left (1.0s burst)")
        print("  [4] Pivot Right (1.0s burst)")
        print("  [+] Increase Turn Speed (+5 PWM)")
        print("  [-] Decrease Turn Speed (-5 PWM)")
        print("  [S] SAVE tested turn speed to config/robot_config.yaml")
        print("  [0] Done / Back to Main Menu")

        choice = input("\nEnter choice: ").strip().upper()
        if choice == "0":
            if session.turn_speed != orig_turn:
                ans = input(f"Save tested turn speed ({session.turn_speed} PWM) to YAML? (Y/n) [default: y]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    session.save_to_config()
            break
        elif choice == "S":
            session.save_to_config()
            orig_turn = session.turn_speed
        elif choice == "1":
            print(f"Pivot Left (0.5s @ {tspd} PWM)...")
            session.timed_drive(-tspd, tspd, 0.5)
        elif choice == "2":
            print(f"Pivot Right (0.5s @ {tspd} PWM)...")
            session.timed_drive(tspd, -tspd, 0.5)
        elif choice == "3":
            print(f"Pivot Left (1.0s @ {tspd} PWM)...")
            session.timed_drive(-tspd, tspd, 1.0)
        elif choice == "4":
            print(f"Pivot Right (1.0s @ {tspd} PWM)...")
            session.timed_drive(tspd, -tspd, 1.0)
        elif choice in ("+", "I"):
            session.turn_speed = min(session.max_speed, session.turn_speed + 5)
            print(f"Turn speed increased to: {session.turn_speed} PWM")
        elif choice in ("-", "D"):
            session.turn_speed = max(100, session.turn_speed - 5)
            print(f"Turn speed decreased to: {session.turn_speed} PWM")


def run_micro_nudge_test(session):
    """Test and calibrate anti-stiction micro-nudge pulse duration and PWM."""
    orig_npwm = session.nudge_pwm
    orig_nms = session.nudge_default_ms
    while True:
        npwm = session.nudge_pwm
        nms = session.nudge_default_ms
        print("\n--- Micro-Nudge Anti-Stiction Calibration ---")
        print(f"Current Nudge Parameters: Torque = {npwm} PWM | Duration = {nms} ms")
        print("Options:")
        print("  [1] Test Nudge Forward  (F)")
        print("  [2] Test Nudge Backward (B)")
        print("  [3] Test Nudge Left     (L)")
        print("  [4] Test Nudge Right    (R)")
        print("  [5] Increase Duration (+10 ms)")
        print("  [6] Decrease Duration (-10 ms)")
        print("  [7] Increase Torque PWM (+5 PWM)")
        print("  [8] Decrease Torque PWM (-5 PWM)")
        print("  [S] SAVE tested nudge settings to config/robot_config.yaml")
        print("  [0] Done / Back to Main Menu")

        choice = input("\nEnter choice: ").strip().upper()
        if choice == "0":
            if session.nudge_pwm != orig_npwm or session.nudge_default_ms != orig_nms:
                ans = input(f"Save tested nudge settings ({session.nudge_pwm} PWM, {session.nudge_default_ms}ms) to YAML? (Y/n) [default: y]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    session.save_to_config()
            break
        elif choice == "S":
            session.save_to_config()
            orig_npwm = session.nudge_pwm
            orig_nms = session.nudge_default_ms
        elif choice == "1":
            print(f"Sending Nudge Forward ({nms}ms @ {npwm} PWM)...")
            session.comm.send_nudge("F", duration_ms=nms, pwm=npwm)
        elif choice == "2":
            print(f"Sending Nudge Backward ({nms}ms @ {npwm} PWM)...")
            session.comm.send_nudge("B", duration_ms=nms, pwm=npwm)
        elif choice == "3":
            print(f"Sending Nudge Left ({nms}ms @ {npwm} PWM)...")
            session.comm.send_nudge("L", duration_ms=nms, pwm=npwm)
        elif choice == "4":
            print(f"Sending Nudge Right ({nms}ms @ {npwm} PWM)...")
            session.comm.send_nudge("R", duration_ms=nms, pwm=npwm)
        elif choice == "5":
            session.nudge_default_ms = min(300, session.nudge_default_ms + 10)
            print(f"Nudge duration set to: {session.nudge_default_ms} ms")
        elif choice == "6":
            session.nudge_default_ms = max(20, session.nudge_default_ms - 10)
            print(f"Nudge duration set to: {session.nudge_default_ms} ms")
        elif choice == "7":
            session.nudge_pwm = min(session.max_speed, session.nudge_pwm + 5)
            print(f"Nudge PWM set to: {session.nudge_pwm} PWM")
        elif choice == "8":
            session.nudge_pwm = max(100, session.nudge_pwm - 5)
            print(f"Nudge PWM set to: {session.nudge_pwm} PWM")


def run_live_teleoperation(session):
    """Live interactive WASD drive mode with auto-stop on key release or timeout."""
    print("\n=======================================================")
    print("   🎮 LIVE TELEOPERATION DRIVE MODE                    ")
    print("=======================================================")
    print("Controls:")
    print("  W : Forward (burst 300ms)")
    print("  S : Backward (burst 300ms)")
    print("  A : Pivot Left (burst 200ms)")
    print("  D : Pivot Right (burst 200ms)")
    print("  SPACE : Immediate Full STOP")
    print("  Q : Exit Teleop Mode")
    print("=======================================================\n")

    if not sys.stdin.isatty():
        print("[Notice] Non-interactive stdin detected. Skipping raw teleop.")
        return

    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        active_dir = None
        motion_deadline = 0.0

        while True:
            # Check for input with short timeout for responsive auto-stop
            rlist, _, _ = select.select([fd], [], [], 0.03)
            now = time.time()

            if rlist:
                ch = sys.stdin.read(1)
                if not ch or ch.lower() == "q":
                    break

                # Drain repetitive buffer bursts to prevent accumulated latency
                drain_stop = False
                while select.select([fd], [], [], 0.0)[0]:
                    extra = sys.stdin.read(1)
                    if extra and extra.lower() == "q":
                        ch = "q"
                        break
                    elif extra == " ":
                        ch = " "
                        drain_stop = True
                        break

                if ch.lower() == "q":
                    break
                elif ch == " " or drain_stop:
                    print("\r>>> FULL STOP                   ", end="", flush=True)
                    session.emergency_stop()
                    active_dir = None
                    motion_deadline = 0.0
                    termios.tcflush(fd, termios.TCIFLUSH)
                    continue

                lpwm, rpwm = session.get_drive_pwms(session.base_speed, direction=1)
                tspd = session.turn_speed

                if ch.lower() == "w":
                    if active_dir != "w":
                        print(f"\r>>> FORWARD ({session.base_speed} PWM)    ", end="", flush=True)
                        session.comm.send_drive(lpwm, rpwm)
                        active_dir = "w"
                    motion_deadline = now + 0.22
                elif ch.lower() == "s":
                    blpwm, brpwm = session.get_drive_pwms(session.base_speed, direction=-1)
                    if active_dir != "s":
                        print(f"\r>>> BACKWARD ({session.base_speed} PWM)   ", end="", flush=True)
                        session.comm.send_drive(blpwm, brpwm)
                        active_dir = "s"
                    motion_deadline = now + 0.22
                elif ch.lower() == "a":
                    if active_dir != "a":
                        print(f"\r>>> PIVOT LEFT ({tspd} PWM) ", end="", flush=True)
                        session.comm.send_drive(-tspd, tspd)
                        active_dir = "a"
                    motion_deadline = now + 0.18
                elif ch.lower() == "d":
                    if active_dir != "d":
                        print(f"\r>>> PIVOT RIGHT ({tspd} PWM)", end="", flush=True)
                        session.comm.send_drive(tspd, -tspd)
                        active_dir = "d"
                    motion_deadline = now + 0.18

            # Auto-stop when key is released / deadline expires
            if active_dir is not None and now >= motion_deadline:
                session.emergency_stop()
                active_dir = None
                print("\r>>> IDLE / STOPPED              ", end="", flush=True)

    finally:
        session.emergency_stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\n[Exited Teleop Mode]")


def main():
    print("=======================================================")
    print("   🏎️  ErovoutikaGrab - Motor & Speed Calibration Tool  ")
    print("=======================================================")

    cfg = load_config()
    comm_cfg = cfg.get("communication", {})
    swap = cfg.get("motors", {}).get("swap_left_right", True)

    use_mock = "--mock" in sys.argv
    if use_mock:
        print("[Mode] Running with MockCommAdapter (Simulation)")
        comm = MockCommAdapter(swap_left_right=swap)
    else:
        port = comm_cfg.get("serial_port", "/dev/rfcomm0")
        baud = comm_cfg.get("baud_rate", 9600)
        mac = comm_cfg.get("bluetooth_mac", "20:25:08:00:46:FB")
        channel = comm_cfg.get("rfcomm_channel", 1)
        print(f"[Connecting] Bluetooth {mac} on {port} @ {baud} baud (swap_left_right={swap})...")
        comm = BluetoothSerialAdapter(port=port, baud_rate=baud, bluetooth_mac=mac, rfcomm_channel=channel, swap_left_right=swap)

    if not comm.connect():
        print("\n[Warning] Bluetooth connection failed.")
        print("Note: If 'egrabbot-web.service' is running, it exclusively holds /dev/rfcomm0.")
        print("  - To free the port: sudo systemctl stop egrabbot-web.service")
        print("  - To restart service: sudo systemctl start egrabbot-web.service")
        print("  - Or test offline: python3 scripts/calibrate_motors.py --mock")
        print("[Fallback] Falling back to MockCommAdapter...")
        comm = MockCommAdapter(swap_left_right=swap)
        comm.connect()

    session = MotorCalibrationSession(comm, cfg)

    try:
        while True:
            print("\n" + "=" * 55)
            print("   CURRENT CALIBRATED MOTOR PARAMETERS:")
            print(f"     1. Stiction Floor (min_overcoming_pwm) : {session.min_overcoming_pwm} PWM")
            print(f"     2. Cruising Speed (base_speed)         : {session.base_speed} PWM")
            print(f"     3. Turn Pivot Speed (turn_speed)       : {session.turn_speed} PWM")
            print(f"     4. Micro-Nudge Torque (nudge_pwm)      : {session.nudge_pwm} PWM")
            print(f"     5. Micro-Nudge Duration (nudge_ms)     : {session.nudge_default_ms} ms")
            print(f"     6. Straight-Line Trim Offset           : {session.trim_offset}")
            print(f"     7. Max Speed Ceiling (max_speed)       : {session.max_speed} PWM")
            print("=" * 55)
            print("CALIBRATION ACTIONS:")
            print("  [1] Stiction Floor Ramp Test (Find Minimum Overcoming PWM)")
            print("  [2] Straight-Line Drive & L/R Trim Calibration")
            print("  [3] Pivot Turn Speed Calibration")
            print("  [4] Anti-Stiction Micro-Nudge Pulse Calibration")
            print("  [5] Live Teleoperation Drive Mode (WASD)")
            print("  [S] SAVE Calibrated Values to config/robot_config.yaml")
            print("  [Q] Exit")

            choice = input("\nEnter choice [1-5, S, Q]: ").strip().upper()

            if choice == "Q":
                break
            elif choice == "1":
                run_stiction_ramp_test(session)
            elif choice == "2":
                run_straight_drive_test(session)
            elif choice == "3":
                run_pivot_turn_test(session)
            elif choice == "4":
                run_micro_nudge_test(session)
            elif choice == "5":
                run_live_teleoperation(session)
            elif choice == "S":
                session.save_to_config()
            else:
                print("Invalid selection.")

    finally:
        session.emergency_stop()
        comm.disconnect()
        print("\nDisconnected safely. Motor calibration closed.")


if __name__ == "__main__":
    main()
