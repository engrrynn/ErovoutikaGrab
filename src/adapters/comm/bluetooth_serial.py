"""Bluetooth / Serial communication adapter for Arduino Nano."""

import json
import os
import subprocess
import threading
import time
from typing import Optional
import serial

from src.adapters.comm.base import BaseCommAdapter
from src.adapters.comm.protocol import ArduinoProtocol
from src.core.events import TelemetryData


class BluetoothSerialAdapter(BaseCommAdapter):
    """Handles communication with the Arduino via Bluetooth SPP (RFCOMM) or USB Serial."""

    def __init__(
        self,
        port: str = "/dev/rfcomm0",
        baud_rate: int = 9600,
        bluetooth_mac: str = "20:25:08:00:46:FB",
        rfcomm_channel: int = 1,
        auto_reconnect: bool = True,
        swap_left_right: Optional[bool] = None
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.bluetooth_mac = bluetooth_mac
        self.rfcomm_channel = rfcomm_channel
        self.auto_reconnect = auto_reconnect

        if swap_left_right is None:
            cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config/robot_config.yaml"))
            if os.path.exists(cfg_path):
                try:
                    import yaml
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    swap_left_right = bool(cfg.get("motors", {}).get("swap_left_right", False))
                except Exception:
                    swap_left_right = False
            else:
                swap_left_right = False
        self.swap_left_right = swap_left_right

        self._serial: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_interval_s: float = 0.10
        self._pulse_duration_s: float = 0.02
        self._is_pulsing: bool = False
        self._state_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/arm_state.json"))

        self._telemetry = TelemetryData()
        self._last_ping_sent: float = 0.0
        self._ping_ms: float = 0.0
        self._last_drive_cmd: Optional[tuple] = None
        self._last_drive_time: float = 0.0
        self._last_assigned_servos: Optional[tuple] = None
        self._load_persisted_servos()

    def _load_persisted_servos(self):
        """Loads assigned servo angles from persistent disk storage upon initialization."""
        if hasattr(self, "_state_path") and os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                    s1 = data.get("s1")
                    s2 = data.get("s2")
                    s3 = data.get("s3")
                    if s1 is not None and s2 is not None and s3 is not None:
                        self._last_assigned_servos = (int(s1), int(s2), int(s3))
            except Exception:
                pass

    def _ensure_rfcomm_binding(self, force_rebind: bool = False) -> bool:
        """Checks if rfcomm device exists and is clean; if closed or missing, releases and rebinds."""
        if not self.bluetooth_mac:
            return os.path.exists(self.port)

        dev_num = self.port.replace("/dev/rfcomm", "")
        if not dev_num.isdigit():
            dev_num = "0"

        try:
            out = subprocess.check_output(["rfcomm", "-a"], stderr=subprocess.STDOUT).decode()
            is_closed = "closed" in out and self.bluetooth_mac.lower() in out.lower()
            if is_closed or force_rebind:
                print(f"[BluetoothComm] RFCOMM device {self.port} is closed/stale. Releasing and rebinding...")
                subprocess.run(["sudo", "rfcomm", "release", dev_num], check=False, timeout=2.0)
                time.sleep(0.2)
                subprocess.run(["sudo", "rfcomm", "bind", dev_num, self.bluetooth_mac, str(self.rfcomm_channel)], check=False, timeout=2.0)
                time.sleep(0.3)
                return os.path.exists(self.port)
            elif self.bluetooth_mac.lower() in out.lower() and os.path.exists(self.port):
                return True
            else:
                subprocess.run(["sudo", "rfcomm", "bind", dev_num, self.bluetooth_mac, str(self.rfcomm_channel)], check=False, timeout=2.0)
                time.sleep(0.3)
                return os.path.exists(self.port)
        except Exception as e:
            print(f"[BluetoothComm] Warning: RFCOMM binding check error: {e}")
        return os.path.exists(self.port)

    def connect(self) -> bool:
        """Open serial port and start background reader thread."""
        with self._lock:
            if self._serial is not None and self._serial.is_open and self._telemetry.connected:
                return True

            # If there's an existing open serial or reader thread, close it first
            self._running = False
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

            self._ensure_rfcomm_binding()

            try:
                print(f"[BluetoothComm] Connecting to {self.port} at {self.baud_rate} baud...")
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=0.1,
                    write_timeout=0.5
                )
                time.sleep(1.0)  # Wait for Arduino bootloader / Bluetooth handshake
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()

                self._running = True
                self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
                self._reader_thread.start()
                self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
                self._watchdog_thread.start()

                # Verify connection with handshake/ping
                self.ping()
                time.sleep(0.3)

                # Check if port dropped immediately due to host being down/unreachable
                if not self._running or self._serial is None:
                    raise serial.SerialException("Bluetooth RFCOMM link dropped immediately (host unreachable or unpowered)")

                if self._last_assigned_servos:
                    s1, s2, s3 = self._last_assigned_servos
                    self._send_raw(ArduinoProtocol.encode_servos(s1, s2, s3))
                print(f"[BluetoothComm] Successfully connected to {self.port}")
                return True
            except Exception as e:
                print(f"[BluetoothComm] Connection failed: {e}")
                self._running = False
                if self._serial is not None:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                self._serial = None
                self._telemetry.connected = False
                # Rebind RFCOMM so port is left in clean state for next retry
                self._ensure_rfcomm_binding(force_rebind=True)
                return False

    def disconnect(self):
        """Cleanly close connection."""
        self._running = False
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._send_raw(ArduinoProtocol.encode_stop())
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
        self._telemetry.connected = False

    def is_connected(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open and self._telemetry.connected

    def _send_raw(self, data: bytes) -> bool:
        with self._lock:
            if not (self._serial and self._serial.is_open):
                return False
            try:
                self._serial.write(data)
                self._serial.flush()
                return True
            except Exception as e:
                print(f"[BluetoothComm] Send error: {e}")
                self._telemetry.connected = False
                return False

    def send_drive(self, left_pwm: int, right_pwm: int) -> bool:
        now = time.time()
        # Rate-limit identical redundant drive commands within 40ms to avoid 9600-baud buffer bloat
        if self._last_drive_cmd == (left_pwm, right_pwm) and (now - self._last_drive_time) < 0.04:
            return True
        self._last_drive_cmd = (left_pwm, right_pwm)
        self._last_drive_time = now

        if self.swap_left_right:
            return self._send_raw(ArduinoProtocol.encode_drive(right_pwm, left_pwm))
        return self._send_raw(ArduinoProtocol.encode_drive(left_pwm, right_pwm))

    def send_stop(self) -> bool:
        self._last_drive_cmd = (0, 0)
        self._last_drive_time = time.time()
        return self._send_raw(ArduinoProtocol.encode_stop())

    def send_nudge(self, direction: str, duration_ms: int = 70, pwm: int = 210) -> bool:
        dir_cmd = direction.upper()
        if self.swap_left_right:
            if dir_cmd == "L":
                dir_cmd = "R"
            elif dir_cmd == "R":
                dir_cmd = "L"
        return self._send_raw(ArduinoProtocol.encode_nudge(dir_cmd, duration_ms, pwm))

    def send_servos(self, s1: int, s2: int, s3: int, pulse: Optional[bool] = None) -> bool:
        prev_servos = self._last_assigned_servos
        prev_s3 = prev_servos[2] if prev_servos else None

        do_pulse = False
        if pulse is True:
            do_pulse = True
        elif pulse is None:
            if prev_s3 is not None and prev_s3 != s3:
                do_pulse = True

        self._last_assigned_servos = (s1, s2, s3)

        if getattr(self, "_state_path", None):
            try:
                os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
                with open(self._state_path, "w") as f:
                    json.dump({"s1": s1, "s2": s2, "s3": s3}, f)
            except Exception:
                pass

        if do_pulse and not getattr(self, "_is_pulsing", False):
            self._is_pulsing = True
            try:
                pulse_s3 = s3 + 1 if s3 < 180 else s3 - 1
                p1 = self._send_raw(ArduinoProtocol.encode_servos(s1, s2, pulse_s3))
                time.sleep(getattr(self, "_pulse_duration_s", 0.02))
                p2 = self._send_raw(ArduinoProtocol.encode_servos(s1, s2, s3))
                return p1 and p2
            finally:
                self._is_pulsing = False
        else:
            return self._send_raw(ArduinoProtocol.encode_servos(s1, s2, s3))

    def _watchdog_loop(self):
        """Continuously persists and pushes assigned servo angles with 100ms gripper pulse in loop."""
        while self._running:
            try:
                interval = getattr(self, "_watchdog_interval_s", 0.10)
                time.sleep(interval)
                if not self._running:
                    break
                # If motors are actively driving, yield 9600-baud serial bus immediately to drive commands
                if self._last_drive_cmd and self._last_drive_cmd != (0, 0):
                    continue
                if self.is_connected() and self._last_assigned_servos and not getattr(self, "_is_pulsing", False):
                    s1, s2, s3 = self._last_assigned_servos
                    pulse_s3 = s3 + 1 if s3 < 180 else s3 - 1
                    # 100ms pulse in loop strictly for gripper servo (never pulse S1 or S2)
                    self._send_raw(ArduinoProtocol.encode_servos(s1, s2, pulse_s3))
                    pulse_dur = getattr(self, "_pulse_duration_s", 0.02)
                    time.sleep(pulse_dur)
                    if not self._running:
                        break
                    # Decrement back to assigned angle
                    s1, s2, s3 = self._last_assigned_servos
                    self._send_raw(ArduinoProtocol.encode_servos(s1, s2, s3))
            except Exception as e:
                if self._running:
                    print(f"[BluetoothComm] Watchdog loop warning: {e}")

    def send_macro(self, name: str) -> bool:
        return self._send_raw(ArduinoProtocol.encode_macro(name))

    def ping(self) -> float:
        self._last_ping_sent = time.time()
        if self._send_raw(ArduinoProtocol.encode_ping()):
            return self._ping_ms
        return -1.0

    def get_latest_telemetry(self) -> TelemetryData:
        with self._lock:
            return self._telemetry

    def _read_loop(self):
        """Background thread reading packets from Arduino."""
        line_buffer = ""
        while self._running:
            try:
                if not (self._serial and self._serial.is_open):
                    time.sleep(0.2)
                    continue

                raw = self._serial.read(self._serial.in_waiting or 1)
                if not raw:
                    continue

                line_buffer += raw.decode("ascii", errors="ignore")
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    parsed = ArduinoProtocol.parse_line(line)
                    if parsed:
                        try:
                            self._handle_incoming_packet(parsed)
                        except Exception as p_err:
                            print(f"[BluetoothComm] Packet handling warning: {p_err}")

            except Exception as e:
                if self._running:
                    print(f"[BluetoothComm] Read loop error: {e}")
                    self._telemetry.connected = False
                    with self._lock:
                        if self._serial:
                            try:
                                self._serial.close()
                            except Exception:
                                pass
                            self._serial = None
                    self._running = False
                    break

    def _handle_incoming_packet(self, packet: dict):
        with self._lock:
            self._telemetry.connected = True
            pkt_type = packet.get("type")

            if pkt_type == "STATUS":
                lpwm = packet.get("left_pwm")
                rpwm = packet.get("right_pwm")
                if lpwm is None or rpwm is None:
                    return
                if self.swap_left_right:
                    self._telemetry.left_pwm = rpwm
                    self._telemetry.right_pwm = lpwm
                else:
                    self._telemetry.left_pwm = lpwm
                    self._telemetry.right_pwm = rpwm
                self._telemetry.s1_shoulder = packet.get("s1", self._telemetry.s1_shoulder)
                self._telemetry.s2_elbow = packet.get("s2", self._telemetry.s2_elbow)
                self._telemetry.s3_gripper = packet.get("s3", self._telemetry.s3_gripper)
                self._telemetry.macro_active = packet.get("macro_active", False)
                self._telemetry.timestamp = time.time()
            elif pkt_type == "PONG":
                if self._last_ping_sent > 0:
                    self._ping_ms = round((time.time() - self._last_ping_sent) * 1000.0, 1)
                    self._telemetry.ping_ms = self._ping_ms
            elif pkt_type == "READY":
                # Arduino experienced a hardware/software reset or boot (<READY:ErovoutikaGrab_v2.0>)
                if self._last_assigned_servos:
                    s1, s2, s3 = self._last_assigned_servos
                    print(f"[BluetoothComm] Arduino reset event detected (<READY>). Re-asserting assigned servo angles: ({s1}, {s2}, {s3})")
                    self._send_raw(ArduinoProtocol.encode_servos(s1, s2, s3))
            elif pkt_type == "ACK":
                pass
