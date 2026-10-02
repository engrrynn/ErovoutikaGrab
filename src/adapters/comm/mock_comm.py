"""Mock communication adapter for offline simulation and automated tests."""

import time
from typing import Optional
from src.adapters.comm.base import BaseCommAdapter
from src.core.events import TelemetryData


class MockCommAdapter(BaseCommAdapter):
    """Simulated Arduino robot controller for testing without hardware."""

    def __init__(self, swap_left_right: bool = False):
        self.swap_left_right = swap_left_right
        self._connected = False
        self._telemetry = TelemetryData(
            left_pwm=0,
            right_pwm=0,
            s1_shoulder=20,
            s2_elbow=70,
            s3_gripper=170,
            macro_active=False,
            connected=True,
            ping_ms=1.5
        )
        self.command_log = []

    def connect(self) -> bool:
        self._connected = True
        self._telemetry.connected = True
        return True

    def disconnect(self):
        self._connected = False
        self._telemetry.connected = False

    def is_connected(self) -> bool:
        return self._connected

    def send_drive(self, left_pwm: int, right_pwm: int) -> bool:
        if self.swap_left_right:
            self.command_log.append(("DRIVE", right_pwm, left_pwm))
            self._telemetry.left_pwm = left_pwm
            self._telemetry.right_pwm = right_pwm
        else:
            self.command_log.append(("DRIVE", left_pwm, right_pwm))
            self._telemetry.left_pwm = left_pwm
            self._telemetry.right_pwm = right_pwm
        return True

    def send_stop(self) -> bool:
        self.command_log.append(("STOP",))
        self._telemetry.left_pwm = 0
        self._telemetry.right_pwm = 0
        return True

    def send_nudge(self, direction: str, duration_ms: int = 70, pwm: int = 210) -> bool:
        dir_cmd = direction.upper()
        if self.swap_left_right:
            if dir_cmd == "L":
                dir_cmd = "R"
            elif dir_cmd == "R":
                dir_cmd = "L"
        self.command_log.append(("NUDGE", dir_cmd, duration_ms, pwm))
        return True

    def send_servos(self, s1: int, s2: int, s3: int, pulse: Optional[bool] = None, **kwargs) -> bool:
        self.command_log.append(("SERVO", s1, s2, s3))
        self._telemetry.s1_shoulder = s1
        self._telemetry.s2_elbow = s2
        self._telemetry.s3_gripper = s3
        return True

    def send_macro(self, name: str) -> bool:
        self.command_log.append(("MACRO", name))
        if name == "PICK":
            self._telemetry.macro_active = True
        elif name == "DEFAULT":
            self._telemetry.s1_shoulder = 20
            self._telemetry.s2_elbow = 70
            self._telemetry.s3_gripper = 170
        return True

    def ping(self) -> float:
        return 1.2

    def get_latest_telemetry(self) -> TelemetryData:
        return self._telemetry
