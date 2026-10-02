"""Comprehensive unit tests for motor left/right swap and universal configuration application."""

import os
import tempfile
import yaml
import pytest

from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.core.context import RobotContext
from src.core.events import DetectionResult
from src.core.state_machine import AutonomousStateMachine, RobotState
from src.manipulation.arm_controller import ArmController
from src.navigation.visual_servoing import VisualServoingController


class DummySerial:
    """Mock serial stream to inspect raw written bytes."""
    def __init__(self):
        self.written_bytes = []
        self.is_open = True
        self.in_waiting = 0

    def write(self, data: bytes):
        self.written_bytes.append(data)

    def flush(self):
        pass

    def read(self, size: int = 1):
        return b""


def test_bluetooth_adapter_swap_motors_drive():
    """Verify that send_drive swaps arguments when swap_left_right is True."""
    adapter = BluetoothSerialAdapter(swap_left_right=True)
    dummy = DummySerial()
    adapter._serial = dummy

    # Request left=120, right=240
    adapter.send_drive(120, 240)

    # Physical Arduino should receive DRIVE:240,120 because Motor A is Right, Motor B is Left
    assert len(dummy.written_bytes) == 1
    assert dummy.written_bytes[0] == b"<DRIVE:240,120>\n"

    # Test without swap
    adapter_no_swap = BluetoothSerialAdapter(swap_left_right=False)
    dummy_no_swap = DummySerial()
    adapter_no_swap._serial = dummy_no_swap
    adapter_no_swap.send_drive(120, 240)
    assert dummy_no_swap.written_bytes[0] == b"<DRIVE:120,240>\n"


def test_bluetooth_adapter_swap_motors_nudge():
    """Verify that send_nudge transposes L <-> R when swap_left_right is True."""
    adapter = BluetoothSerialAdapter(swap_left_right=True)
    dummy = DummySerial()
    adapter._serial = dummy

    adapter.send_nudge("L", duration_ms=250, pwm=245)
    assert dummy.written_bytes[0] == b"<NUDGE:R,250,245>\n"

    adapter.send_nudge("R", duration_ms=250, pwm=245)
    assert dummy.written_bytes[1] == b"<NUDGE:L,250,245>\n"

    # Forward and Backward should remain unchanged
    adapter.send_nudge("F", duration_ms=250, pwm=245)
    assert dummy.written_bytes[2] == b"<NUDGE:F,250,245>\n"


def test_bluetooth_adapter_telemetry_status_swap():
    """Verify telemetry STATUS packets swap left/right reported PWMs."""
    adapter = BluetoothSerialAdapter(swap_left_right=True)
    packet = {
        "type": "STATUS",
        "left_pwm": 150,  # Arduino Motor A (physically right)
        "right_pwm": 250, # Arduino Motor B (physically left)
        "s1": 90,
        "s2": 80,
        "s3": 90,
        "macro_active": False
    }
    adapter._handle_incoming_packet(packet)
    telem = adapter.get_latest_telemetry()
    assert telem.left_pwm == 250
    assert telem.right_pwm == 150


def test_mock_comm_swap_support():
    """Verify MockCommAdapter correctly handles swap_left_right."""
    mock = MockCommAdapter(swap_left_right=True)
    mock.send_drive(100, 200)
    assert mock.command_log[0] == ("DRIVE", 200, 100)

    mock.send_nudge("L", duration_ms=100, pwm=200)
    assert mock.command_log[1] == ("NUDGE", "R", 100, 200)


def test_arm_controller_calibrated_angle_bounds(tmp_path):
    """Verify ArmController does NOT artificially clamp down_angle > max_angle."""
    cfg_file = tmp_path / "robot_config.yaml"
    cfg = {
        "servos": {
            "servo1_shoulder": {
                "min_angle": 60,
                "max_angle": 135,  # Stale limit
                "stow_angle": 60,
                "down_angle": 180,  # Calibrated down reach
                "center_angle": 90
            },
            "servo2_elbow": {
                "min_angle": 0,
                "max_angle": 80,
                "stow_angle": 80,
                "down_angle": 0,
                "center_angle": 80
            },
            "servo3_gripper": {
                "close_angle": 70,
                "open_angle": 110,
                "center_angle": 90
            }
        }
    }
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)

    comm = MockCommAdapter()
    arm = ArmController(comm, config=cfg)
    arm.config_path = str(cfg_file)
    arm.reload_config()

    # s1_down must reach 180 without being truncated to 135
    assert arm.s1_down == 180
    assert arm.s1_max >= 180
    assert arm.s2_down == 0
    assert arm.s2_stow == 80
    assert arm.s3_close == 70
    assert arm.s3_open == 110

    # Moving to down reach must send 180
    arm.arm_down()
    assert comm.command_log[-1] == ("SERVO", 180, 0, 110)


def test_visual_servoing_trim_offset():
    """Verify VisualServoingController applies trim_offset to forward drive."""
    # Trim offset +10 (veers left -> reduce right motor by 10)
    controller = VisualServoingController(
        center_x=320,
        grasp_y=380,
        base_speed=200,
        trim_offset=10
    )
    # Target directly in line horizontally (ex = 0), but far in front (ey > tol_y)
    action = controller.compute_action((320, 200))
    assert action.action_type == "DRIVE"
    assert action.left_pwm == 200
    assert action.right_pwm == 190

    # Trim offset -10 (veers right -> reduce left motor by 10)
    controller.trim_offset = -10
    action2 = controller.compute_action((320, 200))
    assert action2.action_type == "DRIVE"
    assert action2.left_pwm == 190
    assert action2.right_pwm == 200


def test_visual_servoing_reload_config():
    """Verify VisualServoingController updates parameters via reload_config."""
    controller = VisualServoingController(base_speed=200, turn_speed=180)
    new_motors_cfg = {
        "base_speed": 230,
        "turn_speed": 245,
        "trim_offset": 8,
        "nudge_pwm": 250,
        "nudge_default_ms": 220
    }
    controller.reload_config(new_motors_cfg)
    assert controller.base_speed == 230
    assert controller.turn_speed == 245
    assert controller.trim_offset == 8
    assert controller.nudge_pwm == 250
    assert controller.nudge_duration_ms == 220


def test_bluetooth_rate_limiting_and_immediate_stop():
    """Verify BluetoothSerialAdapter suppresses redundant drive packets within 40ms but stops immediately."""
    adapter = BluetoothSerialAdapter(swap_left_right=False)
    dummy = DummySerial()
    adapter._serial = dummy

    # 1. First DRIVE sent
    adapter.send_drive(200, 200)
    assert len(dummy.written_bytes) == 1
    assert dummy.written_bytes[0] == b"<DRIVE:200,200>\n"

    # 2. Immediate duplicate DRIVE (<40ms) suppressed
    adapter.send_drive(200, 200)
    assert len(dummy.written_bytes) == 1

    # 3. send_stop must NEVER be suppressed
    adapter.send_stop()
    assert len(dummy.written_bytes) == 2
    assert dummy.written_bytes[1] == b"<STOP>\n"

    # 4. Immediate DRIVE after STOP must be sent immediately
    adapter.send_drive(200, 200)
    assert len(dummy.written_bytes) == 3
    assert dummy.written_bytes[2] == b"<DRIVE:200,200>\n"


def test_hud_dashboard_key_press_and_release():
    """Verify HUDDashboard stops immediately on keyReleaseEvent and focusOutEvent."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QKeyEvent, QFocusEvent
    from PyQt5.QtCore import QEvent, Qt
    from src.adapters.camera.mock_camera import MockCameraAdapter
    from src.adapters.vision.tracker_adapter import VisualServoingTracker
    from src.ui.hud_dashboard import HUDDashboard

    app = QApplication.instance() or QApplication([])
    ctx = RobotContext()
    comm = MockCommAdapter()
    cam = MockCameraAdapter()
    tracker = VisualServoingTracker()
    servoing = VisualServoingController(320, 380)
    arm = ArmController(comm)
    sm = AutonomousStateMachine(ctx, cam, comm, None, tracker, servoing, arm)

    dashboard = HUDDashboard(ctx, sm)

    # 1. Press W -> DRIVE sent
    ev_press = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
    dashboard.keyPressEvent(ev_press)
    assert len(comm.command_log) > 0
    assert comm.command_log[-1][0] == "DRIVE"

    # 2. Auto-repeat Press W -> Ignored
    ev_repeat = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w", True, 1)
    len_before = len(comm.command_log)
    dashboard.keyPressEvent(ev_repeat)
    assert len(comm.command_log) == len_before

    # 3. Release W -> STOP sent immediately
    ev_release = QKeyEvent(QEvent.KeyRelease, Qt.Key_W, Qt.NoModifier)
    dashboard.keyReleaseEvent(ev_release)
    assert comm.command_log[-1] == ("STOP",)

    # 4. Focus out -> STOP sent immediately
    ev_press_s = QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.NoModifier)
    dashboard.keyPressEvent(ev_press_s)
    assert comm.command_log[-1][0] == "DRIVE"
    dashboard.focusOutEvent(QFocusEvent(QEvent.FocusOut))
    assert comm.command_log[-1] == ("STOP",)
