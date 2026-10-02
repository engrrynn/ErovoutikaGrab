"""Unit tests for Motor and Speed Calibration session and calculations."""

import pytest
import os
import yaml
from src.adapters.comm.mock_comm import MockCommAdapter
from scripts.calibrate_motors import MotorCalibrationSession


@pytest.fixture
def mock_config():
    return {
        "motors": {
            "base_speed": 210,
            "turn_speed": 200,
            "min_overcoming_pwm": 190,
            "nudge_default_ms": 70,
            "nudge_pwm": 210,
            "max_speed": 255,
            "trim_offset": 0
        }
    }


def test_motor_session_init(mock_config):
    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    assert session.base_speed == 210
    assert session.turn_speed == 200
    assert session.min_overcoming_pwm == 190
    assert session.nudge_pwm == 210
    assert session.nudge_default_ms == 70
    assert session.trim_offset == 0


def test_get_drive_pwms_no_trim(mock_config):
    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    # Forward
    l, r = session.get_drive_pwms(210, direction=1)
    assert l == 210
    assert r == 210

    # Backward
    l, r = session.get_drive_pwms(210, direction=-1)
    assert l == -210
    assert r == -210


def test_get_drive_pwms_with_trim(mock_config):
    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    # Positive trim offset (veers left, so compensate by reducing right wheel)
    session.trim_offset = 10
    l, r = session.get_drive_pwms(210, direction=1)
    assert l == 210
    assert r == 200

    # Negative trim offset (veers right, so compensate by reducing left wheel)
    session.trim_offset = -15
    l, r = session.get_drive_pwms(210, direction=1)
    assert l == 195
    assert r == 210


def test_timed_drive_and_stop(mock_config):
    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    session.timed_drive(210, 210, duration_sec=0.01)
    assert len(comm.command_log) == 2
    assert comm.command_log[0] == ("DRIVE", 210, 210)
    assert comm.command_log[1] == ("STOP",)


def test_emergency_stop(mock_config):
    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    session.emergency_stop()
    assert comm.command_log[-1] == ("STOP",)


def test_save_to_config_sync(mock_config, tmp_path, monkeypatch):
    test_file = tmp_path / "test_robot_config.yaml"
    monkeypatch.setattr("scripts.calibrate_motors.CONFIG_PATH", str(test_file))

    comm = MockCommAdapter()
    session = MotorCalibrationSession(comm, mock_config)

    session.base_speed = 225
    session.turn_speed = 205
    session.min_overcoming_pwm = 195
    session.nudge_pwm = 220
    session.nudge_default_ms = 80
    session.trim_offset = 6

    # Test in-memory config update and writing to isolated test_file
    session.save_to_config()
    m = session.config["motors"]
    assert m["base_speed"] == 225
    assert m["turn_speed"] == 205
    assert m["min_overcoming_pwm"] == 195
    assert m["nudge_pwm"] == 220
    assert m["nudge_default_ms"] == 80
    assert m["trim_offset"] == 6

    # Verify the isolated file was written without touching production config
    with open(test_file, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["motors"]["base_speed"] == 225
