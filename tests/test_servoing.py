"""Unit tests for Visual Servoing controller, anti-stiction nudges, and tolerances."""

import pytest
from src.navigation.visual_servoing import VisualServoingController


@pytest.fixture
def controller():
    return VisualServoingController(
        center_x=320,
        grasp_y=380,
        tol_x=20,
        tol_y=25,
        base_speed=210,
        turn_speed=200,
        min_overcoming_pwm=190,
        pulse_threshold_px=50,
        nudge_duration_ms=70,
        nudge_pwm=210
    )


def test_aligned_condition(controller):
    # Target directly inside sweet spot tolerances
    action = controller.compute_action((325, 375))
    assert action.is_aligned is True
    assert action.action_type == "ALIGNED"


def test_far_turn_right(controller):
    # Target way to the right (> pulse_threshold_px)
    action = controller.compute_action((420, 250))
    assert action.is_aligned is False
    assert action.action_type == "DRIVE"
    assert action.left_pwm > 0
    assert action.right_pwm < 0


def test_fine_nudge_turn_left(controller):
    # Target slightly to the left (within pulse_threshold_px but outside tol_x)
    # ex = 285 - 320 = -35. abs(-35) > 20 and <= 50 -> should nudge left!
    action = controller.compute_action((285, 250))
    assert action.is_aligned is False
    assert action.action_type == "NUDGE"
    assert action.direction == "L"
    assert action.duration_ms == 70
    assert action.pwm == 210


def test_forward_approach(controller):
    # Centered horizontally, but far away vertically
    # ex = 0, ey = 380 - 150 = 230 (> 50)
    action = controller.compute_action((320, 150))
    assert action.is_aligned is False
    assert action.action_type == "DRIVE"
    assert action.left_pwm == 210
    assert action.right_pwm == 210


def test_fine_forward_nudge(controller):
    # Centered horizontally, close to grasp line (ey = 380 - 345 = 35 <= 50)
    action = controller.compute_action((320, 345))
    assert action.is_aligned is False
    assert action.action_type == "NUDGE"
    assert action.direction == "F"


def test_stop_on_none_target(controller):
    action = controller.compute_action(None)
    assert action.action_type == "STOP"
