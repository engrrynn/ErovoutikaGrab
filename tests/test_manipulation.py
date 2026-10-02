"""Unit tests for ArmController and sequential Servo 1 -> Servo 2 movements."""

import os
import pytest
from src.adapters.comm.mock_comm import MockCommAdapter
from src.manipulation.arm_controller import ArmController


@pytest.fixture
def mock_arm():
    comm = MockCommAdapter()
    config = {
        "servos": {
            "servo1_shoulder": {
                "min_angle": 60,
                "max_angle": 170,
                "stow_angle": 60,
                "down_angle": 110,
                "center_angle": 90,
            },
            "servo2_elbow": {
                "min_angle": 50,
                "max_angle": 110,
                "stow_angle": 70,
                "down_angle": 50,
                "center_angle": 80,
            },
            "servo3_gripper": {
                "close_angle": 20,
                "open_angle": 100,
                "center_angle": 60,
            },
        }
    }
    arm = ArmController(comm, config=config)
    return arm, comm


def test_simultaneous_move_s1_lead_before_simultaneous(mock_arm):
    arm, comm = mock_arm
    comm.command_log.clear()

    # Initial state: cur_s1 = 60, cur_s2 = 70, cur_s3 = 100
    arm.move_arm_simultaneous(s1=110, s2=50, s3=100, wait_for_s1=False)

    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(servo_cmds) > 2

    # Alternating micro-stepping: Going DOWN, S1 extends first, then S2 lowers
    # Sub-step 1: S1 moves to 65 (S2 held at 70)
    assert servo_cmds[0] == ("SERVO", 65, 70, 100)
    # Sub-step 2: S2 moves to 68 (S1 held at 65)
    assert servo_cmds[1] == ("SERVO", 65, 68, 100)
    # Sub-step 3: S1 moves to 70 (S2 held at 68)
    assert servo_cmds[2] == ("SERVO", 70, 68, 100)
    # Sub-step 4: S2 moves to 66 (S1 held at 70)
    assert servo_cmds[3] == ("SERVO", 70, 66, 100)

    # Final command reaches exact target
    assert servo_cmds[-1] == ("SERVO", 110, 50, 100)
    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 50
    assert arm.cur_s3 == 100


def test_stow_moves_s2_lead_before_simultaneous(mock_arm):
    arm, comm = mock_arm
    # Set current position to arm down
    arm.cur_s1 = 110
    arm.cur_s2 = 50
    comm.command_log.clear()

    # Stow: adding angle to S2 (50 -> 70, lifting elbow) and subtracting from S1 (110 -> 60)
    arm.move_arm_simultaneous(s1=60, s2=70, s3=arm.s3_open, wait_for_s1=False)
    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(servo_cmds) > 2

    # Alternating micro-stepping: Going UP, S2 lifts first off floor, then S1 retracts
    # Sub-step 1: S2 lifts to 52 (S1 held at 110)
    assert servo_cmds[0] == ("SERVO", 110, 52, arm.s3_open)
    # Sub-step 2: S1 retracts to 105 (S2 held at 52)
    assert servo_cmds[1] == ("SERVO", 105, 52, arm.s3_open)
    # Sub-step 3: S2 lifts to 54 (S1 held at 105)
    assert servo_cmds[2] == ("SERVO", 105, 54, arm.s3_open)
    # Sub-step 4: S1 retracts to 100 (S2 held at 54)
    assert servo_cmds[3] == ("SERVO", 100, 54, arm.s3_open)

    # Final command reaches exact target
    assert servo_cmds[-1] == ("SERVO", 60, 70, arm.s3_open)
    assert arm.cur_s1 == 60
    assert arm.cur_s2 == 70


def test_pick_sequence_directional_order(mock_arm):
    arm, comm = mock_arm
    # Start at stow
    arm.cur_s1 = arm.s1_stow  # 60
    arm.cur_s2 = arm.s2_stow  # 70
    comm.command_log.clear()

    arm.execute_pick_sequence()
    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]

    # 1. Open gripper is first command
    assert servo_cmds[0] == ("SERVO", arm.s1_stow, arm.s2_stow, arm.s3_open)

    # 2. Lowering arm to ground: S1 leads first
    assert servo_cmds[1] == ("SERVO", arm.s1_stow + 5, arm.s2_stow, arm.s3_open)

    # Find where gripper closes:
    close_idx = -1
    for idx, cmd in enumerate(servo_cmds):
        if cmd[3] == arm.s3_close:
            close_idx = idx
            break
    assert close_idx > 0
    # Arm must have arrived at down position (110, 50) before gripper close
    assert servo_cmds[close_idx - 1] == ("SERVO", arm.s1_down, arm.s2_down, arm.s3_open)
    assert servo_cmds[close_idx] == ("SERVO", arm.s1_down, arm.s2_down, arm.s3_close)

    # 4. Raising arm back up: S2 lifts first off the ground
    assert servo_cmds[close_idx + 1] == ("SERVO", arm.s1_down, arm.s2_down + 2, arm.s3_close)

    # Final position is stowed with gripper closed
    assert servo_cmds[-1] == ("SERVO", arm.s1_stow, arm.s2_stow, arm.s3_close)
    assert arm.cur_s1 == arm.s1_stow
    assert arm.cur_s2 == arm.s2_stow
    assert arm.cur_s3 == arm.s3_close


def test_servo3_moves_freely_during_arm_down(mock_arm):
    arm, comm = mock_arm
    comm.command_log.clear()

    # Move arm down with wait_for_s1=False while explicitly commanding gripper to 45°
    arm.move_arm_simultaneous(arm.s1_down, arm.s2_down, s3=45, wait_for_s1=False)
    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(servo_cmds) > 2

    # Base moves first while gripper waits at initial s3 (100)
    for cmd in servo_cmds[:-1]:
        assert cmd[3] == 100

    # Gripper moves to 45 after base servos reach destination
    assert servo_cmds[-1] == ("SERVO", arm.s1_down, arm.s2_down, 45)
    assert arm.cur_s1 == arm.s1_down
    assert arm.cur_s2 == arm.s2_down
    assert arm.cur_s3 == 45


def test_servo3_move_gripper_in_down_position(mock_arm):
    arm, comm = mock_arm
    arm.move_arm_sequential(arm.s1_down, arm.s2_down, wait_for_s1=False)
    comm.command_log.clear()

    # Move gripper directly to 75° while arm is in down position
    arm.move_gripper(75)
    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(servo_cmds) == 1
    # Arm stays at down position (S1=110, S2=50), gripper moves to 75
    assert servo_cmds[0][1] == arm.s1_down
    assert servo_cmds[0][2] == arm.s2_down
    assert servo_cmds[0][3] == 75
    assert arm.cur_s3 == 75


def test_servo3_open_close_in_down_position(mock_arm):
    arm, comm = mock_arm
    arm.move_arm_sequential(arm.s1_down, arm.s2_down, wait_for_s1=False)
    comm.command_log.clear()

    # Close gripper while arm is down
    arm.close_gripper()
    assert comm.command_log[-1][0] == "SERVO"
    assert comm.command_log[-1][1] == arm.s1_down
    assert comm.command_log[-1][2] == arm.s2_down
    assert comm.command_log[-1][3] == arm.s3_close
    assert arm.cur_s3 == arm.s3_close

    # Open gripper while arm is down
    arm.open_gripper()
    assert comm.command_log[-1][0] == "SERVO"
    assert comm.command_log[-1][1] == arm.s1_down
    assert comm.command_log[-1][2] == arm.s2_down
    assert comm.command_log[-1][3] == arm.s3_open
    assert arm.cur_s3 == arm.s3_open


def test_servo3_clamps_to_physical_limits_not_presets(mock_arm):
    arm, comm = mock_arm
    arm.config["servos"]["servo3_gripper"]["min_angle"] = 10
    arm.config["servos"]["servo3_gripper"]["max_angle"] = 170
    arm.config["servos"]["servo3_gripper"]["close_angle"] = 20
    arm.config["servos"]["servo3_gripper"]["open_angle"] = 100
    arm.reload_config()

    # Moving to 150° should NOT be clamped to open_angle (100°)
    arm.move_gripper(150)
    assert arm.cur_s3 == 150

    # Moving to 15° should NOT be clamped to close_angle (20°)
    arm.move_gripper(15)
    assert arm.cur_s3 == 15

    # Moving beyond physical max (170°) clamps to 170°
    arm.move_gripper(185)
    assert arm.cur_s3 == 170

    # Moving below physical min (10°) clamps to 10°
    arm.move_gripper(5)
    assert arm.cur_s3 == 10


def test_single_joint_incremental_motion(mock_arm):
    arm, comm = mock_arm
    arm.cur_s1 = 90
    arm.cur_s2 = 80
    arm.cur_s3 = 90
    comm.command_log.clear()

    # Move ONLY S1 from 90 to 110 (S2 stays at 80)
    arm.move_arm_simultaneous(110, 80, 90, wait_for_s1=False)
    cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(cmds) == 4
    assert cmds[0] == ("SERVO", 95, 80, 90)
    assert cmds[-1] == ("SERVO", 110, 80, 90)
    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 80

    # Move ONLY S2 from 80 to 60 (S1 stays at 110)
    comm.command_log.clear()
    arm.move_arm_simultaneous(110, 60, 90, wait_for_s1=False)
    cmds2 = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(cmds2) == 4
    assert cmds2[0] == ("SERVO", 110, 75, 90)
    assert cmds2[-1] == ("SERVO", 110, 60, 90)
    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 60


def test_gripper_first_motion(mock_arm):
    arm, comm = mock_arm
    arm.cur_s1 = 60
    arm.cur_s2 = 70
    arm.cur_s3 = 100
    comm.command_log.clear()

    # Move arm down with gripper_first=True
    arm.move_arm_simultaneous(110, 50, s3=20, wait_for_s1=False, gripper_first=True)
    cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(cmds) > 2

    # Command 1: Gripper moves first before any base motion
    assert cmds[0] == ("SERVO", 60, 70, 20)

    # All subsequent commands maintain gripper at 20 while base moves
    for cmd in cmds[1:]:
        assert cmd[3] == 20

    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 50
    assert arm.cur_s3 == 20


def test_move_arm_alternate_alias(mock_arm):
    arm, comm = mock_arm
    arm.cur_s1 = 60
    arm.cur_s2 = 70
    arm.cur_s3 = 100
    comm.command_log.clear()

    # Calling alias move_arm_alternate redirects to simultaneous increment with directional priority
    arm.move_arm_alternate(110, 50, 100, wait_for_s1=False)
    cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(cmds) > 2
    assert cmds[0] == ("SERVO", 65, 70, 100)
    assert cmds[1] == ("SERVO", 65, 68, 100)
    assert cmds[-1] == ("SERVO", 110, 50, 100)
    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 50


def test_force_sequential_option(mock_arm):
    arm, comm = mock_arm
    arm.cur_s1 = 60
    arm.cur_s2 = 70
    arm.cur_s3 = 100
    comm.command_log.clear()

    # Explicit force_sequential=True executes two distinct full-travel commands
    arm.move_arm_sequential(110, 50, 100, wait_for_s1=False, force_sequential=True)
    cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(cmds) == 2
    assert cmds[0] == ("SERVO", 110, 70, 100)
    assert cmds[1] == ("SERVO", 110, 50, 100)
    assert arm.cur_s1 == 110
    assert arm.cur_s2 == 50


def test_stow_preserves_gripper_grip(mock_arm):
    """Ensure stow does not reset or open the gripper when holding an object."""
    arm, comm = mock_arm
    # Simulate gripping an object while in down reach position
    arm.cur_s1 = arm.s1_down
    arm.cur_s2 = arm.s2_down
    arm.close_gripper()
    assert arm.cur_s3 == arm.s3_close

    comm.command_log.clear()
    # Call stow() without arguments
    arm.stow()

    servo_cmds = [cmd for cmd in comm.command_log if cmd[0] == "SERVO"]
    assert len(servo_cmds) > 0
    # Every intermediate micro-increment and final position must preserve the gripped angle!
    for cmd in servo_cmds:
        assert cmd[3] == arm.s3_close, f"Gripper was unexpectedly modified to {cmd[3]} during stow!"

    assert arm.cur_s1 == arm.s1_stow
    assert arm.cur_s2 == arm.s2_stow
    assert arm.cur_s3 == arm.s3_close


def test_close_gripper_prevents_stall(mock_arm):
    """Ensure close_gripper clamps angle to safe physical bounds to avoid motor stall."""
    arm, comm = mock_arm
    arm.config["servos"]["servo3_gripper"]["min_angle"] = 60
    arm.config["servos"]["servo3_gripper"]["close_angle"] = 70
    arm.reload_config()

    # Commanding an angle below minimum safe bound is clamped
    arm.close_gripper(angle=25)
    assert arm.cur_s3 == 60


def test_arm_controller_persists_state(tmp_path, mock_arm):
    """Ensure ArmController persists assigned angles across process/service resets."""
    import json
    arm, comm = mock_arm
    state_file = str(tmp_path / "arm_state.json")
    arm.state_path = state_file

    # Command arm and gripper to specific assigned angles
    arm.move_arm_alternate(100, 65, 75, wait_for_s1=False)
    assert arm.cur_s1 == 100
    assert arm.cur_s2 == 65
    assert arm.cur_s3 == 75

    assert os.path.exists(state_file)
    with open(state_file) as f:
        data = json.load(f)
        assert data["s1"] == 100
        assert data["s2"] == 65
        assert data["s3"] == 75

    # Simulate process reset: instantiate a brand new ArmController with the persisted state
    new_arm = ArmController(comm, config=arm.config, state_path=state_file)
    assert new_arm.cur_s1 == 100
    assert new_arm.cur_s2 == 65
    assert new_arm.cur_s3 == 75







