"""Unit tests for the autonomous state machine."""

import pytest
from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.adapters.vision.mock_vision import MockVisionAdapter
from src.adapters.vision.tracker_adapter import VisualServoingTracker
from src.core.context import RobotContext
from src.core.state_machine import AutonomousStateMachine, RobotState
from src.manipulation.arm_controller import ArmController
from src.navigation.visual_servoing import VisualServoingController


@pytest.fixture
def autonomous_system():
    context = RobotContext()
    cam = MockCameraAdapter(width=640, height=480)
    comm = MockCommAdapter()
    vision = MockVisionAdapter(mock_bbox=(300, 300, 360, 340))
    tracker = VisualServoingTracker()
    servoing = VisualServoingController(center_x=320, grasp_y=380)
    arm = ArmController(comm)

    sm = AutonomousStateMachine(
        context=context,
        camera=cam,
        comm=comm,
        vision=vision,
        tracker=tracker,
        servoing=servoing,
        arm=arm
    )

    cam.start()
    comm.connect()
    return sm, context, comm, vision


def test_initial_state(autonomous_system):
    sm, context, comm, _ = autonomous_system
    assert sm.state == RobotState.IDLE
    assert context.get_state() == "IDLE"


def test_start_autonomous_run(autonomous_system):
    sm, context, comm, _ = autonomous_system
    sm.start_autonomous_run()
    assert sm.state == RobotState.SEARCHING
    assert any(cmd[0] == "SERVO" for cmd in comm.command_log)


def test_categorization_and_approach_transition(autonomous_system):
    sm, context, comm, vision = autonomous_system
    sm.set_state(RobotState.CATEGORIZING)

    # Execute one step in CATEGORIZING
    sm.step()

    # Vision identified the mock object, initialized tracker, and moved to TRACKING_APPROACH
    assert sm.state == RobotState.TRACKING_APPROACH
    assert context.is_tracking is True
    assert context.get_detection() is not None
    assert context.get_detection().category == "electronic_relay"


def test_emergency_stop(autonomous_system):
    sm, context, comm, _ = autonomous_system
    sm.set_state(RobotState.TRACKING_APPROACH)
    sm.stop()

    assert sm.state == RobotState.IDLE
    assert ("STOP",) in comm.command_log
