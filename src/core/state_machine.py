"""Autonomous State Machine for ErovoutikaGrab."""

import enum
import time
from typing import Optional
import cv2
import numpy as np

from src.adapters.camera.base import BaseCameraAdapter
from src.adapters.comm.base import BaseCommAdapter
from src.adapters.vision.base import BaseVisionAdapter
from src.adapters.vision.tracker_adapter import VisualServoingTracker
from src.core.context import RobotContext
from src.core.events import DetectionResult
from src.manipulation.arm_controller import ArmController
from src.navigation.visual_servoing import VisualServoingController


class RobotState(enum.Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    CATEGORIZING = "CATEGORIZING"
    TRACKING_APPROACH = "TRACKING_APPROACH"
    FINE_ALIGNMENT = "FINE_ALIGNMENT"
    PRE_GRASP_VERIFY = "PRE_GRASP_VERIFY"
    PICKING = "PICKING"
    POST_GRASP_VERIFY = "POST_GRASP_VERIFY"
    OBJECT_SECURED = "OBJECT_SECURED"
    ERROR = "ERROR"


class AutonomousStateMachine:
    """Manages autonomous object detection, visual servoing, and picking pipeline."""

    def __init__(
        self,
        context: RobotContext,
        camera: BaseCameraAdapter,
        comm: BaseCommAdapter,
        vision: BaseVisionAdapter,
        tracker: VisualServoingTracker,
        servoing: VisualServoingController,
        arm: ArmController
    ):
        self.context = context
        self.camera = camera
        self.comm = comm
        self.vision = vision
        self.tracker = tracker
        self.servoing = servoing
        self.arm = arm

        self.state = RobotState.IDLE
        # Search cycle timing loaded from calibrated robot config
        motors_cfg = {}
        if hasattr(self.context, "config") and isinstance(self.context.config, dict):
            motors_cfg = self.context.config.get("robot", {}).get("motors", {})
        self._search_turn_step_ms = int(motors_cfg.get("nudge_default_ms", 80))
        self._search_nudge_pwm = int(motors_cfg.get("nudge_pwm", motors_cfg.get("min_overcoming_pwm", 210)))
        self._last_search_turn = 0.0
        self._search_pause_ms = 800

        # Alignment stability counter
        self._aligned_count = 0
        self._required_aligned_frames = 5

    def set_state(self, new_state: RobotState):
        """Transition to a new state."""
        print(f"[StateMachine] Transition: {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.state_enter_time = time.time()
        self.context.set_state(new_state.value)

    def step(self):
        """Execute one cycle of the state machine."""
        frame = self.camera.get_frame()
        if frame is not None:
            self.context.update_frame(frame)

        if self.state == RobotState.IDLE:
            self._handle_idle()
        elif self.state == RobotState.SEARCHING:
            self._handle_searching(frame)
        elif self.state == RobotState.CATEGORIZING:
            self._handle_categorizing(frame)
        elif self.state == RobotState.TRACKING_APPROACH:
            self._handle_tracking_approach(frame)
        elif self.state == RobotState.FINE_ALIGNMENT:
            self._handle_fine_alignment(frame)
        elif self.state == RobotState.PRE_GRASP_VERIFY:
            self._handle_pre_grasp_verify(frame)
        elif self.state == RobotState.PICKING:
            self._handle_picking()
        elif self.state == RobotState.POST_GRASP_VERIFY:
            self._handle_post_grasp_verify(frame)
        elif self.state == RobotState.OBJECT_SECURED:
            self._handle_object_secured()

    def start_autonomous_run(self):
        """Trigger start of autonomous search and grab sequence."""
        self.arm.reload_config()
        self.arm.open_gripper()
        self.arm.stow()
        self.set_state(RobotState.SEARCHING)

    def stop(self):
        """Emergency stop robot and return to IDLE."""
        self.comm.send_stop()
        self.tracker.stop()
        self.context.set_tracked_roi(None)
        self.set_state(RobotState.IDLE)

    def _handle_idle(self):
        # In IDLE, do nothing except keep motors stopped
        pass

    def _handle_searching(self, frame: Optional[np.ndarray]):
        if frame is None:
            return

        now = time.time()
        # Periodically nudge to rotate and search
        if (now - self._last_search_turn) * 1000.0 > (self._search_turn_step_ms + self._search_pause_ms):
            self._last_search_turn = now
            # Take this opportunity when stationary to trigger VLM categorization
            self.set_state(RobotState.CATEGORIZING)

    def _handle_categorizing(self, frame: Optional[np.ndarray]):
        if frame is None:
            self.set_state(RobotState.SEARCHING)
            return

        # Ensure robot is fully stopped during VLM capture
        self.comm.send_stop()

        start_t = time.time()
        result: DetectionResult = self.vision.categorize(frame)
        self.context.inference_latency_s = round(time.time() - start_t, 2)
        self.context.update_detection(result)

        if result.detected and result.pickable:
            print(f"[StateMachine] Target identified: {result.category} ({result.material_color})")
            # Initialize 30 FPS tracker with bounding box
            success = self.tracker.start_tracking(frame, result.bounding_box)
            if success:
                self.context.set_tracked_roi(self.tracker.current_roi)
                self.set_state(RobotState.TRACKING_APPROACH)
            else:
                self.set_state(RobotState.SEARCHING)
        else:
            # Not found or not pickable; continue search scan
            self.comm.send_nudge("R", duration_ms=self._search_turn_step_ms, pwm=self._search_nudge_pwm)
            self.set_state(RobotState.SEARCHING)

    def _handle_tracking_approach(self, frame: Optional[np.ndarray]):
        if frame is None:
            return

        success, roi = self.tracker.update(frame)
        if not success or roi is None:
            print("[StateMachine] Visual tracker lost target. Re-categorizing...")
            self.tracker.stop()
            self.context.set_tracked_roi(None)
            self.comm.send_stop()
            self.set_state(RobotState.CATEGORIZING)
            return

        self.context.set_tracked_roi(roi)
        ground_pt = self.tracker.ground_contact_point
        action = self.servoing.compute_action(ground_pt)

        if action.action_type == "DRIVE":
            self.comm.send_drive(action.left_pwm, action.right_pwm)
        elif action.action_type == "NUDGE":
            self.comm.send_nudge(action.direction, action.duration_ms, action.pwm)
        elif action.action_type == "ALIGNED":
            self.comm.send_stop()
            self.set_state(RobotState.FINE_ALIGNMENT)

    def _handle_fine_alignment(self, frame: Optional[np.ndarray]):
        if frame is None:
            return

        success, roi = self.tracker.update(frame)
        if not success:
            self.set_state(RobotState.CATEGORIZING)
            return

        self.context.set_tracked_roi(roi)
        ground_pt = self.tracker.ground_contact_point
        action = self.servoing.compute_action(ground_pt)

        if action.is_aligned:
            self._aligned_count += 1
            if self._aligned_count >= self._required_aligned_frames:
                self._aligned_count = 0
                self.comm.send_stop()
                self.set_state(RobotState.PRE_GRASP_VERIFY)
        else:
            self._aligned_count = 0
            if action.action_type == "NUDGE":
                self.comm.send_nudge(action.direction, action.duration_ms, action.pwm)
            elif action.action_type == "DRIVE":
                self.comm.send_drive(action.left_pwm, action.right_pwm)

    def _handle_pre_grasp_verify(self, frame: Optional[np.ndarray]):
        """Verify object is directly in sweet spot before lowering arm."""
        self.comm.send_stop()
        time.sleep(0.3)  # Let vibrations settle

        sweet_spot = (
            self.context.sweet_spot_x,
            self.context.sweet_spot_y,
            self.context.sweet_spot_w,
            self.context.sweet_spot_h
        )

        aligned = self.vision.verify_grasp_alignment(frame, sweet_spot)
        if aligned:
            print("[StateMachine] Pre-grasp alignment confirmed! Executing pick...")
            self.set_state(RobotState.PICKING)
        else:
            print("[StateMachine] Pre-grasp check failed; adjusting alignment.")
            self.set_state(RobotState.FINE_ALIGNMENT)

    def _handle_picking(self):
        """Execute physical pick motion."""
        self.arm.reload_config()
        self.arm.execute_pick_sequence(wait_completion=True, timeout_s=4.5)
        self.set_state(RobotState.POST_GRASP_VERIFY)

    def _handle_post_grasp_verify(self, frame: Optional[np.ndarray]):
        """Check if pick was successful."""
        # The arm is raised. In future, vision can check if jaws hold object.
        # For now, transition to OBJECT_SECURED
        print("[StateMachine] Pick sequence finished. Object secured.")
        self.set_state(RobotState.OBJECT_SECURED)

    def _handle_object_secured(self):
        # Stop and wait in secured state
        self.comm.send_stop()
