"""Visual Servoing controller with anti-stiction micro-nudging and deadband control."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ServoingAction:
    """Action computed by the visual servoing control law."""
    action_type: str  # "DRIVE", "NUDGE", "STOP", "ALIGNED"
    # For DRIVE: (left_pwm, right_pwm)
    left_pwm: int = 0
    right_pwm: int = 0
    # For NUDGE:
    direction: str = "F"  # 'F', 'B', 'L', 'R'
    duration_ms: int = 70
    pwm: int = 210
    # Error metrics
    error_x: int = 0
    error_y: int = 0
    is_aligned: bool = False


class VisualServoingController:
    """Translates visual image-plane error (ex, ey) into motor actuation commands."""

    def __init__(
        self,
        center_x: int = 320,
        grasp_y: int = 380,
        tol_x: int = 20,
        tol_y: int = 25,
        base_speed: int = 210,
        turn_speed: int = 200,
        min_overcoming_pwm: int = 190,
        pulse_threshold_px: int = 45,
        nudge_duration_ms: int = 70,
        nudge_pwm: int = 210,
        trim_offset: int = 0
    ):
        self.center_x = center_x
        self.grasp_y = grasp_y
        self.tol_x = tol_x
        self.tol_y = tol_y
        self.base_speed = base_speed
        self.turn_speed = turn_speed
        self.min_overcoming_pwm = min_overcoming_pwm
        self.pulse_threshold_px = pulse_threshold_px
        self.nudge_duration_ms = nudge_duration_ms
        self.nudge_pwm = nudge_pwm
        self.trim_offset = trim_offset

    def _apply_trim(self, speed: int, direction: int = 1) -> Tuple[int, int]:
        left = speed
        right = speed
        if self.trim_offset > 0:
            right = max(100, right - self.trim_offset)
        elif self.trim_offset < 0:
            left = max(100, left - abs(self.trim_offset))
        return int(left * direction), int(right * direction)

    def reload_config(self, motors_cfg: dict):
        """Update controller with latest calibrated motor parameters."""
        if "base_speed" in motors_cfg:
            self.base_speed = int(motors_cfg["base_speed"])
        if "turn_speed" in motors_cfg:
            self.turn_speed = int(motors_cfg["turn_speed"])
        if "min_overcoming_pwm" in motors_cfg:
            self.min_overcoming_pwm = int(motors_cfg["min_overcoming_pwm"])
        if "nudge_pwm" in motors_cfg:
            self.nudge_pwm = int(motors_cfg["nudge_pwm"])
        if "nudge_default_ms" in motors_cfg:
            self.nudge_duration_ms = int(motors_cfg["nudge_default_ms"])
        if "trim_offset" in motors_cfg:
            self.trim_offset = int(motors_cfg["trim_offset"])

    def compute_action(
        self,
        target_pt: Optional[Tuple[int, int]]
    ) -> ServoingAction:
        """Compute the next motor action given current target (x, y) coordinates."""
        if target_pt is None:
            return ServoingAction(action_type="STOP")

        tx, ty = target_pt
        ex = tx - self.center_x  # Horizontal error (positive: target is to the right)
        ey = self.grasp_y - ty  # Vertical distance error (positive: target is farther away)

        # 1. Check if fully aligned within the gripper sweet spot
        if abs(ex) <= self.tol_x and abs(ey) <= self.tol_y:
            return ServoingAction(
                action_type="ALIGNED",
                error_x=ex,
                error_y=ey,
                is_aligned=True
            )

        # 2. If horizontal alignment is off, prioritize turning to face object
        if abs(ex) > self.tol_x:
            turn_dir = "R" if ex > 0 else "L"

            # Anti-Stiction logic: if close to setpoint, use discrete micro-nudges
            if abs(ex) <= self.pulse_threshold_px:
                return ServoingAction(
                    action_type="NUDGE",
                    direction=turn_dir,
                    duration_ms=self.nudge_duration_ms,
                    pwm=self.nudge_pwm,
                    error_x=ex,
                    error_y=ey
                )
            else:
                # Farther away: continuous proportional pivot
                if turn_dir == "R":
                    return ServoingAction(
                        action_type="DRIVE",
                        left_pwm=self.turn_speed,
                        right_pwm=-self.turn_speed,
                        error_x=ex,
                        error_y=ey
                    )
                else:
                    return ServoingAction(
                        action_type="DRIVE",
                        left_pwm=-self.turn_speed,
                        right_pwm=self.turn_speed,
                        error_x=ex,
                        error_y=ey
                    )

        # 3. Horizontal is centered; now approach vertically toward grasp line
        if ey > self.tol_y:
            # Object is in front, need to move forward
            if ey <= self.pulse_threshold_px:
                # Fine approach: short forward micro-nudge to avoid overshooting grasp line
                return ServoingAction(
                    action_type="NUDGE",
                    direction="F",
                    duration_ms=self.nudge_duration_ms,
                    pwm=self.nudge_pwm,
                    error_x=ex,
                    error_y=ey
                )
            else:
                # Continuous forward drive
                lpwm, rpwm = self._apply_trim(self.base_speed, direction=1)
                return ServoingAction(
                    action_type="DRIVE",
                    left_pwm=lpwm,
                    right_pwm=rpwm,
                    error_x=ex,
                    error_y=ey
                )
        elif ey < -self.tol_y:
            # Over-shot: object is too close / behind grasp line, small reverse nudge
            return ServoingAction(
                action_type="NUDGE",
                direction="B",
                duration_ms=self.nudge_duration_ms,
                pwm=self.nudge_pwm,
                error_x=ex,
                error_y=ey
            )

        # Edge case: within both tolerances
        return ServoingAction(
            action_type="ALIGNED",
            error_x=ex,
            error_y=ey,
            is_aligned=True
        )
