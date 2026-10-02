"""Arm manipulation and pick sequence controller driven by calibrated servo config."""

import json
import os
import time
from typing import Any, Dict, Optional
import yaml

from src.adapters.comm.base import BaseCommAdapter

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml"))
DEFAULT_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/arm_state.json"))


class ArmController:
    """Controls the 3-axis servo arm using angles dynamically loaded from config."""

    def __init__(self, comm_adapter: BaseCommAdapter, config: Optional[Dict[str, Any]] = None, state_path: Optional[str] = None):
        self.comm = comm_adapter
        self.config_path = DEFAULT_CONFIG_PATH
        self._custom_config = config is not None
        self.config = config or self._load_config()
        self.state_path = state_path or (None if self._custom_config else DEFAULT_STATE_PATH)
        self.cur_s1 = 90
        self.cur_s2 = 90
        self.cur_s3 = 90
        self._load_calibrated_angles()
        self.cur_s1 = self.s1_stow
        self.cur_s2 = self.s2_stow
        self.cur_s3 = self.s3_open
        if self.state_path:
            self._load_persisted_state()

    def _load_persisted_state(self):
        """Loads assigned servo angles from persistent storage across process resets."""
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                s1 = data.get("s1")
                s2 = data.get("s2")
                s3 = data.get("s3")
                if s1 is not None and s2 is not None and s3 is not None:
                    self.cur_s1 = max(self.s1_min, min(self.s1_max, int(s1)))
                    self.cur_s2 = max(self.s2_min, min(self.s2_max, int(s2)))
                    self.cur_s3 = max(self.s3_min, min(self.s3_max, int(s3)))
        except Exception:
            pass

    def _persist_state(self):
        """Persists currently assigned servo angles to disk so any reset preserves the pose and grip."""
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "s1": int(self.cur_s1),
                    "s2": int(self.cur_s2),
                    "s3": int(self.cur_s3),
                    "timestamp": time.time()
                }, f)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def _load_config(self) -> Dict[str, Any]:
        if getattr(self, "_custom_config", False):
            return self.config
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def reload_config(self, force_disk: bool = False):
        """Reload latest calibration parameters from disk or refresh calibrated angles."""
        if force_disk:
            self._custom_config = False
        if not getattr(self, "_custom_config", False) and os.path.exists(self.config_path):
            fresh_cfg = self._load_config()
            if fresh_cfg:
                self.config = fresh_cfg
        self._load_calibrated_angles()

    def _load_calibrated_angles(self):
        servos_cfg = self.config.get("servos", {})
        s1 = servos_cfg.get("servo1_shoulder", {})
        s2 = servos_cfg.get("servo2_elbow", {})
        s3 = servos_cfg.get("servo3_gripper", {})

        # Servo 1 (Shoulder)
        s1_down = s1.get("down_angle", 110)
        s1_stow = s1.get("up_angle", s1.get("stow_angle", 60))
        s1_min_cfg = s1.get("min_angle", min(s1_down, s1_stow, 60))
        s1_max_cfg = s1.get("max_angle", max(s1_down, s1_stow, 170))
        self.s1_min = min(s1_min_cfg, s1_down, s1_stow)
        self.s1_max = max(s1_max_cfg, s1_down, s1_stow)
        self.s1_stow = max(self.s1_min, min(self.s1_max, s1_stow))
        self.s1_up = self.s1_stow
        self.s1_down = max(self.s1_min, min(self.s1_max, s1_down))
        self.s1_center = s1.get("center_angle", (self.s1_min + self.s1_max) // 2)

        # Servo 2 (Elbow)
        s2_down = s2.get("down_angle", 50)
        s2_stow = s2.get("up_angle", s2.get("stow_angle", 70))
        s2_min_cfg = s2.get("min_angle", min(s2_down, s2_stow, 50))
        s2_max_cfg = s2.get("max_angle", max(s2_down, s2_stow, 110))
        self.s2_min = min(s2_min_cfg, s2_down, s2_stow)
        self.s2_max = max(s2_max_cfg, s2_down, s2_stow)
        self.s2_stow = max(self.s2_min, min(self.s2_max, s2_stow))
        self.s2_up = self.s2_stow
        self.s2_down = max(self.s2_min, min(self.s2_max, s2_down))
        self.s2_center = s2.get("center_angle", (self.s2_min + self.s2_max) // 2)

        # Servo 3 (Gripper)
        self.s3_close = s3.get("close_angle", 70)
        self.s3_open = s3.get("open_angle", 110)
        self.s3_center = s3.get("center_angle", (self.s3_close + self.s3_open) // 2)
        s3_min_cfg = s3.get("min_angle", min(self.s3_close, self.s3_open, 0))
        s3_max_cfg = s3.get("max_angle", max(self.s3_close, self.s3_open, 180))
        self.s3_min = min(s3_min_cfg, self.s3_close, self.s3_open)
        self.s3_max = max(s3_max_cfg, self.s3_close, self.s3_open)

        # Presentation Salute Pose (from calibrated parameters or midpoint of stow and down)
        self.s1_salute = max(self.s1_min, min(self.s1_max, s1.get("salute_angle", (self.s1_stow + self.s1_down) // 2)))
        self.s2_salute = max(self.s2_min, min(self.s2_max, s2.get("salute_angle", (self.s2_stow + self.s2_down) // 2)))
        self.step_deg = int(servos_cfg.get("step_deg", 5))
        self.step_delay_s = float(servos_cfg.get("step_delay_s", 0.03))
        self.lead_deg = int(servos_cfg.get("lead_deg", 0))

    def _calc_step_delay(self, step_delta: int, base_delay: float) -> float:
        """Calculate required delay for Arduino to step physical servo.
        Arduino firmware steps 1 degree per 20ms (SERVO_STEP_INTERVAL_MS = 20 in egrabbot.ino).
        To prevent firmware stepper starvation and desynchronization,
        delay must be at least abs(step_delta) * 0.022s.
        """
        deg = abs(step_delta)
        if deg == 0:
            return 0.0
        return max(base_delay, deg * 0.022)

    def move_arm_alternate(
        self,
        s1: int,
        s2: int,
        s3: Optional[int] = None,
        wait_for_s1: bool = True,
        step_deg: Optional[int] = None,
        delay_s: Optional[float] = None,
        lead_deg: Optional[int] = None,
        gripper_first: bool = False,
        clamp_to_bounds: bool = True,
        **kwargs
    ) -> bool:
        """Moves arm servos in alternating micro-increments between Servo 1 and Servo 2,
        ensuring smooth simultaneous-like motion while preventing excessive stretching
        and preserving directional lead priority:
        - Directional priority ("first to move going down and going up"):
          - Going UP / Stowing (delta_s2 > 0):
            Servo 2 (Elbow) initiates movement first, taking the first sub-step to lift off the ground,
            then Servo 1 (Shoulder) takes a sub-step, repeating in alternating fashion: S2 -> S1 -> S2 -> S1...
          - Going DOWN / Reaching (delta_s1 > 0 or delta_s2 < 0):
            Servo 1 (Shoulder) initiates movement first, taking the first sub-step,
            then Servo 2 (Elbow) takes a sub-step, repeating in alternating fashion: S1 -> S2 -> S1 -> S2...
        - If lead_deg > 0: an initial lead phase is executed for the leading joint before alternating starts.
        - Servo 3 (Gripper - Pin 11) is strictly isolated:
          - If gripper_first is True: Gripper finishes physical movement before base servos move.
          - Otherwise: Gripper moves after base servos complete movement.
        """
        self.reload_config()
        if s3 is None:
            s3 = self.cur_s3

        if clamp_to_bounds:
            s1 = max(self.s1_min, min(self.s1_max, int(s1)))
            s2 = max(self.s2_min, min(self.s2_max, int(s2)))
            s3 = max(self.s3_min, min(self.s3_max, int(s3)))
        else:
            s1 = max(0, min(180, int(s1)))
            s2 = max(0, min(180, int(s2)))
            s3 = max(0, min(180, int(s3)))

        start_s1 = self.cur_s1
        start_s2 = self.cur_s2
        delta_s1 = s1 - start_s1
        delta_s2 = s2 - start_s2
        delta_s3 = s3 - self.cur_s3

        # Case 1: Neither base servos nor gripper moved
        if delta_s1 == 0 and delta_s2 == 0 and delta_s3 == 0:
            return True

        # Case 2: Only Gripper (Pin 11) moved (base servos 9 & 10 stationary)
        if delta_s1 == 0 and delta_s2 == 0:
            return self.move_gripper(s3, wait=wait_for_s1, clamp_to_bounds=clamp_to_bounds)

        # Case 3: Gripper-first requested when both base and gripper change
        if gripper_first and delta_s3 != 0:
            self.move_gripper(s3, wait=True, clamp_to_bounds=clamp_to_bounds)

        is_mock = self.comm.__class__.__name__ == "MockCommAdapter"
        step = int(step_deg) if step_deg is not None else getattr(self, "step_deg", 5)
        step = max(1, step)
        step_delay = float(delay_s) if delay_s is not None else getattr(self, "step_delay_s", 0.03)
        lead = int(lead_deg) if lead_deg is not None else getattr(self, "lead_deg", 0)
        lead = max(0, lead)

        # Directional priority:
        # If lifting away from ground / stowing (delta_s2 > 0): S2 leads!
        # Otherwise (lowering / reaching down): S1 leads!
        s2_leads = (delta_s2 > 0 and delta_s1 <= 0) or (delta_s2 > 0 and delta_s1 < 0) or (delta_s2 > 0)

        # Phase 1: Lead-in phase for the priority joint if configured (lead > 0) and both joints move
        if delta_s1 != 0 and delta_s2 != 0 and lead > 0:
            if s2_leads:
                lead_target_s2 = min(s2, start_s2 + lead) if delta_s2 > 0 else max(s2, start_s2 - lead)
                lead_delta = lead_target_s2 - start_s2
                if lead_delta != 0:
                    num_lead_steps = max(1, (abs(lead_delta) + step - 1) // step)
                    for i in range(1, num_lead_steps + 1):
                        cur_lead_s2 = start_s2 + int(round(lead_delta * i / num_lead_steps))
                        if cur_lead_s2 != self.cur_s2:
                            step_sub = abs(cur_lead_s2 - self.cur_s2)
                            self.comm.send_servos(self.cur_s1, cur_lead_s2, self.cur_s3)
                            self.cur_s2 = cur_lead_s2
                            delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_sub, step_delay)
                            if delay > 0:
                                time.sleep(delay)
            else:
                lead_target_s1 = min(s1, start_s1 + lead) if delta_s1 > 0 else max(s1, start_s1 - lead)
                lead_delta = lead_target_s1 - start_s1
                if lead_delta != 0:
                    num_lead_steps = max(1, (abs(lead_delta) + step - 1) // step)
                    for i in range(1, num_lead_steps + 1):
                        cur_lead_s1 = start_s1 + int(round(lead_delta * i / num_lead_steps))
                        if cur_lead_s1 != self.cur_s1:
                            step_sub = abs(cur_lead_s1 - self.cur_s1)
                            self.comm.send_servos(cur_lead_s1, self.cur_s2, self.cur_s3)
                            self.cur_s1 = cur_lead_s1
                            delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_sub, step_delay)
                            if delay > 0:
                                time.sleep(delay)

        # Phase 2: Alternating micro-increments for remaining journey
        interp_start_s1 = self.cur_s1
        interp_start_s2 = self.cur_s2
        rem_delta_s1 = s1 - interp_start_s1
        rem_delta_s2 = s2 - interp_start_s2

        if rem_delta_s1 != 0 or rem_delta_s2 != 0:
            max_rem = max(abs(rem_delta_s1), abs(rem_delta_s2))
            num_cycles = max(1, (max_rem + step - 1) // step)

            for cycle in range(1, num_cycles + 1):
                ratio = cycle / float(num_cycles)
                next_s1 = interp_start_s1 + int(round(rem_delta_s1 * ratio))
                next_s2 = interp_start_s2 + int(round(rem_delta_s2 * ratio))

                if s2_leads:
                    # 1. Advance S2 by sub-step (S2 lifts first)
                    if next_s2 != self.cur_s2:
                        step_s2 = abs(next_s2 - self.cur_s2)
                        self.comm.send_servos(self.cur_s1, next_s2, self.cur_s3)
                        self.cur_s2 = next_s2
                        delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_s2, step_delay)
                        if delay > 0:
                            time.sleep(delay)
                    # 2. Advance S1 by sub-step
                    if next_s1 != self.cur_s1:
                        step_s1 = abs(next_s1 - self.cur_s1)
                        self.comm.send_servos(next_s1, self.cur_s2, self.cur_s3)
                        self.cur_s1 = next_s1
                        delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_s1, step_delay)
                        if delay > 0:
                            time.sleep(delay)
                else:
                    # 1. Advance S1 by sub-step (S1 extends first)
                    if next_s1 != self.cur_s1:
                        step_s1 = abs(next_s1 - self.cur_s1)
                        self.comm.send_servos(next_s1, self.cur_s2, self.cur_s3)
                        self.cur_s1 = next_s1
                        delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_s1, step_delay)
                        if delay > 0:
                            time.sleep(delay)
                    # 2. Advance S2 by sub-step
                    if next_s2 != self.cur_s2:
                        step_s2 = abs(next_s2 - self.cur_s2)
                        self.comm.send_servos(self.cur_s1, next_s2, self.cur_s3)
                        self.cur_s2 = next_s2
                        delay = 0.0 if (not wait_for_s1 or is_mock) else self._calc_step_delay(step_s2, step_delay)
                        if delay > 0:
                            time.sleep(delay)

        # Ensure exact targets are recorded
        self.cur_s1 = s1
        self.cur_s2 = s2

        # AFTER base servos finish movement, move gripper if needed (and not already moved via gripper_first)
        if not gripper_first and s3 != self.cur_s3:
            self.move_gripper(s3, wait=wait_for_s1, clamp_to_bounds=clamp_to_bounds)
        else:
            self.cur_s3 = s3

        self._persist_state()
        return True

    def move_arm_simultaneous(
        self,
        s1: int,
        s2: int,
        s3: Optional[int] = None,
        wait_for_s1: bool = True,
        step_deg: Optional[int] = None,
        delay_s: Optional[float] = None,
        lead_deg: Optional[int] = None,
        gripper_first: bool = False,
        clamp_to_bounds: bool = True,
        **kwargs
    ) -> bool:
        """Moves arm servos in synchronized alternating micro-increments (alias/wrapper of move_arm_alternate)."""
        return self.move_arm_alternate(
            s1=s1,
            s2=s2,
            s3=s3,
            wait_for_s1=wait_for_s1,
            step_deg=step_deg,
            delay_s=delay_s,
            lead_deg=lead_deg,
            gripper_first=gripper_first,
            clamp_to_bounds=clamp_to_bounds,
            **kwargs
        )

    def move_arm_sequential(
        self,
        s1: int,
        s2: int,
        s3: Optional[int] = None,
        wait_for_s1: bool = True,
        gripper_first: bool = False,
        clamp_to_bounds: bool = True,
        force_sequential: bool = False,
        **kwargs
    ) -> bool:
        """Moves arm servos. By default delegates to move_arm_alternate with alternating micro-stepping.
        Set force_sequential=True if strict one-joint-at-a-time motion is specifically required.
        """
        if not force_sequential:
            return self.move_arm_alternate(
                s1=s1,
                s2=s2,
                s3=s3,
                wait_for_s1=wait_for_s1,
                gripper_first=gripper_first,
                clamp_to_bounds=clamp_to_bounds,
                **kwargs
            )

        self.reload_config()
        if s3 is None:
            s3 = self.cur_s3

        if clamp_to_bounds:
            s1 = max(self.s1_min, min(self.s1_max, int(s1)))
            s2 = max(self.s2_min, min(self.s2_max, int(s2)))
            s3 = max(self.s3_min, min(self.s3_max, int(s3)))
        else:
            s1 = max(0, min(180, int(s1)))
            s2 = max(0, min(180, int(s2)))
            s3 = max(0, min(180, int(s3)))

        delta_s1 = s1 - self.cur_s1
        delta_s2 = s2 - self.cur_s2
        delta_s3 = s3 - self.cur_s3

        if delta_s1 == 0 and delta_s2 == 0 and delta_s3 == 0:
            return True

        if delta_s1 == 0 and delta_s2 == 0:
            return self.move_gripper(s3, wait=wait_for_s1, clamp_to_bounds=clamp_to_bounds)

        if gripper_first and delta_s3 != 0:
            self.move_gripper(s3, wait=True, clamp_to_bounds=clamp_to_bounds)

        is_mock = self.comm.__class__.__name__ == "MockCommAdapter"
        s2_first = (delta_s2 > 0 and delta_s1 <= 0) or (delta_s2 > 0 and delta_s1 < 0) or (delta_s2 > 0)

        if s2_first:
            if s2 != self.cur_s2:
                self.comm.send_servos(self.cur_s1, s2, self.cur_s3)
                if wait_for_s1 and not is_mock:
                    s2_delta = abs(s2 - self.cur_s2)
                    time.sleep(max(0.35, s2_delta * 0.025 + 0.1))
                self.cur_s2 = s2
            if s1 != self.cur_s1:
                self.comm.send_servos(s1, self.cur_s2, self.cur_s3)
                if wait_for_s1 and not is_mock:
                    s1_delta = abs(s1 - self.cur_s1)
                    time.sleep(max(0.35, s1_delta * 0.025 + 0.1))
                self.cur_s1 = s1
        else:
            if s1 != self.cur_s1:
                self.comm.send_servos(s1, self.cur_s2, self.cur_s3)
                if wait_for_s1 and not is_mock:
                    s1_delta = abs(s1 - self.cur_s1)
                    time.sleep(max(0.35, s1_delta * 0.025 + 0.1))
                self.cur_s1 = s1
            if s2 != self.cur_s2:
                self.comm.send_servos(self.cur_s1, s2, self.cur_s3)
                if wait_for_s1 and not is_mock:
                    s2_delta = abs(s2 - self.cur_s2)
                    time.sleep(max(0.35, s2_delta * 0.025 + 0.1))
                self.cur_s2 = s2

        self.cur_s1 = s1
        self.cur_s2 = s2

        if not gripper_first and s3 != self.cur_s3:
            self.move_gripper(s3, wait=wait_for_s1, clamp_to_bounds=clamp_to_bounds)
        else:
            self.cur_s3 = s3

        return True


    def stow(self, s3: Optional[int] = None) -> bool:
        """Move arm to upright stowed position (Servo 2 lifts up first, then Servo 1 retracts).
        By default, preserves the current gripper angle (self.cur_s3) to securely hold grasped objects.
        """
        self.reload_config()
        target_s3 = s3 if s3 is not None else self.cur_s3
        return self.move_arm_simultaneous(self.s1_stow, self.s2_stow, target_s3)

    def center(self) -> bool:
        """Center all 3 servos to their neutral calibration angles."""
        self.reload_config()
        return self.move_arm_simultaneous(self.s1_center, self.s2_center, self.s3_center)

    def salute(self, s3: Optional[int] = None, wait: bool = True) -> bool:
        """Move arm to elevated presentation salute pose based on calibrated angles."""
        self.reload_config()
        target_s3 = s3 if s3 is not None else self.cur_s3
        return self.move_arm_simultaneous(self.s1_salute, self.s2_salute, target_s3, wait_for_s1=wait)

    def _wait_gripper_motion(self, from_angle: int, to_angle: int, clamping_hold: bool = False, timeout_s: Optional[float] = None) -> bool:
        """Waits until Servo 3 (gripper) finishes physical travel to to_angle.
        
        Firmware smooths motion at ~100 deg/s (2 deg per 20ms tick).
        Calculates required travel duration dynamically: delta / 100 + settling margin.
        If clamping_hold is True, applies a firm mechanical clamping hold on the payload.
        Monitors live telemetry if available.
        """
        delta = abs(from_angle - to_angle)
        hold_s = 0.35 if clamping_hold else 0.15
        if timeout_s is None:
            # Dynamically compute travel time (e.g. 150 deg -> 1.5s + 0.4s + hold = ~2.25s)
            timeout_s = max(0.8, (delta / 100.0) + 0.4 + hold_s)

        # Check if running mock comm adapter (keep unit tests snappy)
        is_mock = self.comm.__class__.__name__ == "MockCommAdapter"
        if is_mock:
            time.sleep(0.05)
            return True

        start_t = time.time()
        while time.time() - start_t < timeout_s:
            telem = self.comm.get_latest_telemetry()
            if hasattr(telem, "s3_gripper") and telem.s3_gripper > 0:
                if abs(telem.s3_gripper - to_angle) <= 3:
                    time.sleep(hold_s)
                    return True
            time.sleep(0.05)

        return True

    def open_gripper(self, wait: bool = False) -> bool:
        """Open gripper to calibrated open_angle. If wait=True, blocks until jaws finish opening."""
        self.reload_config()
        prev_s3 = self.cur_s3
        self.cur_s3 = self.s3_open
        self._persist_state()
        res = self.comm.send_servos(self.cur_s1, self.cur_s2, self.s3_open)
        if wait:
            self._wait_gripper_motion(prev_s3, self.s3_open)
        return res

    def close_gripper(self, wait: bool = False, angle: Optional[int] = None) -> bool:
        """Close gripper to calibrated close_angle. If wait=True, blocks until jaws finish clamping.
        Clamps angle within calibrated bounds to prevent mechanical servo stall and brownout resets.
        """
        self.reload_config()
        prev_s3 = self.cur_s3
        target_s3 = angle if angle is not None else self.s3_close
        target_s3 = max(self.s3_min, min(self.s3_max, int(target_s3)))
        self.cur_s3 = target_s3
        self._persist_state()
        res = self.comm.send_servos(self.cur_s1, self.cur_s2, target_s3)
        if wait:
            self._wait_gripper_motion(prev_s3, target_s3, clamping_hold=True)
        return res

    def move_gripper(self, angle: int, wait: bool = False, clamp_to_bounds: bool = True) -> bool:
        """Move gripper (Servo 3) directly to specified angle while holding current arm position."""
        self.reload_config()
        prev_s3 = self.cur_s3
        if clamp_to_bounds:
            s3 = max(self.s3_min, min(self.s3_max, int(angle)))
        else:
            s3 = max(0, min(180, int(angle)))
        self.cur_s3 = s3
        self._persist_state()
        res = self.comm.send_servos(self.cur_s1, self.cur_s2, s3)
        if wait:
            self._wait_gripper_motion(prev_s3, s3)
        return res

    def arm_down(self, s3: Optional[int] = None) -> bool:
        """Lower arm into ground pick reach position (Servo 1 first, then simultaneous)."""
        self.reload_config()
        target_s3 = s3 if s3 is not None else self.cur_s3
        return self.move_arm_simultaneous(self.s1_down, self.s2_down, target_s3)

    def arm_up(self, s3: Optional[int] = None) -> bool:
        """Raise arm back to stowed upright position (Servo 2 lifts first, then simultaneous)."""
        self.reload_config()
        target_s3 = s3 if s3 is not None else self.cur_s3
        return self.move_arm_simultaneous(self.s1_stow, self.s2_stow, target_s3)

    def execute_pick_sequence(self, wait_completion: bool = False, timeout_s: float = 6.0, clamp_angle: Optional[int] = None) -> bool:
        """Executes full pick-and-lift sequence with smart directional sequencing:
        1. Open gripper wide and confirm jaws finish opening.
        2. Lower arm into ground pick zone (Servo 1 extends first, then simultaneous lowering).
        3. Clamp gripper firmly on object — MUST finish closing completely before stowing!
        4. Raise arm back up holding object (Servo 2 lifts first with object, then simultaneous retraction).
        """
        self.reload_config()

        # Step 1: Open gripper wide and wait for jaws to fully open
        self.open_gripper(wait=True)

        # Step 2: Lower arm into ground pick zone (Servo 1 extends first, then simultaneous lowering)
        self.move_arm_simultaneous(self.s1_down, self.s2_down, self.s3_open, wait_for_s1=True)
        time.sleep(0.3)

        # Step 3: Clamp gripper firmly on object (using calibrated or custom clamp angle)
        target_clamp = clamp_angle if clamp_angle is not None else self.s3_close
        self.close_gripper(wait=True, angle=target_clamp)

        # Step 4: Raise arm back up holding object (Servo 2 lifts first with object, then simultaneous retraction)
        self.move_arm_simultaneous(self.s1_stow, self.s2_stow, target_clamp, wait_for_s1=True)
        time.sleep(0.3)

        return True


