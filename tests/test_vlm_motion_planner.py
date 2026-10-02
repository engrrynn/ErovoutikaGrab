"""Unit tests for VLMMotionPlanner (Anti-Overshoot, Anti-Stiction, and Coordinate/Segmentation integration)."""

import pytest
from src.core.events import DetectionResult
from src.planning.vlm_motion_planner import VLMMotionPlanner, VLMMotionPlan


@pytest.fixture
def planner():
    return VLMMotionPlanner(
        min_overcoming_pwm=190,
        base_speed=210,
        turn_speed=200,
        nudge_pwm=210,
        nudge_default_ms=70,
        align_tol_x=20,
        pulse_threshold_px=45
    )


@pytest.fixture
def sweet_spot():
    return {
        "center_x": 324,
        "center_y": 247,
        "box_width": 196,
        "box_height": 157
    }


def test_no_target_halts(planner, sweet_spot):
    det = DetectionResult(detected=False)
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "STOP"
    assert plan.recommended_pwm == 0
    assert "No target detected" in plan.rationale


def test_fine_horizontal_alignment_avoids_stiction(planner, sweet_spot):
    # dx = 355 - 324 = +31px (within pulse_threshold_px=45, but outside align_tol_x=20)
    # Must use NUDGE_RIGHT with PWM >= min_overcoming_pwm (190)
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.88,
        material_color="Green",
        bounding_box=(200, 315, 330, 395),  # cx = 355
        mask_polygon=[(315, 200), (395, 200), (395, 330), (315, 330)],
        mask_area=10400
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "NUDGE_RIGHT"
    assert plan.recommended_pwm >= 190, "PWM must never drop below 190 to avoid stiction deadband"
    assert plan.alignment_status == "NEEDS_RIGHT"
    assert "stiction" in plan.rationale.lower()


def test_large_horizontal_offset_pivots(planner, sweet_spot):
    # dx = 220 - 324 = -104px (> pulse_threshold_px=45)
    det = DetectionResult(
        detected=True,
        category="can",
        confidence=0.92,
        bounding_box=(200, 180, 330, 260),  # cx = 220
        mask_polygon=[(180, 200), (260, 200), (260, 330), (180, 330)],
        mask_area=10400
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "PIVOT_LEFT"
    assert plan.recommended_pwm == 200
    assert plan.alignment_status == "NEEDS_LEFT"


def test_distant_forward_approach(planner, sweet_spot):
    # Centered: cx = 324. Area = 80 * 100 = 8,000 px^2 (sweet area = 30,772 -> ratio = 26%)
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.85,
        bounding_box=(200, 284, 300, 364),  # cx = 324, w=80, h=100
        mask_polygon=[(284, 200), (364, 200), (364, 300), (284, 300)],
        mask_area=8000
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "DRIVE_FORWARD"
    assert plan.recommended_pwm == 210
    assert plan.alignment_status == "CENTERED"
    assert plan.overshoot_risk == "LOW"


def test_anti_overshoot_damping_near_target(planner, sweet_spot):
    # Centered: cx = 324. Area = 150 * 150 = 22,500 px^2 (sweet area = 30,772 -> ratio = 73%)
    # Approaching sweet spot: MUST damp speed and use micro-nudge to prevent overshoot!
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.95,
        bounding_box=(172, 249, 322, 399),  # cx = 324, w=150, h=150
        mask_polygon=[(249, 172), (399, 172), (399, 322), (249, 322)],
        mask_area=22500
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "NUDGE_FORWARD"
    assert plan.damping_factor < 1.0, "Damping factor must scale down speed near target"
    assert plan.recommended_pwm >= 190, "PWM must still overcome stiction"
    assert "overshoot" in plan.rationale.lower()


def test_aligned_grasp_ready(planner, sweet_spot):
    # Centered: cx = 324. Area = 196 * 157 = 30,772 px^2 (ratio = 100%)
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.96,
        bounding_box=(168, 226, 325, 422),  # cx = 324, w=196, h=157
        mask_polygon=[(226, 168), (422, 168), (422, 325), (226, 325)],
        mask_area=30772
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "ALIGNED_GRASP"
    assert plan.recommended_pwm == 0
    assert plan.alignment_status == "CENTERED"


def test_overshoot_recovery(planner, sweet_spot):
    # Centered: cx = 324. Area = 220 * 190 = 41,800 px^2 (ratio = 135% > 125%)
    # Over-shot: must reverse nudge!
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.97,
        bounding_box=(150, 214, 340, 434),  # w=220, h=190
        mask_polygon=[(214, 150), (434, 150), (434, 340), (214, 340)],
        mask_area=41800
    )
    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "NUDGE_BACK"
    assert plan.overshoot_risk == "HIGH"
    assert "overshoot detected" in plan.rationale.lower()


def test_vlm_speed_adjust_follows_baseline_speed_in_robot_config(sweet_spot):
    # Initialize planner with values from config/robot_config.yaml
    planner_cfg = VLMMotionPlanner(
        min_overcoming_pwm=200,
        base_speed=240,
        turn_speed=228,
        nudge_pwm=245,
        nudge_default_ms=250,
        max_speed=255
    )

    # 1. Distant forward approach: speed matches base_speed (240)
    det_distant = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.90,
        bounding_box=(200, 284, 300, 364),  # ratio = 26%
        mask_polygon=[(284, 200), (364, 200), (364, 300), (284, 300)],
        mask_area=8000
    )
    plan_distant = planner_cfg.plan_movement(det_distant, sweet_spot, use_vlm_llm=False)
    assert plan_distant.action == "DRIVE_FORWARD"
    assert plan_distant.base_speed == 240
    assert plan_distant.recommended_pwm == 240
    assert plan_distant.to_dict()["base_speed"] == 240

    # 2. Near target: speed is damped proportionally from base_speed (240)
    # Area = 150 * 150 = 22,500 px^2 (sweet area = 30,772 -> ratio = 73.1%)
    # damping = max(0.6, 1.0 - (0.731 - 0.60) * 1.5) = ~0.8035
    # adjusted_pwm = round(240 * 0.8035) = 193 -> clamped >= min_overcoming_pwm (200) -> 200
    det_near = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.95,
        bounding_box=(172, 249, 322, 399),
        mask_polygon=[(249, 172), (399, 172), (399, 322), (249, 322)],
        mask_area=22500
    )
    plan_near = planner_cfg.plan_movement(det_near, sweet_spot, use_vlm_llm=False)
    assert plan_near.action == "NUDGE_FORWARD"
    assert plan_near.base_speed == 240
    assert plan_near.damping_factor < 1.0
    assert plan_near.recommended_pwm >= 200


def test_vlm_speed_adjust_dynamic_reload(planner, sweet_spot):
    # Initially base_speed = 210
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.90,
        bounding_box=(200, 284, 300, 364),
        mask_polygon=[(284, 200), (364, 200), (364, 300), (284, 300)],
        mask_area=8000
    )
    plan1 = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan1.base_speed == 210
    assert plan1.recommended_pwm == 210

    # Reload robot config with higher base speed (e.g., 250)
    planner.reload_config({"base_speed": 250, "min_overcoming_pwm": 215, "max_speed": 255})
    assert planner.base_speed == 250

    plan2 = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan2.base_speed == 250
    assert plan2.recommended_pwm == 250


def test_vlm_plans_arm_grasp_according_to_robot_config(sweet_spot):
    # Initialize planner with calibrated angles from config/robot_config.yaml
    planner_arm = VLMMotionPlanner(
        min_overcoming_pwm=230,
        base_speed=230,
        turn_speed=228,
        s1_stow=93,
        s1_down=170,
        s1_min=60,
        s1_max=170,
        s2_stow=45,
        s2_down=0,
        s2_min=0,
        s2_max=45,
        s3_open=170,
        s3_close=40,
        s3_min=40,
        s3_max=170
    )

    # Centered target in sweet-spot zone (area_ratio = 100%)
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.96,
        bounding_box=(168, 226, 325, 422),
        mask_polygon=[(226, 168), (422, 168), (422, 325), (226, 325)],
        mask_area=30772
    )

    plan = planner_arm.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.action == "ALIGNED_GRASP"
    assert plan.recommended_pwm == 0
    assert plan.arm_plan["arm_action"] == "EXECUTE_GRAB"
    assert plan.arm_plan["grasp_readiness"] == "READY"

    # Verify calibrated angles match robot config
    cal = plan.arm_plan["calibrated_angles"]
    assert cal["s1_stow"] == 93
    assert cal["s1_down"] == 170
    assert cal["s2_stow"] == 45
    assert cal["s2_down"] == 0
    assert cal["s3_open"] == 170
    assert cal["s3_close"] == 40
    assert cal["grip_target"] >= 40 and cal["grip_target"] <= 170

    # Verify 4-stage sequential grab trajectory
    traj = plan.arm_plan["trajectory"]
    assert len(traj) == 4
    assert traj[0]["name"] == "PRE_OPEN_JAWS"
    assert traj[0]["s1"] == 93 and traj[0]["s2"] == 45 and traj[0]["s3"] == 170
    assert traj[1]["name"] == "LOWER_TO_TARGET"
    assert traj[1]["s1"] == 170 and traj[1]["s2"] == 0 and traj[1]["s3"] == 170
    assert traj[2]["name"] == "CLAMP_GRIPPER"
    assert traj[2]["s1"] == 170 and traj[2]["s2"] == 0 and traj[2]["s3"] == cal["grip_target"]
    assert traj[3]["name"] == "LIFT_AND_STOW"
    assert traj[3]["s1"] == 93 and traj[3]["s2"] == 45 and traj[3]["s3"] == cal["grip_target"]

    # Verify rationale details the planned arm motion
    assert "servo arm grab planned" in plan.rationale.lower()
    assert "s1(170°)" in plan.rationale.lower()
    assert "s2(0°)" in plan.rationale.lower()


def test_vlm_arm_plan_dynamic_reload(planner, sweet_spot):
    # Dynamic reload with updated servo calibration (e.g. from web sliders)
    new_servos = {
        "s1_stow": 95,
        "s1_down": 165,
        "s2_stow": 40,
        "s2_down": 5,
        "s3_open": 160,
        "s3_close": 45
    }
    planner.reload_config(servos_cfg=new_servos)

    det = DetectionResult(
        detected=True,
        category="cup",
        confidence=0.94,
        bounding_box=(168, 226, 325, 422),
        mask_polygon=[(226, 168), (422, 168), (422, 325), (226, 325)],
        mask_area=30772
    )

    plan = planner.plan_movement(det, sweet_spot, use_vlm_llm=False)
    assert plan.arm_plan["arm_action"] == "EXECUTE_GRAB"
    cal = plan.arm_plan["calibrated_angles"]
    assert cal["s1_down"] == 165
    assert cal["s2_down"] == 5
    assert cal["s3_open"] == 160
    assert cal["s3_close"] == 45


def test_continuous_path_approach(planner, sweet_spot):
    """Verify that when distant, VLM planner plans a smooth continuous approach path."""
    det = DetectionResult(
        detected=True,
        category="person",
        confidence=0.90,
        bounding_box=(150, 274, 350, 374),  # centered cx=324
        mask_polygon=[(274, 150), (374, 150), (374, 350), (274, 350)],
        mask_area=20000
    )
    plan = planner.plan_movement(det, sweet_spot, dist_cm=85.0, target_dist_cm=45.0, use_vlm_llm=False)
    assert plan.action == "DRIVE_FORWARD"
    assert plan.continuous_drive is True
    assert plan.path_type == "CONTINUOUS_APPROACH"
    assert plan.left_pwm == planner.base_speed
    assert plan.right_pwm == planner.base_speed
    assert plan.is_overextended is False
    assert plan.overshoot_risk == "LOW"


def test_continuous_path_arc_alignment(planner, sweet_spot):
    """Verify that when offset horizontally, VLM planner plans an efficient continuous steering arc."""
    # dx = 360 - 324 = +36px
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.92,
        bounding_box=(150, 310, 350, 410),  # cx=360
        mask_polygon=[(310, 150), (410, 150), (410, 350), (310, 350)],
        mask_area=20000
    )
    plan = planner.plan_movement(det, sweet_spot, dist_cm=75.0, target_dist_cm=45.0, use_vlm_llm=False)
    assert plan.action == "TURN_RIGHT"
    assert plan.continuous_drive is True
    assert plan.path_type == "ARC_ALIGN"
    assert plan.left_pwm > plan.right_pwm
    assert plan.is_overextended is False


def test_anti_overextension_standoff_reached(planner, sweet_spot):
    """Verify that when the standoff distance is reached, VLM planner halts the robot to prevent over-extension."""
    det = DetectionResult(
        detected=True,
        category="cup",
        confidence=0.95,
        bounding_box=(150, 274, 350, 374),  # centered cx=324
        mask_polygon=[(274, 150), (374, 150), (374, 350), (274, 350)],
        mask_area=20000
    )
    plan = planner.plan_movement(det, sweet_spot, dist_cm=45.0, target_dist_cm=45.0, use_vlm_llm=False)
    assert plan.action in ("ALIGNED_HOLD", "ALIGNED_GRASP")
    assert plan.continuous_drive is False
    assert plan.path_type == "HOLD_STANDOFF"
    assert plan.left_pwm == 0
    assert plan.right_pwm == 0
    assert plan.is_overextended is False


def test_anti_overextension_reverse_recovery(planner, sweet_spot):
    """Verify that if the robot is closer than safe standoff, VLM planner detects over-extension and commands reverse clear."""
    det = DetectionResult(
        detected=True,
        category="bottle",
        confidence=0.93,
        bounding_box=(100, 250, 420, 398),  # close in camera
        mask_polygon=[(250, 100), (398, 100), (398, 420), (250, 420)],
        mask_area=47000
    )
    plan = planner.plan_movement(det, sweet_spot, dist_cm=32.0, target_dist_cm=45.0, use_vlm_llm=False)
    assert plan.action == "NUDGE_BACK"
    assert plan.continuous_drive is False
    assert plan.path_type == "REVERSE_CLEAR"
    assert plan.is_overextended is True
    assert plan.overshoot_risk == "HIGH"
    assert plan.left_pwm < 0
    assert plan.right_pwm < 0



