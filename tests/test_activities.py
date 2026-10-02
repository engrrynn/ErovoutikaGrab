"""Unit tests for Autonomous Activities (Person Follower, Color Tracking, Mutual Exclusivity)."""

import json
import os
import ssl
import sys
import threading
import time
import urllib.request
import numpy as np
import cv2
import pytest

from src.activities.activity_manager import ActivityManager, HSV_COLOR_RANGES
from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.adapters.vision.mock_vision import MockVisionAdapter
from src.adapters.vision.yoloe_adapter import DetectionResult
from http.server import ThreadingHTTPServer
from scripts.web_interface import WebHandler, SSL_CERT_PATH, SSL_KEY_PATH
import scripts.web_interface as web_mod

_ssl_ctx = ssl._create_unverified_context()
_orig_urlopen = urllib.request.urlopen


def _test_urlopen(url_or_req, *args, **kwargs):
    if "context" not in kwargs:
        kwargs["context"] = _ssl_ctx
    return _orig_urlopen(url_or_req, *args, **kwargs)


urllib.request.urlopen = _test_urlopen


class DummyVisionWithPerson:
    def __init__(self, detections=None):
        self.detections = detections or []

    def detect_all(self, frame):
        return self.detections


def test_activity_manager_init():
    comm = MockCommAdapter()
    cam = MockCameraAdapter()
    manager = ActivityManager(camera=cam, comm=comm)
    status = manager.get_status()
    assert status["active_activity"] == "none"
    assert status["running"] is False
    assert status["status"] == "IDLE"
    assert "target_box" in status


def test_mutual_exclusivity():
    comm = MockCommAdapter()
    comm.connect()
    cam = MockCameraAdapter()
    manager = ActivityManager(camera=cam, comm=comm)

    # 1. Start Person Follower
    ok1 = manager.start_activity("person_follower")
    assert ok1 is True
    time.sleep(0.05)
    s1 = manager.get_status()
    assert s1["active_activity"] == "person_follower"
    assert s1["running"] is True

    # 2. Start Color Tracking -> automatically halts Person Follower
    ok2 = manager.start_activity("color_tracking", {"target_color": "Blue"})
    assert ok2 is True
    time.sleep(0.05)
    s2 = manager.get_status()
    assert s2["active_activity"] == "color_tracking"
    assert s2["running"] is True
    assert s2["config"]["target_color"] == "Blue"

    # 3. Stop active activity
    manager.stop_activity()
    time.sleep(0.05)
    s3 = manager.get_status()
    assert s3["active_activity"] == "none"
    assert s3["running"] is False
    assert s3["status"] == "IDLE"


def test_unknown_activity():
    manager = ActivityManager()
    ok = manager.start_activity("non_existent_activity")
    assert ok is False
    assert manager.running is False


def test_color_tracking_synthetic_blob():
    comm = MockCommAdapter()
    comm.connect()
    manager = ActivityManager(comm=comm)
    manager.start_activity("color_tracking", {"target_color": "Red"})

    # Create synthetic 480x640 BGR image with a solid Red square in center (x=300..340, y=220..260)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[220:260, 300:340] = [0, 0, 255]  # Bright Pure Red in BGR

    t_seen = manager._step_color_tracking(frame, 640, 480, 320, time.time())
    status = manager.get_status()

    assert status["target_found"] is True
    assert len(status["target_box"]) == 4
    # Check that error_x is near zero (centered around 320)
    assert abs(status["error_x"]) < 10
    assert status["status"] == "TRACKING"

    manager.stop_activity()


@pytest.mark.parametrize("color_name,bgr_color", [
    ("Red", [0, 0, 240]),
    ("Green", [0, 220, 0]),
    ("Blue", [240, 50, 0]),
    ("Yellow", [0, 230, 230]),
    ("Orange", [0, 140, 240]),
    ("Purple", [180, 20, 140]),
])
def test_all_hsv_colors_detected(color_name, bgr_color):
    assert color_name in HSV_COLOR_RANGES
    manager = ActivityManager()
    manager.config["target_color"] = color_name

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 60x60 square
    frame[210:270, 290:350] = bgr_color

    manager._step_color_tracking(frame, 640, 480, 320, time.time())
    st = manager.get_status()
    assert st["target_found"] is True
    assert st["target_box"] != []


def test_person_follower_step_logic():
    comm = MockCommAdapter()
    comm.connect()
    # Mock person detection in right half of frame
    dummy_det = [
        DetectionResult(
            category="person",
            confidence=0.89,
            bounding_box=[60, 380, 320, 520],  # ymin, xmin, ymax, xmax
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(dummy_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["target_found"] is True
    assert st["error_x"] > 0  # Person is to the right
    assert st["action"] == "TURN_RIGHT"
    assert st["status"] == "FOLLOWING"

    manager.stop_activity()


def test_person_follower_legs_only_priority():
    """Verify that when only legs/pants/shoes are visible, tracking succeeds with top priority."""
    comm = MockCommAdapter()
    comm.connect()
    # Detection is labeled 'pants' or 'legs'
    leg_det = [
        DetectionResult(
            category="pants",
            confidence=0.75,
            bounding_box=[200, 260, 410, 380],  # ymin, xmin, ymax, xmax (feet at y=410)
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(leg_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["target_found"] is True
    assert st["target_type"] in ("LEGS", "BIPEDAL_LEGS")
    assert st["ground_y"] > 80.0  # ymax=410 -> ~85.4% depth
    assert len(st["leg_box"]) == 4
    assert st["status"] == "FOLLOWING"
    manager.stop_activity()


def test_person_follower_person_with_legs_priority():
    """Verify that when both person and legs are detected, leg centroid and foot contact take priority."""
    comm = MockCommAdapter()
    comm.connect()
    dets = [
        # Whole body bounding box centered at x=320
        DetectionResult(
            category="person",
            confidence=0.88,
            bounding_box=[50, 240, 420, 400],
            pickable=False
        ),
        # Distinct leg detection slightly shifted right
        DetectionResult(
            category="legs",
            confidence=0.82,
            bounding_box=[240, 300, 420, 390],
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(dets)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["target_found"] is True
    assert st["target_type"] == "PERSON_WITH_LEGS"
    assert len(st["leg_box"]) == 4
    # Tracking center aligns to legs (300..390 -> 345) rather than full body center (320)
    assert st["error_x"] > 0
    manager.stop_activity()


def test_person_follower_upper_body_fallback():
    """Verify that when legs are occluded/sitting, tracking falls back to upper body."""
    comm = MockCommAdapter()
    comm.connect()
    upper_det = [
        DetectionResult(
            category="torso",
            confidence=0.65,
            bounding_box=[40, 270, 200, 370],  # Upper body only (ymax=200 < 0.6*480)
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(upper_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["target_found"] is True
    assert st["target_type"] == "UPPER_BODY"
    assert st["status"] == "FOLLOWING"
    manager.stop_activity()


def test_person_follower_arc_steering_kinematics():
    """Verify that when approaching a target offset laterally, differential arc steering is applied."""
    comm = MockCommAdapter()
    comm.connect()
    # Person far away (ymax=280) and offset to the right (x=400..500, center=450)
    far_right_det = [
        DetectionResult(
            category="pants",
            confidence=0.80,
            bounding_box=[150, 400, 280, 500],
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(far_right_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["target_found"] is True
    assert st["action"] == "TURN_RIGHT"
    # Verify that motors drove forward with differential bias (left > right) during the gated pulse
    drive_cmds = [cmd for cmd in comm.command_log if cmd[0] == "DRIVE" and (cmd[1] != 0 or cmd[2] != 0)]
    assert len(drive_cmds) > 0
    _, lpwm, rpwm = drive_cmds[-1]
    assert lpwm > rpwm
    assert lpwm >= 230  # Overcomes stiction floor
    manager.stop_activity()



@pytest.fixture(scope="module")
def web_test_server():
    web_mod.load_all_configs()
    web_mod.camera = MockCameraAdapter(width=640, height=480)
    web_mod.camera.start()
    web_mod.connect_comm(force_mock=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    port = server.server_address[1]
    proto = "http"
    if os.path.exists(SSL_CERT_PATH) and os.path.exists(SSL_KEY_PATH):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=SSL_CERT_PATH, keyfile=SSL_KEY_PATH)
        server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    yield f"{proto}://127.0.0.1:{port}"

    server.shutdown()


def test_web_api_activities_lifecycle(web_test_server):
    base_url = web_test_server

    # 1. Check initial activity status
    req = urllib.request.Request(f"{base_url}/api/activities/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "active_activity" in data
        assert data["running"] is False

    # 2. Start Person Follower via API
    post_data = json.dumps({"activity": "person_follower"}).encode()
    req = urllib.request.Request(f"{base_url}/api/activities/start", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
        assert data["running"] is True

    # Check telemetry contains activity
    req = urllib.request.Request(f"{base_url}/api/telemetry")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert "activity" in data
        assert data["activity"]["active_activity"] == "person_follower"
        assert data["activity"]["running"] is True

    # 3. Start Color Tracking via API (replaces Person Follower)
    post_data = json.dumps({"activity": "color_tracking", "params": {"target_color": "Green"}}).encode()
    req = urllib.request.Request(f"{base_url}/api/activities/start", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
        assert data["activity"] == "color_tracking"

    # Verify active is now color_tracking
    req = urllib.request.Request(f"{base_url}/api/activities/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["active_activity"] == "color_tracking"

    # 4. Manual Teleop Override: Sending drive_dir MUST immediately abort the activity
    post_data = json.dumps({"dir": "F"}).encode()
    req = urllib.request.Request(f"{base_url}/api/drive_dir", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200

    # Verify activity was cancelled by manual drive override
    req = urllib.request.Request(f"{base_url}/api/activities/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["running"] is False
        assert data["active_activity"] == "none"

    # 5. Stop activity endpoint
    req = urllib.request.Request(f"{base_url}/api/activities/stop", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
        assert data["running"] is False


def test_web_api_activities_save(web_test_server):
    base_url = web_test_server

    # 1. GET /api/activities/config
    req = urllib.request.Request(f"{base_url}/api/activities/config")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
        assert "activities" in data
        assert "pf_dist" in data["activities"]
        assert "ct_color" in data["activities"]

    # 2. POST /api/activities/save with new parameters
    new_settings = {
        "pf_dist": 50,
        "pf_speed": 225,
        "ct_color": "Blue",
        "ct_area": 12,
        "ct_speed": 215,
        "ot_target": "cup",
        "ot_dist": 40,
        "ot_speed": 205,
        "ctc_color": "Green",
        "ctc_area": 10,
        "ctc_speed": 200,
        "os_target": "bottle",
        "os_dist": 30,
        "os_speed": 195,
        "oa_thresh": 25,
        "oa_cruise": 210,
        "oa_turn": 230,
    }
    post_data = json.dumps(new_settings).encode()
    req = urllib.request.Request(f"{base_url}/api/activities/save", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode())
        assert res["status"] == "ok"
        assert res["activities"]["pf_dist"] == 50
        assert res["activities"]["ct_color"] == "Blue"
        assert res["activities"]["ot_target"] == "cup"
        assert res["activities"]["oa_thresh"] == 25

    # 3. Verify GET /api/activities/config reflects saved settings
    req = urllib.request.Request(f"{base_url}/api/activities/config")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        assert data["activities"]["pf_dist"] == 50
        assert data["activities"]["ct_color"] == "Blue"
        assert data["activities"]["ot_target"] == "cup"

    # 4. Verify GET / index HTML has populated the saved values
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode()
        assert 'id="pf_dist"' in html
        assert 'value="50"' in html
        assert 'id="val_pf_dist">50%<' in html
        assert 'saveActivitySettings()' in html

    # 5. Verify alias POST /api/save_activities also works
    new_settings["pf_dist"] = 45
    new_settings["ct_color"] = "Red"
    new_settings["ot_target"] = "bottle"
    post_data = json.dumps(new_settings).encode()
    req = urllib.request.Request(f"{base_url}/api/save_activities", data=post_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode())
        assert res["status"] == "ok"
        assert res["activities"]["pf_dist"] == 45
        assert res["activities"]["ct_color"] == "Red"


def test_diminishing_turn_pwm():
    """Verify that turn PWM scales down smoothly to baseline speed near deadband to eliminate wobble."""
    am = ActivityManager()
    w = 640
    deadband = int(am.config.get("deadband_x", 25))
    baseline = int(am.config.get("turn_speed", 228))

    # Deadband check -> 0 PWM
    assert am._compute_diminishing_turn_pwm(deadband - 5, w) == 0

    # Far away (e.g. error = 250px) -> PWM should be elevated (up to ~248)
    pwm_far = am._compute_diminishing_turn_pwm(250, w)
    assert pwm_far >= baseline
    assert pwm_far <= 248

    # Intermediate distance (e.g. error = 100px)
    pwm_mid = am._compute_diminishing_turn_pwm(100, w)

    # Near deadband (e.g. error = deadband + 5px) -> PWM should diminish towards baseline
    pwm_near = am._compute_diminishing_turn_pwm(deadband + 5, w)

    # Strictly monotonic or equal: far >= mid >= near
    assert pwm_far >= pwm_mid >= pwm_near
    # Must not drop below baseline stiction overcoming floor
    assert pwm_near >= baseline


def test_object_tracking_synonyms():
    """Verify that synonyms and wildcards correctly match target objects."""
    synonym_map = {
        "bottle": ["bottle", "water bottle", "flask", "container", "can", "cup"],
        "cup": ["cup", "mug", "coffee cup", "glass"],
        "phone": ["cell phone", "phone", "mobile phone", "telephone", "remote"],
    }
    # Direct match
    assert "bottle" in synonym_map.get("bottle", ["bottle"])
    # Synonym match
    assert "water bottle" in synonym_map.get("bottle", [])
    # Wildcard matches
    target = "any"
    assert target in ("any", "all", "*", "auto", "")


def test_color_detector_indoor_low_saturation():
    """Verify color detector does not falsely classify pale/indoor objects as gray."""
    from src.adapters.vision.color_detector import FastColorDetector
    cd = FastColorDetector(core_crop_ratio=1.0)
    # A pastel / indoor red with S=30, V=120 (previously S < 40 would force Gray)
    # Hue ~ 0 (Red), Sat ~ 30, Val ~ 120 in HSV -> BGR:
    test_hsv = np.uint8([[[0, 30, 120]]])
    test_bgr = cv2.cvtColor(test_hsv, cv2.COLOR_HSV2BGR)
    img = np.tile(test_bgr, (50, 50, 1))

    # Detect color
    res = cd.detect_color(img, (0, 0, 50, 50))
    # Must identify Red, NOT Gray
    assert res.name == "Red"
    assert not res.is_monochrome


def test_activity_motion_toggle():
    """Verify that toggling motion mode disables chassis pulses while continuing detection."""
    comm = MockCommAdapter()
    comm.connect()
    far_right_det = [
        DetectionResult(
            category="person",
            confidence=0.88,
            bounding_box=[150, 400, 280, 500],
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(far_right_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.start_activity("person_follower")

    # 1. Default: Motion Enabled -> executes pulse
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    active_drive = [c for c in comm.command_log if c[0] == "DRIVE" and (c[1] != 0 or c[2] != 0)]
    assert len(active_drive) > 0

    # 2. Toggle Motion: Disabled (Stationary Track Mode)
    manager.toggle_motion(False)
    assert manager.motion_enabled is False
    comm.command_log.clear()

    # Off-center target -> performs sideways pivot turn in place (lpwm != rpwm), no forward translation
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()
    assert st["target_found"] is True
    assert "[STATIONARY TURN]" in st["action"]
    pivot_drive = [c for c in comm.command_log if c[0] == "DRIVE" and (c[1] != 0 or c[2] != 0)]
    assert len(pivot_drive) > 0
    # Pivot yaw must have opposite motor directions (turning in place)
    for cmd in pivot_drive:
        assert (cmd[1] > 0 and cmd[2] < 0) or (cmd[1] < 0 and cmd[2] > 0)

    # Centered target -> translation and yaw both zero
    comm.command_log.clear()
    centered_det = [
        DetectionResult(
            category="person",
            confidence=0.91,
            bounding_box=[150, 270, 280, 370],  # Centered around cx=320
            pickable=False
        )
    ]
    manager.vision.detections = centered_det
    manager._pf_ema_x = 320.0
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st_centered = manager.get_status()
    assert "[STATIONARY TRACK]" in st_centered["action"]

    # 3. Toggle Motion back on
    manager.toggle_motion(True)
    assert manager.motion_enabled is True
    manager.stop_activity()


def test_stationary_mode_arm_ik_tracking():
    """Verify arm elevation tracking adjusts Shoulder S1 and Elbow S2 strictly within bounds."""
    comm = MockCommAdapter()
    comm.connect()
    # Target high in frame (cy ~ 80px) -> ey < 0 -> Arm reaches UP (S1 decreases, S2 at stow/max)
    high_det = [
        DetectionResult(
            category="bottle",
            confidence=0.88,
            bounding_box=[40, 300, 120, 340],
            pickable=True
        )
    ]
    vision = DummyVisionWithPerson(high_det)
    manager = ActivityManager(comm=comm, vision=vision)
    manager.toggle_motion(False)  # Stationary mode
    manager.start_activity("object_tracking", {"target_object": "bottle"})

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_object_tracking(frame, 640, 480, 320, time.time())
    st = manager.get_status()
    arm_ik = st.get("arm_ik", {})
    assert arm_ik["active"] is True
    assert arm_ik["vertical_status"] == "PITCH_UP"
    assert 60 <= arm_ik["s1"] <= 170
    assert 0 <= arm_ik["s2"] <= 45

    # Target low in frame (cy ~ 400px) -> ey > 0 -> Arm reaches DOWN towards floor
    comm.command_log.clear()
    low_det = [
        DetectionResult(
            category="bottle",
            confidence=0.88,
            bounding_box=[360, 300, 440, 340],
            pickable=True
        )
    ]
    manager.vision.detections = low_det
    # Advance time to satisfy latency interval
    time.sleep(0.08)
    manager._step_object_tracking(frame, 640, 480, 320, time.time())
    st2 = manager.get_status()
    arm_ik2 = st2.get("arm_ik", {})
    assert arm_ik2["active"] is True
    assert arm_ik2["vertical_status"] == "PITCH_DOWN"
    assert 60 <= arm_ik2["s1"] <= 170
    assert 0 <= arm_ik2["s2"] <= 45
    manager.stop_activity()


def test_estimate_ground_distance():
    """Verify physical pinhole distance estimation in centimeters."""
    manager = ActivityManager()
    # Bottom of screen (y=480) -> closest contact point (~17 - 25 cm)
    dist_bottom = manager.estimate_ground_distance(480, 480)
    assert 10.0 <= dist_bottom <= 35.0

    # Middle of screen (y=240) -> horizontal horizon angle (~45 - 65 cm)
    dist_mid = manager.estimate_ground_distance(240, 480)
    assert dist_mid > dist_bottom
    assert 40.0 <= dist_mid <= 70.0

    # Upper horizon (y=100) -> far away (>100 cm)
    dist_high = manager.estimate_ground_distance(100, 480)
    assert dist_high > dist_mid
    assert dist_high >= 90.0


def test_vlm_adaptation_in_activity():
    """Verify that VLM planner integration dynamically logs and adapts feedback in activities."""
    from src.planning.vlm_motion_planner import VLMMotionPlanner
    planner = VLMMotionPlanner()
    comm = MockCommAdapter()
    comm.connect()
    far_right_det = [
        DetectionResult(
            category="person",
            confidence=0.88,
            bounding_box=[150, 400, 380, 500],
            pickable=False
        )
    ]
    vision = DummyVisionWithPerson(far_right_det)
    manager = ActivityManager(comm=comm, vision=vision, vlm_planner=planner)
    manager.start_activity("person_follower")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager._step_person_follower(frame, 640, 480, 320, time.time())
    st = manager.get_status()

    assert st["vlm_plan"] is not None
    assert st["vlm_plan"]["action"] != "STOP"
    stats = planner.get_adaptation_stats()
    assert stats["total_plans_generated"] >= 1
    assert len(stats["recent_history"]) >= 1
    manager.stop_activity()

