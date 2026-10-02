"""Unit tests for Communication, Camera, and Vision Adapters."""

import numpy as np
import pytest
from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.comm.mock_comm import MockCommAdapter
from src.adapters.vision.mock_vision import MockVisionAdapter


def test_mock_comm_adapter():
    comm = MockCommAdapter()
    assert comm.connect() is True
    assert comm.is_connected() is True

    comm.send_drive(150, 150)
    telem = comm.get_latest_telemetry()
    assert telem.left_pwm == 150
    assert telem.right_pwm == 150

    comm.send_stop()
    telem = comm.get_latest_telemetry()
    assert telem.left_pwm == 0
    assert telem.right_pwm == 0

    comm.send_servos(45, 55, 130)
    telem = comm.get_latest_telemetry()
    assert telem.s1_shoulder == 45
    assert telem.s2_elbow == 55
    assert telem.s3_gripper == 130

    comm.send_macro("PICK")
    telem = comm.get_latest_telemetry()
    assert telem.macro_active is True

    assert comm.ping() > 0
    comm.disconnect()
    assert comm.is_connected() is False


def test_mock_camera_adapter():
    cam = MockCameraAdapter(width=320, height=240, fps=30)
    assert cam.start() is True
    assert cam.is_opened() is True

    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert cam.get_resolution() == (320, 240)
    assert cam.get_fps() == 30.0

    cam.stop()
    assert cam.is_opened() is False


def test_mock_vision_adapter():
    vision = MockVisionAdapter(
        mock_category="resistor_tray",
        mock_color="gray",
        mock_pickable=True,
        mock_confidence=0.98,
        mock_bbox=(100, 120, 160, 200)
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = vision.categorize(frame)
    assert res.detected is True
    assert res.category == "resistor_tray"
    assert res.confidence == 0.98
    assert res.center == (160, 130)

    # Grasp sweet spot verification
    # Sweet spot centered at (160, 130) with box 60x60
    assert vision.verify_grasp_alignment(frame, (160, 130, 60, 60)) is True
    # Sweet spot far away at (50, 50)
    assert vision.verify_grasp_alignment(frame, (50, 50, 40, 40)) is False


def test_bluetooth_resends_assigned_angles_on_ready_reset():
    """Verify BluetoothSerialAdapter automatically re-asserts assigned angles upon Arduino reset."""
    from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
    adapter = BluetoothSerialAdapter(port="/dev/null")
    sent_packets = []
    adapter._send_raw = lambda data: sent_packets.append(data) or True

    # 1. Assign servo angles
    adapter.send_servos(100, 40, 75)
    assert adapter._last_assigned_servos == (100, 40, 75)
    assert sent_packets[-1] == b"<SERVO:100,40,75>\n"

    # 2. Simulate incoming Arduino reset event (<READY:...>)
    sent_packets.clear()
    adapter._handle_incoming_packet({"type": "READY", "version": "ErovoutikaGrab_v2.0"})

    # 3. Verify adapter automatically re-asserted the assigned servo angles
    assert len(sent_packets) == 1
    assert sent_packets[0] == b"<SERVO:100,40,75>\n"


def test_bluetooth_watchdog_pushes_every_100ms(tmp_path):
    """Verify watchdog loop continuously pushes assigned angles every interval without continuous jitter."""
    import threading
    import time
    from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
    adapter = BluetoothSerialAdapter(port="/dev/null")
    adapter._state_path = str(tmp_path / "arm_state.json")
    sent_packets = []
    adapter._send_raw = lambda data: sent_packets.append(data) or True
    adapter._telemetry.connected = True
    adapter.is_connected = lambda: True

    adapter.send_servos(93, 45, 70, pulse=False)
    assert sent_packets[-1] == b"<SERVO:93,45,70>\n"
    sent_packets.clear()

    # Run watchdog loop briefly
    adapter._watchdog_interval_s = 0.04  # Accelerate for snappy unit test
    adapter._pulse_duration_s = 0.01
    adapter._running = True
    t = threading.Thread(target=adapter._watchdog_loop, daemon=True)
    t.start()
    time.sleep(0.14)
    adapter._running = False
    t.join(timeout=0.3)

    # Should have pushed packets every interval pulsing S3 between 70 and 71
    assert len(sent_packets) >= 2
    assert all(pkt in (b"<SERVO:93,45,70>\n", b"<SERVO:93,45,71>\n") for pkt in sent_packets)


def test_bluetooth_send_servos_pulses_plus_one_then_back():
    """Verify send_servos increments 1 degree then decrements back to original angle like a pulse."""
    from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
    adapter = BluetoothSerialAdapter(port="/dev/null")
    sent_packets = []
    adapter._send_raw = lambda data: sent_packets.append(data) or True

    adapter.send_servos(93, 45, 70, pulse=True)
    assert len(sent_packets) == 2
    # 1. Pulse packet: 70 + 1 = 71 (only S3 is incremented by 1, S1 and S2 untouched)
    assert sent_packets[0] == b"<SERVO:93,45,71>\n"
    # 2. Target packet: decremented back to 70
    assert sent_packets[1] == b"<SERVO:93,45,70>\n"
    assert adapter._last_assigned_servos == (93, 45, 70)


def test_bluetooth_send_servos_gripper_only_pulse_gating():
    """Verify pulse happens ONLY for gripper servo and never when moving S1/S2."""
    from src.adapters.comm.bluetooth_serial import BluetoothSerialAdapter
    adapter = BluetoothSerialAdapter(port="/dev/null")
    sent_packets = []
    adapter._send_raw = lambda data: sent_packets.append(data) or True

    # 1. Initial position set
    adapter.send_servos(93, 45, 70)
    sent_packets.clear()

    # 2. Move S1 only (Shoulder): S3 remains 70 -> MUST NOT pulse
    adapter.send_servos(100, 45, 70)
    assert len(sent_packets) == 1
    assert sent_packets[0] == b"<SERVO:100,45,70>\n"
    sent_packets.clear()

    # 3. Move S2 only (Elbow): S3 remains 70 -> MUST NOT pulse
    adapter.send_servos(100, 50, 70)
    assert len(sent_packets) == 1
    assert sent_packets[0] == b"<SERVO:100,50,70>\n"
    sent_packets.clear()

    # 4. Move S3 (Gripper changed from 70 to 80) -> MUST pulse S3 (+1 degree, then target)
    adapter.send_servos(100, 50, 80)
    assert len(sent_packets) == 2
    assert sent_packets[0] == b"<SERVO:100,50,81>\n"  # S1 and S2 stay identical, only S3 pulses
    assert sent_packets[1] == b"<SERVO:100,50,80>\n"
    sent_packets.clear()

    # 5. Explicit pulse=False: even if S3 changes, suppress pulse
    adapter.send_servos(100, 50, 90, pulse=False)
    assert len(sent_packets) == 1
    assert sent_packets[0] == b"<SERVO:100,50,90>\n"




