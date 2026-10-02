"""Unit tests and benchmarks for FastColorDetector."""

import time
import numpy as np
import pytest
from src.adapters.vision.color_detector import FastColorDetector, detect_dominant_color, ColorInfo


def test_color_detector_primary_colors():
    """Verify detection of pure primary and secondary colors."""
    detector = FastColorDetector(core_crop_ratio=0.6)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. Red Patch (OpenCV BGR format)
    frame[:] = (0, 0, 255)
    c_red = detector.detect_color(frame, (20, 20, 180, 180))
    assert c_red.name == "Red"
    assert c_red.hex_code == "#E53935"
    assert c_red.is_monochrome is False

    # 2. Green Patch
    frame[:] = (0, 255, 0)
    c_green = detector.detect_color(frame, (20, 20, 180, 180))
    assert c_green.name == "Green"
    assert c_green.hex_code == "#43A047"

    # 3. Blue Patch
    frame[:] = (255, 0, 0)
    c_blue = detector.detect_color(frame, (20, 20, 180, 180))
    assert c_blue.name == "Blue"
    assert c_blue.hex_code == "#1E88E5"

    # 4. Yellow Patch (Red + Green in BGR)
    frame[:] = (0, 255, 255)
    c_yellow = detector.detect_color(frame, (20, 20, 180, 180))
    assert c_yellow.name == "Yellow"
    assert c_yellow.hex_code == "#FDD835"


def test_color_detector_monochrome():
    """Verify detection of Black, White, and Gray."""
    detector = FastColorDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Black
    frame[:] = (10, 10, 10)
    c_black = detector.detect_color(frame, (10, 10, 90, 90))
    assert c_black.name == "Black"
    assert c_black.is_monochrome is True

    # White
    frame[:] = (250, 250, 250)
    c_white = detector.detect_color(frame, (10, 10, 90, 90))
    assert c_white.name == "White"
    assert c_white.is_monochrome is True

    # Gray
    frame[:] = (120, 120, 120)
    c_gray = detector.detect_color(frame, (10, 10, 90, 90))
    assert c_gray.name == "Gray"
    assert c_gray.is_monochrome is True


def test_color_detector_invalid_inputs():
    """Verify detector handles None and malformed inputs gracefully."""
    detector = FastColorDetector()
    assert detector.detect_color(None, (0, 0, 10, 10)).name == "Unknown"
    assert detector.detect_color(np.zeros((10, 10, 3)), None).name == "Unknown"
    assert detector.detect_color(np.zeros((10, 10, 3)), (0, 0)).name == "Unknown"


def test_color_detector_submillisecond_latency():
    """Verify HSV color classification completes in sub-millisecond time (< 1.0 ms)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = (0, 200, 100)

    # Warmup
    detect_dominant_color(frame, (100, 100, 300, 300))

    runs = 50
    t0 = time.time()
    for _ in range(runs):
        detect_dominant_color(frame, (100, 100, 300, 300))
    mean_ms = ((time.time() - t0) / runs) * 1000.0

    print(f"\n[Benchmark] FastColorDetector mean latency: {mean_ms:.3f} ms")
    assert mean_ms < 1.0, f"Color detection took too long: {mean_ms:.3f} ms (expected < 1.0 ms)"
