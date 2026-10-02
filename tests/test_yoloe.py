"""Unit tests for YOLOE Open-Vocabulary Object Detection & Categorization Adapter."""

import os
import numpy as np
import pytest
from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter, DEFAULT_TARGET_CLASSES
from src.core.events import DetectionResult

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_TFLITE_S = os.path.join(BASE_DIR, "models", "yolo11s.tflite")
MODEL_TFLITE_N = os.path.join(BASE_DIR, "models", "yolo11n.tflite")
MODEL_TFLITE_L = os.path.join(BASE_DIR, "models", "yolo11l.tflite")
MODEL_26L = os.path.join(BASE_DIR, "models", "yoloe-26l-seg.pt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yoloe-11s-seg.pt")
ALT_MODEL_PATH = os.path.join(BASE_DIR, "models", "yoloe-26s-seg.pt")
NANO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yoloe-26n-seg.pt")
PF_MODEL_PATH = os.path.join(BASE_DIR, "models", "yoloe-26n-seg-pf.pt")


def get_available_model():
    for p in [MODEL_TFLITE_S, MODEL_TFLITE_N, MODEL_TFLITE_L, MODEL_26L, MODEL_PATH, ALT_MODEL_PATH, NANO_MODEL_PATH, PF_MODEL_PATH]:
        if os.path.exists(p):
            return p
    return None


def test_yoloe_init_default():
    """Verify YOLOE adapter initializes with default settings."""
    model = get_available_model()
    if not model:
        pytest.skip("YOLOE model weights not found in models/ directory")

    adapter = YOLOEVisionAdapter(model_path=model, confidence_threshold=0.25, imgsz=320)
    assert adapter.model_loaded is True
    assert adapter.confidence_threshold == 0.25
    assert adapter.imgsz == 320
    assert adapter.enforce_limitations is False
    assert len(adapter.target_classes) > 0


def test_yoloe_blank_frame():
    """Verify blank frame yields no false-positive detections."""
    model = get_available_model()
    if not model:
        pytest.skip("YOLOE model weights not found in models/ directory")

    adapter = YOLOEVisionAdapter(model_path=model, confidence_threshold=0.5, imgsz=320)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = adapter.categorize(blank)

    assert isinstance(res, DetectionResult)
    assert res.detected is False
    assert res.pickable is False


def test_yoloe_invalid_inputs():
    """Verify adapter handles None and empty frames gracefully without exceptions."""
    adapter = YOLOEVisionAdapter(model_path="nonexistent_model.pt")
    assert adapter.model_loaded is False

    res = adapter.categorize(None)
    assert isinstance(res, DetectionResult)
    assert res.detected is False

    all_dets = adapter.detect_all(None)
    assert isinstance(all_dets, list)
    assert len(all_dets) == 0


def test_yoloe_dynamic_target_classes():
    """Verify open-vocabulary target classes can be dynamically configured."""
    model = get_available_model()
    if not model:
        pytest.skip("No YOLOE model found")

    adapter = YOLOEVisionAdapter(model_path=model, confidence_threshold=0.25)
    if adapter.is_pre_fused:
        pytest.skip("Model is pre-fused; open-vocabulary text encoder not active")

    new_classes = ["bottle", "soda can", "plastic cup"]
    adapter.set_target_classes(new_classes)
    assert "bottle" in adapter.target_classes
    assert "soda_can" in adapter.target_classes
    assert "plastic_cup" in adapter.target_classes


def test_yoloe_enforce_limitations_flag():
    """Verify enforce_limitations parameter defaults to False (unconstrained)."""
    adapter = YOLOEVisionAdapter(model_path="dummy.pt", enforce_limitations=False)
    assert adapter.enforce_limitations is False

    adapter_limited = YOLOEVisionAdapter(model_path="dummy.pt", enforce_limitations=True)
    assert adapter_limited.enforce_limitations is True


def test_yoloe_verify_grasp_alignment():
    """Verify sweet spot grasp alignment logic."""
    adapter = YOLOEVisionAdapter(model_path="dummy.pt", confidence_threshold=0.25)

    # When no object is detected, verify_grasp_alignment must return False
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    aligned = adapter.verify_grasp_alignment(blank, (320, 240, 100, 100))
    assert aligned is False


def test_yoloe_detect_all_returns_list():
    """Verify detect_all returns a list of DetectionResults."""
    model = get_available_model()
    if not model:
        pytest.skip("YOLOE model weights not found in models/ directory")

    adapter = YOLOEVisionAdapter(model_path=model, confidence_threshold=0.25)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    results = adapter.detect_all(frame)
    assert isinstance(results, list)

