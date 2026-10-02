"""Unit tests for LiteRT (TFLite) inference runtime integration and color detection in vision adapter."""

import os
import numpy as np
import pytest
from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter
from src.adapters.vision.color_detector import detect_dominant_color, ColorInfo
from src.core.events import DetectionResult

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_ai_edge_litert_installed():
    """Verify Google's ai-edge-litert runtime is installed and accessible."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        assert Interpreter is not None
    except ImportError:
        pytest.fail("ai-edge-litert is not installed or importable in this environment")


def test_yoloe_adapter_backend_type_detection():
    """Verify adapter sets backend_type to LiteRT when .tflite path is given."""
    adapter_tflite = YOLOEVisionAdapter(model_path="dummy_model.tflite")
    assert adapter_tflite.backend_type == "LiteRT"
    assert adapter_tflite.model_loaded is False

    adapter_pt = YOLOEVisionAdapter(model_path="dummy_model.pt")
    assert adapter_pt.backend_type == "PyTorch"
    assert adapter_pt.model_loaded is False


def test_adapter_color_detection_flag():
    """Verify color_detection_enabled flag is present and active."""
    adapter = YOLOEVisionAdapter(model_path="dummy.pt")
    assert hasattr(adapter, "color_detection_enabled")
    assert adapter.color_detection_enabled is True


def test_detection_result_with_color():
    """Verify DetectionResult receives color information properly."""
    res = DetectionResult(
        detected=True,
        category="Bottle",
        material_color="Red",
        confidence=0.92,
        bounding_box=(100, 100, 200, 200),
        raw_response='{"model": "LiteRT", "color": "Red", "color_hex": "#E53935"}'
    )
    assert res.category == "Bottle"
    assert res.material_color == "Red"
    assert "#E53935" in res.raw_response
    assert "LiteRT" in res.raw_response


def test_litert_yolo11s_inference():
    """Verify yolo11s.tflite loads and executes with LiteRTEngine."""
    p = os.path.join(BASE_DIR, "models", "yolo11s.tflite")
    if not os.path.exists(p):
        pytest.skip("models/yolo11s.tflite not found")
    adapter = YOLOEVisionAdapter(model_path=p, confidence_threshold=0.25, num_threads=4)
    assert adapter.model_loaded is True
    assert adapter.backend_type == "LiteRT"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = adapter.categorize(frame)
    assert isinstance(res, DetectionResult)
    assert res.detected is False


def test_litert_yolo11l_inference():
    """Verify yolo11l.tflite (Large) loads and executes with LiteRTEngine."""
    p = os.path.join(BASE_DIR, "models", "yolo11l.tflite")
    if not os.path.exists(p):
        pytest.skip("models/yolo11l.tflite not found")
    adapter = YOLOEVisionAdapter(model_path=p, confidence_threshold=0.25, num_threads=4)
    assert adapter.model_loaded is True
    assert adapter.backend_type == "LiteRT"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = adapter.categorize(frame)
    assert isinstance(res, DetectionResult)
    assert res.detected is False


def test_litert_threads_parameter():
    """Verify num_threads is stored and passed correctly."""
    adapter = YOLOEVisionAdapter(model_path="dummy.tflite", num_threads=4)
    assert adapter.num_threads == 4
