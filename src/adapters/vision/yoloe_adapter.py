"""YOLOE & Google LiteRT Object Detection and Segmentation Adapter.

Implements BaseVisionAdapter supporting both Google LiteRT (.tflite with multi-threaded
XNNPACK acceleration) and Ultralytics YOLOE (.pt open-vocabulary models) with sub-millisecond
HSV color detection for autonomous object categorization and visual servoing on Raspberry Pi 5.
"""

import os
import sys
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.adapters.vision.base import BaseVisionAdapter
from src.adapters.vision.color_detector import detect_dominant_color, ColorInfo
from src.core.events import DetectionResult

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Auto-discover local virtual environment packages if needed
venv_site = os.path.join(BASE_DIR, ".venv", "lib", "python3.13", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# 80 standard COCO classes for LiteRT models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Default unconstrained target classes for open-vocabulary detection
DEFAULT_TARGET_CLASSES = [
    "object",
    "item",
]
# Default pickable classes — objects the arm is allowed to grasp.
# The vision model detects and reports ALL objects; only classes listed
# here (or matching via substring) get marked pickable=True.
DEFAULT_PICKABLE_CLASSES = [
    "bottle",
    "can",
    "cup",
    "box",
    "package",
    "plastic",
    "paper",
    "metal",
    "trash",
    "toy",
    "tool",
    "phone",
    "ball",
    "device",
    "container",
]

# Non-pickable classes to reject (when limitations are explicitly enforced)
NON_PICKABLE_CLASSES = {
    "person", "table", "chair", "wall", "floor", "desk", "ceiling", "door", "window"
}


# 90-class COCO mapping (0-indexed where 0=person, 27=umbrella) for TF Lite detection models (SSD & EfficientDet)
COCO_CLASSES_90 = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic light", 10: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    26: "backpack", 27: "umbrella", 30: "handbag", 31: "tie", 32: "suitcase", 33: "frisbee", 34: "skis", 35: "snowboard",
    36: "sports ball", 37: "kite", 38: "baseball bat", 39: "baseball glove", 40: "skateboard", 41: "surfboard", 42: "tennis racket",
    43: "bottle", 45: "wine glass", 46: "cup", 47: "fork", 48: "knife", 49: "spoon", 50: "bowl",
    51: "banana", 52: "apple", 53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot", 57: "hot dog", 58: "pizza", 59: "donut", 60: "cake",
    61: "chair", 62: "couch", 63: "potted plant", 64: "bed", 66: "dining table", 69: "toilet",
    71: "tv", 72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard", 76: "cell phone",
    77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator", 83: "book", 84: "clock", 85: "vase", 86: "scissors",
    87: "teddy bear", 88: "hair drier", 89: "toothbrush"
}


class LiteRTEngine:
    """High-performance direct Google LiteRT inference engine with multi-architecture support (YOLO11, SSD, EfficientDet)."""

    def __init__(self, model_path: str, num_threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = Interpreter(model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

        self.input_shape = self.inputs[0]["shape"]  # [1, H, W, C]
        self.input_dtype = self.inputs[0]["dtype"]
        self.h_in, self.w_in = int(self.input_shape[1]), int(self.input_shape[2])

        # Detect model architecture based on output tensor signatures
        out0_shape = self.outputs[0]["shape"]
        if len(out0_shape) == 3 and out0_shape[1] > 1000:
            self.model_type = "yolo"
            self.names = {i: name for i, name in enumerate(COCO_CLASSES)}
            self.box_scale, self.box_zero = self.outputs[0]["quantization"]
            self.score_scale, self.score_zero = self.outputs[1]["quantization"]
        else:
            self.model_type = "ssd_efficientdet"
            self.names = COCO_CLASSES_90
            self.box_scale, self.box_zero = self.outputs[0].get("quantization", (0.0, 0))
            # In SSD/EfficientDet: outputs[1] is classes, outputs[2] is scores
            self.score_scale, self.score_zero = self.outputs[2].get("quantization", (0.0, 0)) if len(self.outputs) > 2 else (0.0, 0)

    def predict(self, frame: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """Executes inference and returns filtered detection dictionaries fitting the specific model architecture."""
        h0, w0 = frame.shape[:2]
        resized = cv2.resize(frame, (self.w_in, self.h_in))
        inp = np.expand_dims(resized, axis=0).astype(self.input_dtype)

        self.interpreter.set_tensor(self.inputs[0]["index"], inp)
        self.interpreter.invoke()

        if self.model_type == "ssd_efficientdet":
            # Output 0: boxes [1, N, 4] normalized [ymin, xmin, ymax, xmax] in [0, 1]
            # Output 1: classes [1, N]
            # Output 2: scores [1, N]
            raw_boxes = self.interpreter.get_tensor(self.outputs[0]["index"])[0]
            raw_classes = self.interpreter.get_tensor(self.outputs[1]["index"])[0]
            raw_scores = self.interpreter.get_tensor(self.outputs[2]["index"])[0]

            if self.score_scale > 0:
                scores = (raw_scores.astype(np.float32) - self.score_zero) * self.score_scale
            else:
                scores = raw_scores.astype(np.float32)

            mask = scores >= conf_thresh
            if not np.any(mask):
                return []

            v_boxes = raw_boxes[mask]
            v_scores = scores[mask]
            v_classes = raw_classes[mask]

            detections = []
            for b, s, c in zip(v_boxes, v_scores, v_classes):
                ymin = max(0, min(h0, int(b[0] * h0)))
                xmin = max(0, min(w0, int(b[1] * w0)))
                ymax = max(0, min(h0, int(b[2] * h0)))
                xmax = max(0, min(w0, int(b[3] * w0)))
                bw = max(1, xmax - xmin)
                bh = max(1, ymax - ymin)
                cls_id = int(c)
                lbl = self.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "label": lbl,
                    "confidence": float(s),
                    "bbox": (ymin, xmin, ymax, xmax),
                    "center": (xmin + bw // 2, ymin + bh // 2),
                    "box_w": bw,
                    "box_h": bh,
                })
            return detections

        # Default YOLO11 dense anchor branch
        raw_boxes = self.interpreter.get_tensor(self.outputs[0]["index"])[0]
        raw_scores = self.interpreter.get_tensor(self.outputs[1]["index"])[0]
        raw_classes = self.interpreter.get_tensor(self.outputs[2]["index"])[0]

        if self.score_scale > 0:
            scores = (raw_scores.astype(np.float32) - self.score_zero) * self.score_scale
        else:
            scores = raw_scores.astype(np.float32)

        if self.box_scale > 0:
            boxes = (raw_boxes.astype(np.float32) - self.box_zero) * self.box_scale
        else:
            boxes = raw_boxes.astype(np.float32)

        mask = scores >= conf_thresh
        if not np.any(mask):
            return []

        v_boxes = boxes[mask]
        v_scores = scores[mask]
        v_classes = raw_classes[mask]

        scale_x = w0 / float(self.w_in)
        scale_y = h0 / float(self.h_in)

        cv_boxes = []
        for b in v_boxes:
            x1, y1, x2, y2 = b
            cv_boxes.append([
                int(x1 * scale_x),
                int(y1 * scale_y),
                int((x2 - x1) * scale_x),
                int((y2 - y1) * scale_y)
            ])

        indices = cv2.dnn.NMSBoxes(cv_boxes, v_scores.tolist(), score_threshold=conf_thresh, nms_threshold=iou_thresh)
        if len(indices) == 0:
            return []

        detections = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            bx, by, bw, bh = cv_boxes[idx]
            ymin = max(0, min(h0, by))
            xmin = max(0, min(w0, bx))
            ymax = max(0, min(h0, by + bh))
            xmax = max(0, min(w0, bx + bw))
            cls_id = int(v_classes[idx])
            lbl = self.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class_id": cls_id,
                "label": lbl,
                "confidence": float(v_scores[idx]),
                "bbox": (ymin, xmin, ymax, xmax),
                "center": (xmin + bw // 2, ymin + bh // 2),
                "box_w": bw,
                "box_h": bh,
            })
        return detections


class YOLOEVisionAdapter(BaseVisionAdapter):
    """Real-time YOLOE / LiteRT vision adapter for autonomous object categorization and visual servoing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        imgsz: int = 320,
        target_classes: Optional[List[str]] = None,
        pickable_classes: Optional[List[str]] = None,
        device: str = "cpu",
        enforce_limitations: bool = False,
        num_threads: int = 4
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.enforce_limitations = bool(enforce_limitations)
        self.target_classes = list(target_classes) if target_classes else list(DEFAULT_TARGET_CLASSES)
        # Accept both pickable_classes (new) and target_classes (legacy compat)
        pc = pickable_classes or target_classes
        self.pickable_classes = list(pc) if pc else list(DEFAULT_PICKABLE_CLASSES)
        # Keep legacy alias for backward compatibility
        self.target_classes = self.pickable_classes
        self.num_threads = int(num_threads)
        self.model = None
        self.model_loaded = False
        self.is_pre_fused = False
        self.backend_type = "PyTorch"
        self.color_detection_enabled = True

        if model_path is None:
            # Check for LiteRT models first, followed by PyTorch models
            m_tflite_s = os.path.join(BASE_DIR, "models", "yolo11s.tflite")
            m_tflite_n = os.path.join(BASE_DIR, "models", "yolo11n.tflite")
            m_tflite_l = os.path.join(BASE_DIR, "models", "yolo11l.tflite")
            m11s = os.path.join(BASE_DIR, "models", "yoloe-11s-seg.pt")
            m26l = os.path.join(BASE_DIR, "models", "yoloe-26l-seg.pt")
            m26s = os.path.join(BASE_DIR, "models", "yoloe-26s-seg.pt")
            m26n = os.path.join(BASE_DIR, "models", "yoloe-26n-seg.pt")

            if os.path.exists(m_tflite_s):
                model_path = m_tflite_s
            elif os.path.exists(m_tflite_n):
                model_path = m_tflite_n
            elif os.path.exists(m_tflite_l):
                model_path = m_tflite_l
            elif os.path.exists(m11s):
                model_path = m11s
            elif os.path.exists(m26l):
                model_path = m26l
            elif os.path.exists(m26s):
                model_path = m26s
            elif os.path.exists(m26n):
                model_path = m26n
            else:
                pf_path = os.path.join(BASE_DIR, "models", "yoloe-26n-seg-pf.pt")
                if os.path.exists(pf_path):
                    model_path = pf_path

        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Initializes and warms up the YOLOE / LiteRT model."""
        is_tflite = bool(self.model_path and self.model_path.endswith(".tflite"))
        self.backend_type = "LiteRT" if is_tflite else "PyTorch"

        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[YOLOE] Notice: Model weights not found at {self.model_path}. Running in mock-fallback mode.")
            self.model = None
            self.model_loaded = False
            return

        try:
            t0 = time.time()
            if is_tflite:
                print(f"[YOLOE/LiteRT] Loading LiteRT (.tflite) model from: {self.model_path} with {self.num_threads} threads...")
                self.model = LiteRTEngine(self.model_path, num_threads=self.num_threads)
                self.model_loaded = True
                self.is_pre_fused = True
                # Warmup inference with blank frame
                warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.model.predict(warmup_frame, conf_thresh=self.confidence_threshold)
            else:
                from ultralytics import YOLOE
                print(f"[YOLOE/PyTorch] Loading model from: {self.model_path} on {self.device}...")
                self.model = YOLOE(self.model_path)
                self.model_loaded = True

                # Pre-fused check
                if "-pf" in os.path.basename(self.model_path):
                    self.is_pre_fused = True
                    print(f"[YOLOE] Pre-fused model detected with {len(getattr(self.model, 'names', {}))} classes.")
                else:
                    self._configure_classes(self.target_classes)

                # Warmup inference with blank frame
                warmup_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                self.model(warmup_frame, imgsz=self.imgsz, device=self.device, verbose=False)

            print(f"[YOLOE/{self.backend_type}] Ready in {time.time() - t0:.2f}s (conf={self.confidence_threshold})")
        except Exception as e:
            print(f"[YOLOE] Warning: Failed to load {self.backend_type} model: {e}")
            self.model = None
            self.model_loaded = False

    def _configure_classes(self, classes: List[str]):
        """Sets open-vocabulary classification prompts (for PyTorch models)."""
        if not self.model or self.is_pre_fused or self.backend_type == "LiteRT":
            return
        try:
            clean_classes = [c.strip().replace(" ", "_") for c in classes if c.strip()]
            self.model.set_classes(clean_classes)
            self.target_classes = clean_classes
            print(f"[YOLOE] Open-vocabulary classes set ({len(clean_classes)}): {clean_classes}")
        except Exception as e:
            print(f"[YOLOE] Warning setting classes: {e}")

    def set_target_classes(self, classes: List[str]):
        """Dynamically update target classes at runtime."""
        self.target_classes = [c.strip().replace(" ", "_") for c in classes if c.strip()]
        """Dynamically update pickable classes at runtime (legacy alias)."""
        self.set_pickable_classes(classes)

    def set_pickable_classes(self, classes: List[str]):
        """Dynamically update the list of classes eligible for autonomous grasp."""
        clean = [c.strip().replace(" ", "_") for c in classes if c.strip()]
        self.pickable_classes = clean
        self.target_classes = clean  # keep legacy alias in sync
        if self.backend_type != "LiteRT":
            self._configure_classes(classes)

    def categorize(self, frame: np.ndarray) -> DetectionResult:
        """Analyzes frame, detects graspable target object, and returns DetectionResult."""
        if frame is None or not self.model_loaded or self.model is None:
            return DetectionResult(detected=False, raw_response="YOLOE: Model offline or invalid frame")

        h, w = frame.shape[:2]
        t0 = time.time()

        try:
            if self.backend_type == "LiteRT":
                detections = self.model.predict(frame, conf_thresh=self.confidence_threshold)
                elapsed = time.time() - t0

                if not detections:
                    return DetectionResult(detected=False, raw_response=f"LiteRT: No detections ({elapsed:.3f}s)")

                best_det = None
                best_score = -1.0

                for det in detections:
                    label = det["label"].lower()
                    conf = det["confidence"]
                    ymin, xmin, ymax, xmax = det["bbox"]
                    box_w = det["box_w"]
                    box_h = det["box_h"]

                    # Always check if this class is in the user's pickable list
                    is_non_pickable = any(npk in label for npk in NON_PICKABLE_CLASSES)
                    in_pickable_list = any(pc.lower() in label or label in pc.lower() for pc in self.pickable_classes)
                    pickable = in_pickable_list and not is_non_pickable

                    too_large = (box_w > int(w * 0.75)) or (box_h > int(h * 0.85))
                    too_small = (box_w < 15) or (box_h < 15)

                    if self.enforce_limitations:
                        pickable = (not is_non_pickable) and (not too_large) and (not too_small)
                    else:
                        pickable = (not too_large) and (not too_small)

                    # Boost score for pickable objects so they get priority
                    score = conf + (0.5 if pickable else 0.0)

                    if score > best_score:
                        best_score = score
                        best_det = det
                        best_det["pickable"] = pickable

                if best_det is None:
                    return DetectionResult(detected=False, raw_response="LiteRT: No candidates passed thresholds")

                coords = best_det["bbox"]
                label = best_det["label"]
                conf = best_det["confidence"]
                pickable = best_det["pickable"]
                display_category = label.replace("_", " ").title()

                color_info = detect_dominant_color(frame, coords) if self.color_detection_enabled else ColorInfo(name=label)

                ymin, xmin, ymax, xmax = coords
                mask_poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                mask_area = int((xmax - xmin) * (ymax - ymin))

                return DetectionResult(
                    detected=True,
                    category=display_category,
                    material_color=color_info.name,
                    pickable=pickable,
                    confidence=round(conf, 3),
                    bounding_box=coords,
                    mask_polygon=mask_poly,
                    mask_area=mask_area,
                    raw_response=f'{{"model": "LiteRT", "label": "{label}", "color": "{color_info.name}", "color_hex": "{color_info.hex_code}", "confidence": {conf:.3f}, "latency_s": {elapsed:.3f}}}'
                )

            else:
                # PyTorch Ultralytics backend
                results = self.model(frame, imgsz=self.imgsz, conf=self.confidence_threshold, device=self.device, verbose=False)
                elapsed = time.time() - t0

                if not results or len(results) == 0:
                    return DetectionResult(detected=False, raw_response=f"YOLOE: No detections ({elapsed:.3f}s)")

                res = results[0]
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    return DetectionResult(detected=False, raw_response=f"YOLOE: 0 objects detected ({elapsed:.3f}s)")

                best_box = None
                best_conf = 0.0
                best_label = "unknown"
                best_pickable = False
                best_coords = (0, 0, 0, 0)
                names = getattr(self.model, "names", {})

                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = float(box.conf[0].cpu().item()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
                    if conf < self.confidence_threshold:
                        continue

                    cls_id = int(box.cls[0].cpu().item()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
                    label = names.get(cls_id, str(cls_id)).lower()

                    xyxy = box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0], dtype=int)
                    xmin = max(0, min(w, int(xyxy[0])))
                    ymin = max(0, min(h, int(xyxy[1])))
                    xmax = max(0, min(w, int(xyxy[2])))
                    ymax = max(0, min(h, int(xyxy[3])))
                    box_w = xmax - xmin
                    box_h = ymax - ymin

                    # Always check if this class is in the user's pickable list
                    is_non_pickable = any(npk in label for npk in NON_PICKABLE_CLASSES)
                    in_pickable_list = any(pc.lower() in label or label in pc.lower() for pc in self.pickable_classes)
                    pickable = in_pickable_list and not is_non_pickable

                    too_large = (box_w > int(w * 0.75)) or (box_h > int(h * 0.85))
                    too_small = (box_w < 15) or (box_h < 15)

                    if self.enforce_limitations:
                        pickable = (not is_non_pickable) and (not too_large) and (not too_small)
                    else:
                        pickable = (not too_large) and (not too_small)

                    # Boost score for pickable objects so they get priority
                    score = conf + (0.5 if pickable else 0.0)

                    if score > best_conf:
                        best_conf = score
                        best_box = box
                        best_idx = i
                        best_label = label
                        best_pickable = pickable
                        best_coords = (ymin, xmin, ymax, xmax)

                if best_box is None:
                    return DetectionResult(detected=False, raw_response="YOLOE: No candidates passed thresholds")

                raw_conf = float(best_box.conf[0].cpu().item()) if hasattr(best_box.conf[0], "cpu") else float(best_box.conf[0])
                display_category = best_label.replace("_", " ").title()
                color_info = detect_dominant_color(frame, best_coords) if self.color_detection_enabled else ColorInfo(name=best_label)

                best_mask_poly = None
                best_mask_area = 0
                if getattr(res, "masks", None) is not None and getattr(res.masks, "xy", None) is not None and len(res.masks.xy) > best_idx:
                    poly_arr = res.masks.xy[best_idx]
                    if poly_arr is not None and len(poly_arr) >= 3:
                        best_mask_poly = [(int(pt[0]), int(pt[1])) for pt in poly_arr]
                        best_mask_area = int(cv2.contourArea(np.array(best_mask_poly, dtype=np.int32)))

                return DetectionResult(
                    detected=True,
                    category=display_category,
                    material_color=color_info.name,
                    pickable=best_pickable,
                    confidence=round(raw_conf, 3),
                    bounding_box=best_coords,
                    mask_polygon=best_mask_poly,
                    mask_area=best_mask_area,
                    raw_response=f'{{"model": "PyTorch", "label": "{best_label}", "color": "{color_info.name}", "color_hex": "{color_info.hex_code}", "confidence": {raw_conf:.3f}, "latency_s": {elapsed:.3f}}}'
                )
        except Exception as e:
            return DetectionResult(detected=False, raw_response=f"Vision inference error: {e}")

    def detect_all(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detects and returns all candidate objects in the frame."""
        if frame is None or not self.model_loaded or self.model is None:
            return []

        h, w = frame.shape[:2]
        detections = []
        try:
            if self.backend_type == "LiteRT":
                raw_dets = self.model.predict(frame, conf_thresh=self.confidence_threshold)
                for det in raw_dets:
                    label = det["label"]
                    conf = det["confidence"]
                    coords = det["bbox"]
                    color_info = detect_dominant_color(frame, coords) if self.color_detection_enabled else ColorInfo(name=label)
                    pickable = True if not self.enforce_limitations else not any(npk in label for npk in NON_PICKABLE_CLASSES)

                    is_non_pickable = any(npk in label.lower() for npk in NON_PICKABLE_CLASSES)
                    in_pickable_list = any(pc.lower() in label.lower() or label.lower() in pc.lower() for pc in self.pickable_classes)
                    ymin, xmin, ymax, xmax = coords
                    mask_poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                    mask_area = int((xmax - xmin) * (ymax - ymin))
                    detections.append(DetectionResult(
                        detected=True,
                        category=label.replace("_", " ").title(),
                        material_color=color_info.name,
                        pickable=pickable,
                        confidence=round(conf, 3),
                        bounding_box=coords,
                        mask_polygon=mask_poly,
                        mask_area=mask_area,
                        raw_response=f'{{"model": "LiteRT", "label": "{label}", "color": "{color_info.name}", "color_hex": "{color_info.hex_code}", "conf": {conf:.3f}}}'
                    ))
            else:
                results = self.model(frame, imgsz=self.imgsz, conf=self.confidence_threshold, device=self.device, verbose=False)
                if not results or len(results) == 0:
                    return []
                res = results[0]
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    return []

                names = getattr(self.model, "names", {})
                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = float(box.conf[0].cpu().item()) if hasattr(box.conf[0], "cpu") else float(box.conf[0])
                    if conf < self.confidence_threshold:
                        continue

                    cls_id = int(box.cls[0].cpu().item()) if hasattr(box.cls[0], "cpu") else int(box.cls[0])
                    label = names.get(cls_id, str(cls_id)).lower()
                    xyxy = box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0], dtype=int)
                    xmin = max(0, min(w, int(xyxy[0])))
                    ymin = max(0, min(h, int(xyxy[1])))
                    xmax = max(0, min(w, int(xyxy[2])))
                    ymax = max(0, min(h, int(xyxy[3])))
                    coords = (ymin, xmin, ymax, xmax)

                    color_info = detect_dominant_color(frame, coords) if self.color_detection_enabled else ColorInfo(name=label)
                    pickable = True if not self.enforce_limitations else not any(npk in label for npk in NON_PICKABLE_CLASSES)
                    is_non_pickable = any(npk in label.lower() for npk in NON_PICKABLE_CLASSES)
                    in_pickable_list = any(pc.lower() in label.lower() or label.lower() in pc.lower() for pc in self.pickable_classes)
                    pickable = in_pickable_list and not is_non_pickable

                    mask_poly = None
                    mask_area = 0
                    if getattr(res, "masks", None) is not None and getattr(res.masks, "xy", None) is not None and len(res.masks.xy) > i:
                        poly_arr = res.masks.xy[i]
                        if poly_arr is not None and len(poly_arr) >= 3:
                            mask_poly = [(int(pt[0]), int(pt[1])) for pt in poly_arr]
                            mask_area = int(cv2.contourArea(np.array(mask_poly, dtype=np.int32)))

                    detections.append(DetectionResult(
                        detected=True,
                        category=label.replace("_", " ").title(),
                        material_color=color_info.name,
                        pickable=pickable,
                        confidence=round(conf, 3),
                        bounding_box=coords,
                        mask_polygon=mask_poly,
                        mask_area=mask_area,
                        raw_response=f'{{"model": "PyTorch", "label": "{label}", "color": "{color_info.name}", "color_hex": "{color_info.hex_code}", "conf": {conf:.3f}}}'
                    ))
        except Exception as e:
            print(f"[YOLOE] Error in detect_all: {e}")
        return detections

    def verify_grasp_alignment(
        self,
        frame: np.ndarray,
        sweet_spot: Tuple[int, int, int, int]
    ) -> bool:
        """Verifies whether the target object's landing centroid is inside the gripper sweet spot."""
        res = self.categorize(frame)
        if not res.detected:
            return False
        if self.enforce_limitations and not res.pickable:
            return False

        bx, by, bw, bh = sweet_spot
        cx, cy = res.center
        in_x = (bx - bw // 2) <= cx <= (bx + bw // 2)
        in_y = (by - bh // 2) <= cy <= (by + bh // 2)
        return in_x and in_y
