import os
import threading
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import yaml
from src.core.events import DetectionResult, TelemetryData


class RobotContext:
    """Thread-safe context storing robot telemetry, vision state, and settings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._lock = threading.RLock()
        if config is None:
            robot_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml"))
            vision_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/vision_config.yaml"))
            r_cfg = {}
            v_cfg = {}
            if os.path.exists(robot_cfg_path):
                try:
                    with open(robot_cfg_path, "r") as f:
                        r_cfg = yaml.safe_load(f) or {}
                except Exception:
                    pass
            if os.path.exists(vision_cfg_path):
                try:
                    with open(vision_cfg_path, "r") as f:
                        v_cfg = yaml.safe_load(f) or {}
                except Exception:
                    pass
            self.config = {"robot": r_cfg, "vision": v_cfg}
        else:
            self.config = config

        vision_cfg = self.config.get("vision", {})
        cam_cfg = vision_cfg.get("camera", {})
        spot = vision_cfg.get("grasp_sweet_spot", {})

        # Robot State Machine State
        self.current_state: str = "IDLE"
        self.state_start_time: float = time.time()

        # Latest camera frame (BGR format)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_time: float = 0.0
        self.frame_width: int = cam_cfg.get("width", 640)
        self.frame_height: int = cam_cfg.get("height", 480)

        # Latest detection & tracker ROI
        self.latest_detection: Optional[DetectionResult] = None
        self.tracked_roi: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, width, height)
        self.is_tracking: bool = False

        # Calibrated Grasp Sweet Spot (pixel coordinates from vision_config.yaml)
        self.sweet_spot_x: int = spot.get("center_x", 324)
        self.sweet_spot_y: int = spot.get("center_y", 247)
        self.sweet_spot_w: int = spot.get("box_width", 256)
        self.sweet_spot_h: int = spot.get("box_height", 231)
        self.align_tolerance_x: int = spot.get("align_tolerance_x", 20)
        self.align_tolerance_y: int = spot.get("align_tolerance_y", 25)

        # Telemetry
        self.telemetry = TelemetryData()

        # Statistics
        self.fps: float = 0.0
        self.inference_latency_s: float = 0.0

    def update_frame(self, frame: np.ndarray):
        """Update latest frame and dimensions safely."""
        with self._lock:
            self.latest_frame = frame
            self.latest_frame_time = time.time()
            if frame is not None and frame.ndim >= 2:
                self.frame_height, self.frame_width = frame.shape[:2]

    def get_frame(self) -> Optional[np.ndarray]:
        """Retrieve copy of latest frame."""
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def set_state(self, new_state: str):
        """Update current autonomous state."""
        with self._lock:
            self.current_state = new_state
            self.state_start_time = time.time()

    def get_state(self) -> str:
        """Get current state name."""
        with self._lock:
            return self.current_state

    def update_telemetry(self, data: TelemetryData):
        """Update telemetry snapshot."""
        with self._lock:
            self.telemetry = data

    def get_telemetry(self) -> TelemetryData:
        """Get copy of telemetry data."""
        with self._lock:
            return self.telemetry

    def update_detection(self, detection: DetectionResult):
        """Update latest VLM categorization result."""
        with self._lock:
            self.latest_detection = detection

    def get_detection(self) -> Optional[DetectionResult]:
        """Get latest detection."""
        with self._lock:
            return self.latest_detection

    def set_tracked_roi(self, roi: Optional[Tuple[int, int, int, int]]):
        """Set or clear the active tracker ROI (xmin, ymin, w, h)."""
        with self._lock:
            self.tracked_roi = roi
            self.is_tracking = roi is not None

    def get_tracked_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Get active tracker ROI."""
        with self._lock:
            return self.tracked_roi

    def set_sweet_spot(self, x: int, y: int, w: int, h: int):
        """Update the calibrated grasp sweet spot coordinates."""
        with self._lock:
            self.sweet_spot_x = x
            self.sweet_spot_y = y
            self.sweet_spot_w = w
            self.sweet_spot_h = h
