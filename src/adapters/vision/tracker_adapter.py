"""High-speed 30 FPS Visual Servoing Object Tracker using OpenCV."""

from typing import Optional, Tuple
import cv2
import numpy as np


class VisualServoingTracker:
    """Tracks an object bounding box frame-by-frame at 30 FPS to guide motor steering."""

    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker_type = tracker_type.upper()
        self._tracker: Optional[Any] = None
        self._is_tracking = False
        self._last_roi: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, w, h)

    def _create_tracker(self):
        """Factory for OpenCV tracker instances."""
        if hasattr(cv2, "TrackerCSRT"):
            return cv2.TrackerCSRT.create()
        elif hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerMIL"):
            return cv2.TrackerMIL.create()
        elif hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        raise RuntimeError("No suitable OpenCV Tracker implementation found.")

    def start_tracking(self, frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> bool:
        """Initialize tracker with [ymin, xmin, ymax, xmax] bounding box."""
        if frame is None or frame.size == 0:
            return False

        ymin, xmin, ymax, xmax = bbox_xyxy
        w = max(10, xmax - xmin)
        h = max(10, ymax - ymin)
        roi = (int(xmin), int(ymin), int(w), int(h))

        try:
            self._tracker = self._create_tracker()
            self._tracker.init(frame, roi)
            self._is_tracking = True
            self._last_roi = roi
            return True
        except Exception as e:
            print(f"[Tracker] Initialization error: {e}")
            self._is_tracking = False
            self._last_roi = None
            return False

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Update tracker on a new frame.
        
        Returns:
            (success, (xmin, ymin, width, height))
        """
        if not self._is_tracking or self._tracker is None or frame is None:
            return False, None

        try:
            success, box = self._tracker.update(frame)
            if success:
                xmin, ymin, w, h = [int(v) for v in box]
                self._last_roi = (xmin, ymin, w, h)
                return True, self._last_roi
            else:
                self._is_tracking = False
                return False, None
        except Exception as e:
            print(f"[Tracker] Update error: {e}")
            self._is_tracking = False
            return False, None

    def stop(self):
        """Stop tracking and reset."""
        self._is_tracking = False
        self._tracker = None
        self._last_roi = None

    @property
    def is_tracking(self) -> bool:
        return self._is_tracking

    @property
    def current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_roi

    @property
    def target_center(self) -> Optional[Tuple[int, int]]:
        """Returns (cx, cy) pixel coordinates."""
        if self._last_roi is None:
            return None
        xmin, ymin, w, h = self._last_roi
        return xmin + w // 2, ymin + h // 2

    @property
    def ground_contact_point(self) -> Optional[Tuple[int, int]]:
        """Returns (cx, y_bottom) where the object contacts the surface."""
        if self._last_roi is None:
            return None
        xmin, ymin, w, h = self._last_roi
        return xmin + w // 2, ymin + h
