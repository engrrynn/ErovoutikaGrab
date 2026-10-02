"""Mock camera adapter generating synthetic video frames for testing."""

import math
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.adapters.camera.base import BaseCameraAdapter


class MockCameraAdapter(BaseCameraAdapter):
    """Generates synthetic camera frames with a moving small target object."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._start_time = time.time()

        # Simulated object motion parameters
        self.obj_x = 320.0
        self.obj_y = 200.0
        self.obj_size = 40

    def start(self) -> bool:
        self._running = True
        self._start_time = time.time()
        return True

    def stop(self):
        self._running = False

    def is_opened(self) -> bool:
        return self._running

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def get_fps(self) -> float:
        return float(self.fps)

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None

        # Create workbench floor background
        frame = np.full((self.height, self.width, 3), 60, dtype=np.uint8)

        # Draw tabletop grid lines
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (75, 75, 75), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (75, 75, 75), 1)

        # Gentle floating trajectory for the simulated object
        t = time.time() - self._start_time
        cx = int(self.width / 2 + 80 * math.sin(t * 0.5))
        cy = int(220 + 60 * math.cos(t * 0.3))

        # Draw a simulated small blue object (e.g. AA battery / small block)
        top_left = (cx - self.obj_size // 2, cy - self.obj_size // 2)
        bottom_right = (cx + self.obj_size // 2, cy + self.obj_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 120, 30), -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
        cv2.putText(frame, "SIM_OBJECT", (top_left[0], top_left[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame
