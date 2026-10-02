"""High-throughput threaded V4L2 / OpenCV camera adapter for A4Tech USB Cam."""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.adapters.camera.base import BaseCameraAdapter


class V4L2CameraAdapter(BaseCameraAdapter):
    """Threaded OpenCV capture ensuring zero-latency, fresh frames for visual servoing."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        fourcc: str = "MJPG"
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.fourcc = fourcc

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.RLock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None

        self._frame_count = 0
        self._fps_start_time = time.time()
        self._measured_fps = 0.0

    def start(self) -> bool:
        with self._lock:
            if self.is_opened():
                return True

            print(f"[Camera] Initializing /dev/video{self.device_index} ({self.width}x{self.height} @ {self.target_fps}fps)...")
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)

            if not self._cap.isOpened():
                # Fallback to default backend if V4L2 backend flag fails
                self._cap = cv2.VideoCapture(self.device_index)

            if not self._cap.isOpened():
                print(f"[Camera] Failed to open /dev/video{self.device_index}")
                return False

            # Configure properties
            if self.fourcc and len(self.fourcc) == 4:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read initial frame to verify
            ret, frame = self._cap.read()
            if not ret or frame is None:
                print("[Camera] Warning: First frame read failed.")
            else:
                self._latest_frame = frame
                self.height, self.width = frame.shape[:2]

            self._running = True
            self._fps_start_time = time.time()
            self._frame_count = 0
            self._capture_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._capture_thread.start()

            print(f"[Camera] Video stream active: {self.width}x{self.height}")
            return True

    def stop(self):
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = None
            self._latest_frame = None

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened() and self._running

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def get_fps(self) -> float:
        return self._measured_fps

    def _grab_loop(self):
        """Continuously grab frames in background thread to drain hardware buffer."""
        while self._running:
            if not (self._cap and self._cap.isOpened()):
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                self._frame_count += 1

                elapsed = time.time() - self._fps_start_time
                if elapsed >= 1.0:
                    self._measured_fps = round(self._frame_count / elapsed, 1)
                    self._frame_count = 0
                    self._fps_start_time = time.time()
            else:
                time.sleep(0.01)
