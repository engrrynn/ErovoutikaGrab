"""Mock vision adapter returning simulated object detections for automated tests."""

from typing import Tuple
import numpy as np
from src.adapters.vision.base import BaseVisionAdapter
from src.core.events import DetectionResult


class MockVisionAdapter(BaseVisionAdapter):
    """Simulates dynamic object categorization for unit testing and offline development."""

    def __init__(
        self,
        mock_category: str = "electronic_relay",
        mock_color: str = "blue plastic",
        mock_pickable: bool = True,
        mock_confidence: float = 0.94,
        mock_bbox: Tuple[int, int, int, int] = (180, 280, 240, 360)
    ):
        self.mock_category = mock_category
        self.mock_color = mock_color
        self.mock_pickable = mock_pickable
        self.mock_confidence = mock_confidence
        self.mock_bbox = mock_bbox
        self.should_detect = True

    def categorize(self, frame: np.ndarray) -> DetectionResult:
        if not self.should_detect or frame is None:
            return DetectionResult(detected=False, raw_response="Mock: No detection")

        return DetectionResult(
            detected=True,
            category=self.mock_category,
            material_color=self.mock_color,
            pickable=self.mock_pickable,
            confidence=self.mock_confidence,
            bounding_box=self.mock_bbox,
            raw_response='{"detected": true, "category": "' + self.mock_category + '"}'
        )

    def verify_grasp_alignment(
        self,
        frame: np.ndarray,
        sweet_spot: Tuple[int, int, int, int]
    ) -> bool:
        bx, by, bw, bh = sweet_spot
        ymin, xmin, ymax, xmax = self.mock_bbox
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        return (bx - bw // 2 <= cx <= bx + bw // 2) and (by - bh // 2 <= cy <= by + bh // 2)
