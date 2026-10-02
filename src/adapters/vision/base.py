"""Abstract base class for vision and categorization adapters."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from src.core.events import DetectionResult


class BaseVisionAdapter(ABC):
    """Abstract interface for object detection and categorization."""

    @abstractmethod
    def categorize(self, frame: np.ndarray) -> DetectionResult:
        """Analyze frame, detect small graspable object, categorize it, and return bounding box."""
        pass

    @abstractmethod
    def verify_grasp_alignment(
        self,
        frame: np.ndarray,
        sweet_spot: Tuple[int, int, int, int]
    ) -> bool:
        """Verify whether target object is positioned directly within gripper jaw sweet spot."""
        pass
