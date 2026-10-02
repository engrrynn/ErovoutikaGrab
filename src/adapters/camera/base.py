"""Abstract base class for camera capture adapters."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseCameraAdapter(ABC):
    """Abstract interface for video capture devices."""

    @abstractmethod
    def start(self) -> bool:
        """Initialize and start video capture."""
        pass

    @abstractmethod
    def stop(self):
        """Release camera resource."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is currently capturing."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent fresh frame (BGR format)."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height)."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return measured capture frame rate."""
        pass
