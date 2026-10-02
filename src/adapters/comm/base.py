"""Abstract base class for robot communication adapters."""

from abc import ABC, abstractmethod
from typing import Optional
from src.core.events import TelemetryData


class BaseCommAdapter(ABC):
    """Abstract interface for communicating with the Arduino robot controller."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the microcontroller."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is alive."""
        pass

    @abstractmethod
    def send_drive(self, left_pwm: int, right_pwm: int) -> bool:
        """Command continuous motor drive."""
        pass

    @abstractmethod
    def send_stop(self) -> bool:
        """Emergency stop all motors."""
        pass

    @abstractmethod
    def send_nudge(self, direction: str, duration_ms: int = 70, pwm: int = 210) -> bool:
        """Command discrete anti-stiction pulse."""
        pass

    @abstractmethod
    def send_servos(self, s1: int, s2: int, s3: int, pulse: Optional[bool] = None, **kwargs) -> bool:
        """Command specific target angles for shoulder, elbow, gripper."""
        pass

    @abstractmethod
    def send_macro(self, name: str) -> bool:
        """Trigger an arm macro (PICK, DOWN, UP, OPEN, CLOSE, DEFAULT)."""
        pass

    @abstractmethod
    def ping(self) -> float:
        """Measure round-trip ping time in milliseconds. Returns -1 on failure."""
        pass

    @abstractmethod
    def get_latest_telemetry(self) -> TelemetryData:
        """Retrieve latest telemetry snapshot."""
        pass
