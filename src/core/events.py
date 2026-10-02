"""Event definitions and Pub/Sub EventBus for ErovoutikaGrab."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class DetectionResult:
    """Structured output from VLM or vision adapter."""
    detected: bool = False
    category: str = "unknown"
    material_color: str = ""
    pickable: bool = False
    confidence: float = 0.0
    # Bounding box in [ymin, xmin, ymax, xmax] normalized (0.0 to 1.0) or pixel coords
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mask_polygon: Optional[List[Tuple[int, int]]] = None
    mask_area: int = 0
    raw_response: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def center(self) -> Tuple[int, int]:
        """Returns (cx, cy) pixel coordinates."""
        ymin, xmin, ymax, xmax = self.bounding_box
        return (xmin + xmax) // 2, (ymin + ymax) // 2

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Returns (cx, ymax) pixel coordinates (ground contact point)."""
        ymin, xmin, ymax, xmax = self.bounding_box
        return (xmin + xmax) // 2, ymax


@dataclass
class TelemetryData:
    """Live telemetry from Arduino microcontroller."""
    left_pwm: int = 0
    right_pwm: int = 0
    s1_shoulder: int = 20
    s2_elbow: int = 70
    s3_gripper: int = 170
    macro_active: bool = False
    connected: bool = False
    ping_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class StateChangeEvent:
    """Robot state transition event."""
    old_state: str
    new_state: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe event dispatcher for decoupled communication."""

    def __init__(self):
        self._subscribers: Dict[type, List[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: type, callback: Callable[[Any], None]):
        """Subscribe a callback to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: type, callback: Callable[[Any], None]):
        """Unsubscribe a callback."""
        with self._lock:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)

    def publish(self, event: Any):
        """Publish an event to all subscribers."""
        event_type = type(event)
        callbacks = []
        with self._lock:
            if event_type in self._subscribers:
                callbacks = list(self._subscribers[event_type])

        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[EventBus] Error in callback {cb} for event {event_type.__name__}: {e}")
