"""Observability hooks: structured events and callbacks."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

class EventType(enum.Enum):
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_SUCCESS = "retry_success"
    RETRY_EXHAUSTED = "retry_exhausted"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    CIRCUIT_CLOSED = "circuit_closed"
    FALLBACK_TRIGGERED = "fallback_triggered"
    DEAD_LETTER = "dead_letter"


@dataclass
class RetryEvent:
    """Structured event emitted during retry operations."""
    type: EventType
    function_name: str
    timestamp: float = field(default_factory=time.time)
    attempt: int = 0
    wait_time: float = 0.0
    error: Optional[BaseException] = None
    error_type: Optional[str] = None
    circuit_state: Any = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.error and self.error_type is None:
            self.error_type = type(self.error).__name__


class EventDispatcher:
    """Dispatches retry events to registered callbacks."""

    def __init__(self, callback: Callable[[RetryEvent], Any] | None = None) -> None:
        self._callbacks: list[Callable[[RetryEvent], Any]] = []
        if callback is not None:
            self._callbacks.append(callback)

    def emit(self, event: RetryEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Don't let event handlers break the retry loop

    def add_callback(self, callback: Callable[[RetryEvent], Any]) -> None:
        self._callbacks.append(callback)
