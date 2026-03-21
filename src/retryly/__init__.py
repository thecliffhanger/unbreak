"""retryly — Smart retry library with adaptive backoff, circuit breaker, fallback chains, and dead letter queues."""

from retryly.retry import retry, RetryResult
from retryly.backoff import BackoffStrategy, FixedBackoff, ExponentialBackoff, AdaptiveBackoff
from retryly.circuit import CircuitBreaker, CircuitState
from retryly.errors import is_retryable, register_retryable, unregister_retryable
from retryly.budget import BudgetManager
from retryly.fallback import run_fallback_chain
from retryly.dead_letter import DeadLetterQueue, replay
from retryly.jitter import apply_jitter, coordinated_jitter
from retryly.events import EventType, RetryEvent
from retryly.history import RetryHistory, RetryTimeline

__version__ = "0.1.0"
__all__ = [
    "retry", "RetryResult",
    "BackoffStrategy", "FixedBackoff", "ExponentialBackoff", "AdaptiveBackoff",
    "CircuitBreaker", "CircuitState",
    "is_retryable", "register_retryable", "unregister_retryable",
    "BudgetManager",
    "run_fallback_chain",
    "DeadLetterQueue", "replay",
    "apply_jitter", "coordinated_jitter",
    "EventType", "RetryEvent",
    "RetryHistory", "RetryTimeline",
]
