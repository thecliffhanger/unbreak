"""retryly — Smart retry library with adaptive backoff, circuit breaker, fallback chains, and dead letter queues."""

from retryly.retry import retry, RetryResult, CircuitBreakerOpenError
from retryly.backoff import BackoffStrategy, FixedBackoff, ExponentialBackoff, AdaptiveBackoff, get_backoff
from retryly.circuit import CircuitBreaker, CircuitState
from retryly.errors import (
    is_retryable, register_retryable, unregister_retryable,
    register_retryable_check, clear_custom_retryable, get_retry_after,
)
from retryly.budget import BudgetManager, parse_budget
from retryly.fallback import run_fallback_chain, run_fallback_chain_async, FallbackChainError
from retryly.dead_letter import DeadLetterQueue, replay
from retryly.jitter import apply_jitter, coordinated_jitter
from retryly.events import EventType, RetryEvent, EventDispatcher
from retryly.history import RetryHistory, RetryTimeline

__version__ = "0.1.0"
__all__ = [
    "retry", "RetryResult", "CircuitBreakerOpenError",
    "BackoffStrategy", "FixedBackoff", "ExponentialBackoff", "AdaptiveBackoff", "get_backoff",
    "CircuitBreaker", "CircuitState",
    "is_retryable", "register_retryable", "unregister_retryable",
    "register_retryable_check", "clear_custom_retryable", "get_retry_after",
    "BudgetManager", "parse_budget",
    "run_fallback_chain", "run_fallback_chain_async", "FallbackChainError",
    "DeadLetterQueue", "replay",
    "apply_jitter", "coordinated_jitter",
    "EventType", "RetryEvent", "EventDispatcher",
    "RetryHistory", "RetryTimeline",
]
