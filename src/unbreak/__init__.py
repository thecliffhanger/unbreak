"""unbreak — Smart retry library with adaptive backoff, circuit breaker, fallback chains, and dead letter queues."""

from unbreak.retry import retry, RetryResult, CircuitBreakerOpenError
from unbreak.backoff import BackoffStrategy, FixedBackoff, ExponentialBackoff, AdaptiveBackoff, get_backoff
from unbreak.circuit import CircuitBreaker, CircuitState
from unbreak.errors import (
    is_retryable, register_retryable, unregister_retryable,
    register_retryable_check, clear_custom_retryable, get_retry_after,
)
from unbreak.budget import BudgetManager, parse_budget
from unbreak.fallback import run_fallback_chain, run_fallback_chain_async, FallbackChainError
from unbreak.dead_letter import DeadLetterQueue, replay
from unbreak.jitter import apply_jitter, coordinated_jitter
from unbreak.events import EventType, RetryEvent, EventDispatcher
from unbreak.history import RetryHistory, RetryTimeline

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
