"""Core @retry decorator with circuit breaker, fallback, dead letter, budget, and history."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from datetime import timedelta
from typing import Any, Callable, Sequence, Union

from unbreak.backoff import BackoffStrategy, AdaptiveBackoff, get_backoff
from unbreak.budget import BudgetManager, parse_budget
from unbreak.circuit import CircuitBreaker
from unbreak.dead_letter import DeadLetterQueue
from unbreak.errors import get_retry_after, is_retryable
from unbreak.events import EventDispatcher, EventType, RetryEvent
from unbreak.fallback import run_fallback_chain, run_fallback_chain_async, FallbackChainError
from unbreak.history import RetryHistory
from unbreak.jitter import apply_jitter, coordinated_jitter


class RetryResult:
    """Wrapper that exposes retry metadata alongside the result."""

    def __init__(self, value: Any, history: RetryHistory) -> None:
        self._value = value
        self._history = history

    def __getattr__(self, name: str) -> Any:
        if name in ("retries", "total_wait", "errors", "timeline", "elapsed"):
            return getattr(self._history, name)
        return getattr(self._value, name)

    def __repr__(self) -> str:
        return f"RetryResult(value={self._value!r}, retries={self.retries})"

    @property
    def value(self) -> Any:
        return self._value


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open."""


def _validate_fallbacks(fallbacks: Sequence[Callable[..., Any]] | None) -> None:
    if fallbacks is None:
        return
    for i, fb in enumerate(fallbacks):
        if not callable(fb):
            raise TypeError(
                f"Fallback at index {i} is not callable: {fb!r}"
            )


def retry(
    *,
    max: int = 3,
    backoff: str | BackoffStrategy = "exponential",
    on: tuple[type[Exception], ...] | None = None,
    until: Callable[[Any], bool] | None = None,
    jitter: str | float | None = "equal",
    jitter_key: str | None = None,
    circuit: bool | CircuitBreaker = False,
    fallback: Sequence[Callable[..., Any]] | None = None,
    dead_letter: Any = None,
    budget: Union[str, timedelta, float, int, None] = None,
    on_event: Callable[[RetryEvent], Any] | None = None,
    base: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 60.0,
    delay: float = 1.0,
    smoothing: float = 0.3,
    wrap_result: bool = True,
) -> Callable:
    backoff_strategy = get_backoff(backoff, base=base, factor=factor,
                                   max_delay=max_delay, delay=delay, smoothing=smoothing)
    is_adaptive = isinstance(backoff_strategy, AdaptiveBackoff)

    _validate_fallbacks(fallback)

    cb: CircuitBreaker | None = None
    if circuit is True:
        cb = CircuitBreaker()
    elif isinstance(circuit, CircuitBreaker):
        cb = circuit

    dlq: DeadLetterQueue | None = None
    if dead_letter is not None:
        dlq = DeadLetterQueue(dead_letter) if not isinstance(dead_letter, DeadLetterQueue) else dead_letter

    dispatcher = EventDispatcher(on_event)
    if cb is not None:
        cb.set_dispatcher(dispatcher)

    budget_td: timedelta | None = None
    if budget is not None:
        budget_td = parse_budget(budget)

    def decorator(fn: Callable) -> Callable:
        is_async_fn = inspect.iscoroutinefunction(fn)
        func_id = id(fn)

        if is_async_fn:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _do_retry_async(
                    fn, args, kwargs, max, backoff_strategy, is_adaptive,
                    func_id, on, until, jitter, jitter_key, cb, fallback,
                    dlq, budget_td, dispatcher, wrap_result,
                )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _do_retry_sync(
                    fn, args, kwargs, max, backoff_strategy, is_adaptive,
                    func_id, on, until, jitter, jitter_key, cb, fallback,
                    dlq, budget_td, dispatcher, wrap_result,
                )
            return sync_wrapper

    return decorator


def _should_retry_error(error: BaseException, on: tuple | None) -> bool:
    if on is not None:
        return isinstance(error, on)
    return is_retryable(error)


def _should_stop(attempt: int, max: int, bm: BudgetManager | None) -> bool:
    if bm and bm.is_exhausted():
        return True
    return attempt >= max


def _compute_wait(
    backoff: BackoffStrategy, adaptive_backoff: AdaptiveBackoff | None,
    func_id: int, attempt: int,
    jitter: str | float | None, jitter_key: str | None,
    bm: BudgetManager | None, error: BaseException | None,
) -> float:
    if bm and not bm.is_exhausted():
        wait = bm.compute_wait(attempt)
    elif adaptive_backoff:
        wait = adaptive_backoff.compute_for(attempt, func_id)
    else:
        wait = backoff.compute(attempt)

    if error is not None:
        ra = get_retry_after(error)
        if ra is not None:
            wait = max(wait, ra)

    if jitter_key:
        wait = coordinated_jitter(wait, attempt, key=jitter_key)
    elif jitter:
        wait = apply_jitter(wait, jitter)

    return wait


def _do_retry_sync(
    fn, args, kwargs, max, backoff, is_adaptive, func_id,
    on, until, jitter, jitter_key, cb, fallback, dlq, budget_td, dispatcher, wrap_result,
):
    """Synchronous retry loop."""
    history = RetryHistory()
    bm: BudgetManager | None = BudgetManager(budget_td) if budget_td else None
    adaptive_backoff: AdaptiveBackoff = backoff if is_adaptive else None  # type: ignore
    attempt = 0
    total_wait = 0.0
    last_error: BaseException | None = None

    if bm:
        bm.start()

    while True:
        attempt += 1

        if cb and not cb.allow():
            dispatcher.emit(RetryEvent(
                type=EventType.CIRCUIT_OPEN, function_name=fn.__name__,
                attempt=attempt, circuit_state=cb.state,
            ))
            err = CircuitBreakerOpenError(f"Circuit is open for {fn.__name__}")
            if fallback:
                dispatcher.emit(RetryEvent(
                    type=EventType.FALLBACK_TRIGGERED, function_name=fn.__name__,
                    attempt=attempt, error=err,
                ))
                result = run_fallback_chain(fallback, args, kwargs)
                return RetryResult(result, history) if wrap_result else result
            raise err

        if bm and bm.is_exhausted():
            break

        try:
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - t0

            if adaptive_backoff:
                adaptive_backoff.record_success(func_id, elapsed)

            if cb and cb.record_response_time(elapsed):
                cb.record_failure(attempt, fn_name=fn.__name__)

            if until and not until(result):
                dispatcher.emit(RetryEvent(
                    type=EventType.RETRY_ATTEMPT, function_name=fn.__name__,
                    attempt=attempt, error=None,
                    extra={"reason": "until_condition_not_met"},
                ))
                if _should_stop(attempt, max, bm):
                    history.record(attempt, 0.0, None)
                    break
                wait = _compute_wait(backoff, adaptive_backoff, func_id, attempt,
                                     jitter, jitter_key, bm, last_error)
                total_wait += wait
                history.record(attempt, wait, None)
                time.sleep(wait)
                continue

            if cb:
                cb.record_success(fn_name=fn.__name__)
            history.record(attempt, 0.0, None)
            cb_state = cb.state if cb else None
            dispatcher.emit(RetryEvent(
                type=EventType.RETRY_SUCCESS, function_name=fn.__name__,
                attempt=attempt, wait_time=total_wait, circuit_state=cb_state,
            ))
            return RetryResult(result, history) if wrap_result else result

        except Exception as e:
            last_error = e
            should_retry = _should_retry_error(e, on)

            if cb and should_retry:
                cb.record_failure(attempt, fn_name=fn.__name__)
                cb.record_attempt(attempt, False)

            if not should_retry or _should_stop(attempt, max, bm):
                history.record(attempt, 0.0, e)
                break

            dispatcher.emit(RetryEvent(
                type=EventType.RETRY_ATTEMPT, function_name=fn.__name__,
                attempt=attempt, error=e, circuit_state=cb.state if cb else None,
            ))

            wait = _compute_wait(backoff, adaptive_backoff, func_id, attempt,
                                 jitter, jitter_key, bm, e)
            total_wait += wait
            history.record(attempt, wait, e)
            time.sleep(wait)

    # Exhausted
    dispatcher.emit(RetryEvent(
        type=EventType.RETRY_EXHAUSTED, function_name=fn.__name__,
        attempt=attempt, error=last_error,
    ))

    if dlq and last_error:
        entry = dlq.record(fn.__name__, args, kwargs, last_error, attempt, total_wait)
        dispatcher.emit(RetryEvent(
            type=EventType.DEAD_LETTER, function_name=fn.__name__,
            attempt=attempt, error=last_error,
            extra={"entry": entry},
        ))

    if fallback:
        dispatcher.emit(RetryEvent(
            type=EventType.FALLBACK_TRIGGERED, function_name=fn.__name__,
            attempt=attempt, error=last_error,
        ))
        try:
            result = run_fallback_chain(fallback, args, kwargs)
            return RetryResult(result, history) if wrap_result else result
        except FallbackChainError:
            raise

    if last_error:
        raise last_error
    raise RuntimeError(f"All {attempt} attempts failed for {fn.__name__}")


async def _do_retry_async(
    fn, args, kwargs, max, backoff, is_adaptive, func_id,
    on, until, jitter, jitter_key, cb, fallback, dlq, budget_td, dispatcher, wrap_result,
):
    """Asynchronous retry loop."""
    history = RetryHistory()
    bm: BudgetManager | None = BudgetManager(budget_td) if budget_td else None
    adaptive_backoff: AdaptiveBackoff = backoff if is_adaptive else None  # type: ignore
    attempt = 0
    total_wait = 0.0
    last_error: BaseException | None = None

    if bm:
        bm.start()

    while True:
        attempt += 1

        if cb and not cb.allow():
            dispatcher.emit(RetryEvent(
                type=EventType.CIRCUIT_OPEN, function_name=fn.__name__,
                attempt=attempt, circuit_state=cb.state,
            ))
            err = CircuitBreakerOpenError(f"Circuit is open for {fn.__name__}")
            if fallback:
                dispatcher.emit(RetryEvent(
                    type=EventType.FALLBACK_TRIGGERED, function_name=fn.__name__,
                    attempt=attempt, error=err,
                ))
                result = await run_fallback_chain_async(fallback, args, kwargs)
                return RetryResult(result, history) if wrap_result else result
            raise err

        if bm and bm.is_exhausted():
            break

        try:
            t0 = time.monotonic()
            result = await fn(*args, **kwargs)
            elapsed = time.monotonic() - t0

            if adaptive_backoff:
                adaptive_backoff.record_success(func_id, elapsed)

            if cb and cb.record_response_time(elapsed):
                cb.record_failure(attempt, fn_name=fn.__name__)

            if until and not until(result):
                dispatcher.emit(RetryEvent(
                    type=EventType.RETRY_ATTEMPT, function_name=fn.__name__,
                    attempt=attempt, error=None,
                    extra={"reason": "until_condition_not_met"},
                ))
                if _should_stop(attempt, max, bm):
                    history.record(attempt, 0.0, None)
                    break
                wait = _compute_wait(backoff, adaptive_backoff, func_id, attempt,
                                     jitter, jitter_key, bm, last_error)
                total_wait += wait
                history.record(attempt, wait, None)
                await asyncio.sleep(wait)
                continue

            if cb:
                cb.record_success(fn_name=fn.__name__)
            history.record(attempt, 0.0, None)
            cb_state = cb.state if cb else None
            dispatcher.emit(RetryEvent(
                type=EventType.RETRY_SUCCESS, function_name=fn.__name__,
                attempt=attempt, wait_time=total_wait, circuit_state=cb_state,
            ))
            return RetryResult(result, history) if wrap_result else result

        except Exception as e:
            last_error = e
            should_retry = _should_retry_error(e, on)

            if cb and should_retry:
                cb.record_failure(attempt, fn_name=fn.__name__)
                cb.record_attempt(attempt, False)

            if not should_retry or _should_stop(attempt, max, bm):
                history.record(attempt, 0.0, e)
                break

            dispatcher.emit(RetryEvent(
                type=EventType.RETRY_ATTEMPT, function_name=fn.__name__,
                attempt=attempt, error=e, circuit_state=cb.state if cb else None,
            ))

            wait = _compute_wait(backoff, adaptive_backoff, func_id, attempt,
                                 jitter, jitter_key, bm, e)
            total_wait += wait
            history.record(attempt, wait, e)
            await asyncio.sleep(wait)

    # Exhausted
    dispatcher.emit(RetryEvent(
        type=EventType.RETRY_EXHAUSTED, function_name=fn.__name__,
        attempt=attempt, error=last_error,
    ))

    if dlq and last_error:
        entry = dlq.record(fn.__name__, args, kwargs, last_error, attempt, total_wait)
        dispatcher.emit(RetryEvent(
            type=EventType.DEAD_LETTER, function_name=fn.__name__,
            attempt=attempt, error=last_error,
            extra={"entry": entry},
        ))

    if fallback:
        dispatcher.emit(RetryEvent(
            type=EventType.FALLBACK_TRIGGERED, function_name=fn.__name__,
            attempt=attempt, error=last_error,
        ))
        try:
            result = await run_fallback_chain_async(fallback, args, kwargs)
            return RetryResult(result, history) if wrap_result else result
        except FallbackChainError:
            raise

    if last_error:
        raise last_error
    raise RuntimeError(f"All {attempt} attempts failed for {fn.__name__}")
