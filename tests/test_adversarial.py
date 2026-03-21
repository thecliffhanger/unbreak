"""Adversarial/fuzz tests for retryly — edge cases, concurrency, corruption, type errors."""

import asyncio
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pytest

from retryly import (
    retry,
    AdaptiveBackoff,
    CircuitBreaker,
    CircuitState,
    DeadLetterQueue,
    RetryResult,
    apply_jitter,
    coordinated_jitter,
    run_fallback_chain,
)


# ─── 1. Edge cases ──────────────────────────────────────────────────────────


class TestRetryEdgeCases:
    """retry(0), retry(max=1), negative, None, huge max."""

    def test_max_zero_runs_once_then_fails(self):
        """max=0 means attempt 1 >= 0, so should stop immediately after first failure."""
        call_count = 0

        @retry(max=0, backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        assert call_count == 1

    def test_max_one_retries_exactly_once(self):
        call_count = 0

        @retry(max=1, backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        assert call_count == 1  # max=1 means only 1 attempt total

    def test_negative_max(self):
        """Negative max should still try at least once."""
        call_count = 0

        @retry(max=-5, backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        assert call_count == 1

    def test_extremely_large_max(self):
        """max=999999 should still work, but we only call once if it succeeds."""
        @retry(max=999999, backoff="fixed", delay=0.001, wrap_result=False)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_function_that_returns_none(self):
        """Function returning None should be treated as success."""

        @retry(max=3, backoff="fixed", delay=0.001, wrap_result=False)
        def returns_none():
            return None

        assert returns_none() is None

    def test_none_fallback_ignored(self):
        """fallback=None should be treated as no fallback."""
        call_count = 0

        @retry(max=1, backoff="fixed", delay=0.001, fallback=None, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()

    def test_budget_zero_seconds(self):
        """budget='0s' should exhaust immediately."""
        call_count = 0

        @retry(budget="0s", backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        # Should have tried at least once, but budget exhausted quickly
        assert call_count >= 1


# ─── 2. Different exception types on each call ──────────────────────────────


class TestMixedExceptionTypes:
    def test_alternating_exception_types_all_retryable(self):
        """All are network errors, should keep retrying until max."""
        call_count = 0
        errors = [ConnectionError, ConnectionRefusedError, TimeoutError, BrokenPipeError]

        @retry(max=5, backoff="fixed", delay=0.001, wrap_result=False)
        def fail_mixed():
            nonlocal call_count
            call_count += 1
            raise errors[(call_count - 1) % len(errors)]("boom")

        with pytest.raises((ConnectionError, ConnectionRefusedError, TimeoutError, BrokenPipeError)):
            fail_mixed()
        assert call_count == 5

    def test_non_retryable_stops_immediately(self):
        call_count = 0

        @retry(max=5, backoff="fixed", delay=0.001, wrap_result=False)
        def fail_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            fail_value_error()
        assert call_count == 1


# ─── 3. Concurrent calls with shared circuit breaker ────────────────────────


class TestConcurrentCircuitBreaker:
    def test_concurrent_failures_open_circuit(self):
        """Rapid concurrent failures should open the circuit."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        call_count = 0

        @retry(max=3, circuit=cb, backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        def worker():
            try:
                fail()
            except (ConnectionError, Exception):
                pass

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state == CircuitState.OPEN

    def test_concurrent_race_on_circuit_state(self):
        """Stress test: many threads reading/writing circuit breaker state."""
        cb = CircuitBreaker(failure_threshold=10, window_size=20)
        errors = []

        def worker(i):
            try:
                @retry(max=2, circuit=cb, backoff="fixed", delay=0.001, wrap_result=False)
                def fail():
                    raise ConnectionError(f"fail-{i}")
                fail()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash; circuit should eventually open
        assert cb.state == CircuitState.OPEN


# ─── 4. Dead letter queue edge cases ────────────────────────────────────────


class TestDeadLetterEdgeCases:
    def test_corrupted_jsonl_file(self):
        """Reading a JSONL file with corrupt lines should handle gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"function_name": "foo", "args": [], "kwargs": {}, "error": "e", "error_type": "E", "timestamp": 1.0, "retry_count": 1, "total_wait_time": 0.0}\n')
            f.write("THIS IS NOT JSON\n")
            f.write('{"function_name": "bar", "args": [], "kwargs": {}, "error": "e", "error_type": "E", "timestamp": 2.0, "retry_count": 2, "total_wait_time": 0.0}\n')
            path = f.name

        from retryly.dead_letter import FileBackend
        try:
            backend = FileBackend(path)
            # Corrupt line should be skipped, valid lines returned
            entries = backend.read_all()
            assert len(entries) == 2
            assert entries[0].function_name == "foo"
            assert entries[1].function_name == "bar"
        finally:
            os.unlink(path)

    def test_permission_denied_write(self):
        """File DLQ with read-only directory should raise on write."""
        with tempfile.TemporaryDirectory() as d:
            ro_dir = Path(d) / "readonly"
            ro_dir.mkdir()
            os.chmod(ro_dir, 0o444)
            path = ro_dir / "dlq.jsonl"

            try:
                from retryly.dead_letter import FileBackend
                backend = FileBackend(str(path))
                entry = backend.read_all.__self__  # just access to trigger init
                # The write should fail
                from retryly.dead_letter import DeadLetterEntry
                entry = DeadLetterEntry(
                    function_name="test", args=(), kwargs={}, error="e",
                    error_type="E", timestamp=1.0, retry_count=1, total_wait_time=0.0,
                )
                with pytest.raises(PermissionError):
                    backend.write(entry)
            finally:
                os.chmod(ro_dir, 0o755)

    def test_empty_fallback_list_in_dlq_path(self):
        """Empty fallback list should raise RuntimeError from run_fallback_chain."""
        with pytest.raises(RuntimeError, match="No fallbacks"):
            run_fallback_chain([], args=(1, 2), kwargs={"x": 3})


# ─── 5. Adaptive backoff with wildly varying response times ─────────────────


class TestAdaptiveBackoffEdgeCases:
    def test_wildly_varying_response_times(self):
        """Adaptive backoff should handle 0.001s, 60s, 0.001s without breaking."""
        ab = AdaptiveBackoff(base=0.1, max_delay=120.0)
        func_id = 42

        # Simulate wildly varying response times
        ab.record_success(func_id, 0.001)
        delay1 = ab.compute_for(1, func_id)
        assert delay1 >= 0.001

        ab.record_success(func_id, 60.0)
        delay2 = ab.compute_for(2, func_id)
        assert delay2 <= 120.0  # capped

        ab.record_success(func_id, 0.001)
        delay3 = ab.compute_for(3, func_id)
        # EMA should be pulling down but not below base
        assert delay3 >= 0.1

    def test_zero_response_time(self):
        ab = AdaptiveBackoff(base=0.1)
        ab.record_success(1, 0.0)
        delay = ab.compute_for(1, 1)
        assert delay >= 0

    def test_extremely_large_response_time(self):
        ab = AdaptiveBackoff(base=0.1, max_delay=60.0)
        ab.record_success(1, 999999.0)
        delay = ab.compute_for(1, 1)
        assert delay <= 60.0


# ─── 6. Circuit breaker edge cases ──────────────────────────────────────────


class TestCircuitBreakerEdgeCases:
    def test_threshold_of_one(self):
        """Circuit with threshold=1 should open after first failure."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        assert cb.state == CircuitState.CLOSED
        cb.record_failure(1)
        assert cb.state == CircuitState.OPEN

    def test_recovery_timeout_zero(self):
        """Circuit with recovery_timeout=0 transitions to HALF_OPEN immediately on next state check."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure(1)
        # With recovery_timeout=0, state transitions immediately to HALF_OPEN
        # because _maybe_transition checks elapsed >= recovery_timeout (0 >= 0)
        assert cb.state == CircuitState.HALF_OPEN

    def test_rapid_fire_successes_after_open(self):
        """Multiple successes after half-open should close circuit."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure(1)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow()  # consume half-open permit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_window_size_one(self):
        """With window_size=1, circuit should be very volatile."""
        cb = CircuitBreaker(failure_threshold=1, window_size=1)
        cb.record_failure(1)
        assert cb.state == CircuitState.OPEN

    def test_record_attempt_pattern_learning(self):
        """Pattern learning should track attempt statistics."""
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(5):
            cb.record_attempt(1, False)
        for _ in range(3):
            cb.record_attempt(1, True)
        # Threshold should be reduced due to high failure rate at attempt 1
        # 5 failures / 8 total = 62.5% < 80% threshold, so no reduction
        learned = cb._learned_threshold()
        assert learned <= 5


# ─── 7. Budget mode edge cases ──────────────────────────────────────────────


class TestBudgetEdgeCases:
    def test_budget_smaller_than_backoff(self):
        """Budget of 10ms with exponential backoff starting at 500ms."""
        call_count = 0

        @retry(budget="10ms", backoff="exponential", base=0.5, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        # Should get at least 1 attempt, maybe 2 depending on timing
        assert call_count >= 1

    def test_budget_with_successful_first_call(self):
        @retry(budget="5s", backoff="fixed", delay=0.001, wrap_result=False)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_budget_negative(self):
        """Negative budget should raise ValueError at decoration time."""
        call_count = 0

        with pytest.raises(ValueError, match="non-negative"):
            @retry(budget=-1, backoff="fixed", delay=0.001, wrap_result=False)
            def fail():
                raise ConnectionError("boom")

    def test_budget_zero_float(self):
        @retry(budget=0.0, backoff="fixed", delay=0.001, wrap_result=False)
        def fail():
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()


# ─── 8. Async edge cases ────────────────────────────────────────────────────


class TestAsyncEdgeCases:
    @pytest.mark.asyncio
    async def test_cancel_mid_retry(self):
        """Cancelling an async task during retry sleep."""
        call_count = 0

        @retry(max=10, backoff="fixed", delay=1.0, wrap_result=False)
        async def slow_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        task = asyncio.create_task(slow_fail())
        await asyncio.sleep(0.05)  # let first attempt run
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Should have made at least one attempt
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_timeout_during_retry(self):
        """Using asyncio.wait_for to timeout during retries."""
        call_count = 0

        @retry(max=100, backoff="fixed", delay=0.5, wrap_result=False)
        async def slow_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_fail(), timeout=0.1)
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_async_fallback_all_fail(self):
        """All async fallbacks fail - FallbackChainError raised with all error info."""

        @retry(max=1, backoff="fixed", delay=0.001,
               fallback=[lambda: (_ for _ in ()).throw(ValueError("fb1")),
                         lambda: (_ for _ in ()).throw(TypeError("fb2"))],
               wrap_result=False)
        async def fail():
            raise ConnectionError("main")

        from retryly.fallback import FallbackChainError
        with pytest.raises(FallbackChainError):
            await fail()


# ─── 9. Coordinated jitter ─────────────────────────────────────────────────


class TestCoordinatedJitterEdgeCases:
    def test_same_key_same_attempt_deterministic(self):
        """Same key and attempt should produce same jitter within window."""
        d1 = coordinated_jitter(1.0, 1, "my-service")
        d2 = coordinated_jitter(1.0, 1, "my-service")
        # Within the same 10-second window, should be identical
        assert d1 == d2

    def test_different_keys_different_jitter(self):
        d1 = coordinated_jitter(1.0, 1, "service-a")
        d2 = coordinated_jitter(1.0, 1, "service-b")
        assert d1 != d2

    def test_empty_key(self):
        """Empty string key should work."""
        d = coordinated_jitter(1.0, 1, "")
        assert 0.001 <= d <= 1.0

    def test_zero_delay(self):
        d = coordinated_jitter(0.0, 1, "key")
        assert d >= 0.001  # should still be positive due to max(0.001, ...)

    def test_negative_delay(self):
        d = coordinated_jitter(-1.0, 1, "key")
        assert d >= 0.001

    def test_very_large_delay(self):
        d = coordinated_jitter(999999.0, 1, "key")
        assert d > 0


# ─── 10. Fallback edge cases ────────────────────────────────────────────────


class TestFallbackEdgeCases:
    def test_all_fallbacks_fail(self):
        """All fallbacks raising should propagate FallbackChainError with all error info."""

        @retry(max=1, backoff="fixed", delay=0.001,
               fallback=[lambda: (_ for _ in ()).throw(ValueError("a")),
                         lambda: (_ for _ in ()).throw(TypeError("b"))],
               wrap_result=False)
        def fail():
            raise ConnectionError("main")

        from retryly.fallback import FallbackChainError
        with pytest.raises(FallbackChainError):
            fail()

    def test_fallback_returns_none(self):
        """Fallback returning None should be accepted as success."""

        @retry(max=1, backoff="fixed", delay=0.001,
               fallback=[lambda: None], wrap_result=False)
        def fail():
            raise ConnectionError("main")

        assert fail() is None

    def test_fallback_receives_correct_args(self):
        received = {}

        def fb(x, y=0):
            received["args"] = (x, y)
            return "fallback"

        @retry(max=1, backoff="fixed", delay=0.001,
               fallback=[fb], wrap_result=False)
        def fail(x, y=0):
            raise ConnectionError("main")

        result = fail(42, y=99)
        assert result == "fallback"
        assert received["args"] == (42, 99)


# ─── 11. Type errors and wrong params ───────────────────────────────────────


class TestTypeErrors:
    def test_non_callable_fallback(self):
        """Passing a non-callable as fallback - TypeError at decoration time."""

        with pytest.raises(TypeError, match="not callable"):
            @retry(max=1, backoff="fixed", delay=0.001, fallback=["not a callable"], wrap_result=False)
            def fail():
                raise ConnectionError("boom")

    def test_invalid_backoff_string(self):
        with pytest.raises(ValueError, match="Unknown backoff strategy"):
            retry(backoff="nonexistent_strategy")

    def test_invalid_budget_string(self):
        with pytest.raises(ValueError, match="Cannot parse budget"):
            retry(budget="not-a-budget")

    def test_invalid_budget_type(self):
        with pytest.raises(TypeError, match="Invalid budget type"):
            retry(budget=object())

    def test_invalid_jitter_mode(self):
        with pytest.raises(ValueError, match="Unknown jitter mode"):
            apply_jitter(1.0, "invalid_mode")

    def test_negative_base(self):
        """Negative base should raise ValueError."""
        from retryly.backoff import ExponentialBackoff
        with pytest.raises(ValueError, match="must be > 0"):
            ExponentialBackoff(base=-1.0, factor=2.0, max_delay=60.0)

    def test_zero_base(self):
        from retryly.backoff import ExponentialBackoff
        with pytest.raises(ValueError, match="must be > 0"):
            ExponentialBackoff(base=0.0, factor=2.0, max_delay=60.0)


# ─── 12. Wrap result edge cases ─────────────────────────────────────────────


class TestWrapResultEdgeCases:
    def test_wrap_result_false_returns_raw(self):
        @retry(max=3, backoff="fixed", delay=0.001, wrap_result=False)
        def succeed():
            return 42

        result = succeed()
        assert result == 42
        assert not isinstance(result, RetryResult)

    def test_wrap_result_true_returns_retry_result(self):
        @retry(max=3, backoff="fixed", delay=0.001, wrap_result=True)
        def succeed():
            return 42

        result = succeed()
        assert isinstance(result, RetryResult)
        assert result.value == 42

    def test_retry_result_exposes_history(self):
        @retry(max=3, backoff="fixed", delay=0.001, wrap_result=True)
        def succeed():
            return 42

        result = succeed()
        # retries=0 because it succeeded on first attempt (no retries needed)
        assert result.retries == 0
        assert result.total_wait == 0.0


# ─── 13. Full pipeline adversarial ──────────────────────────────────────────


class TestFullPipelineAdversarial:
    def test_everything_enabled_at_once(self):
        """Circuit + fallback + DLQ + budget + adaptive backoff + jitter."""
        dlq_entries = []
        dlq = DeadLetterQueue(lambda entry: dlq_entries.append(entry))
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        call_count = 0

        @retry(
            max=5, backoff="adaptive", base=0.001, jitter="equal",
            circuit=cb, fallback=[lambda: "fallback_ok"],
            dead_letter=dlq, budget="2s", wrap_result=True,
        )
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        result = fail()
        assert result.value == "fallback_ok"
        # DLQ should have recorded the failure (but fallback succeeded, so DLQ
        # may or may not have been written - check behavior)
        # Circuit should have opened
        assert cb.state == CircuitState.OPEN

    def test_until_with_always_false(self):
        """until predicate that always returns False should exhaust retries."""
        call_count = 0

        @retry(max=3, backoff="fixed", delay=0.001, until=lambda x: False, wrap_result=False)
        def succeed():
            nonlocal call_count
            call_count += 1
            return call_count

        with pytest.raises(RuntimeError, match="All.*attempts failed"):
            succeed()
        assert call_count == 3

    def test_until_with_exception(self):
        """until predicate should not be called when function raises."""
        call_count = 0

        @retry(max=3, backoff="fixed", delay=0.001, until=lambda x: True, wrap_result=False)
        def fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            fail()
        assert call_count == 3


# ─── BUG SUMMARY ────────────────────────────────────────────────────────────
#
# Bugs found:
#
# 1. **Negative exponential backoff produces negative delay (backoff.py)**:
#    ExponentialBackoff(base=-1.0) computes negative delays. time.sleep(negative)
#    raises ValueError. Should clamp to >= 0.
#
# 2. **Corrupted JSONL crashes read_all (dead_letter.py)**: FileBackend.read_all()
#    doesn't handle JSON parse errors — a single corrupt line crashes the entire
#    read. Should skip corrupt lines gracefully.
#
# 3. **Fallback errors silently swallowed (retry.py)**: When fallback chain is
#    provided and ALL fallbacks fail, the `except Exception: pass` block silently
#    swallows all fallback errors and falls through to re-raise the *original*
#    error from the main function. Users have no idea their fallbacks failed.
#    Should at minimum log/warn, or raise a composite error.
#
# 4. **Non-callable fallback silently fails (retry.py)**: Same mechanism as #3 —
#    passing a string instead of a callable as a fallback causes TypeError inside
#    run_fallback_chain, which is silently caught, and the original error is
#    re-raised. Validation should happen at decoration time.
#
# 5. **Negative budget accepted (budget.py)**: Negative numeric budget creates a
#    negative timedelta. BudgetManager works but the function runs once then gets
#    RuntimeError("All 1 attempts failed") instead of a clear "budget exhausted"
#    error. parse_budget should reject negative values.
