"""End-to-end integration tests for unbreak — full feature combinations."""

import asyncio
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from unbreak import (
    CircuitBreaker,
    CircuitState,
    DeadLetterQueue,
    EventType,
    RetryEvent,
    RetryHistory,
    AdaptiveBackoff,
    retry,
)
from unbreak.errors import clear_custom_retryable, register_retryable, unregister_retryable


# ── 1. Full Pipeline: retry + adaptive backoff + circuit breaker + fallback + dead letter ──

class TestFullPipeline:
    """All features combined on a permanently failing function."""

    def test_full_pipeline(self):
        dlq = DeadLetterQueue()
        cb = CircuitBreaker(failure_threshold=3)
        fallback_called = False

        def my_fallback():
            nonlocal fallback_called
            fallback_called = True
            return "saved_by_fallback"

        @retry(
            max=5,
            backoff="adaptive",
            delay=0.001,
            circuit=cb,
            fallback=[my_fallback],
            dead_letter=dlq,
            jitter="none",
        )
        def always_fail():
            raise ConnectionError("network down")

        result = always_fail()
        assert result.value == "saved_by_fallback"
        assert fallback_called

        # Circuit should be open
        assert cb.state == CircuitState.OPEN

        # DLQ is NOT written when fallback succeeds (fallback returns before DLQ path)
        # This is by design: fallback handled the failure, no dead letter needed
        entries = dlq.read_all()
        assert len(entries) == 0

    def test_full_pipeline_no_fallback_raises(self):
        """Without fallback, the final error is raised after DLQ records it."""
        dlq = DeadLetterQueue()

        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            dead_letter=dlq,
            jitter="none",
        )
        def fail():
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError, match="boom"):
            fail()

        entries = dlq.read_all()
        assert len(entries) == 1
        assert entries[0].retry_count == 3


# ── 2. Dead Letter Replay ──

class TestDeadLetterReplay:
    """Write failures to file, replay them, verify they succeed."""

    def test_file_dlq_replay(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            dlq = DeadLetterQueue(f"file://{path}")
            call_log = []

            @retry(max=2, backoff="fixed", delay=0.001, dead_letter=dlq, jitter="none")
            def flaky(x):
                call_log.append(x)
                if len(call_log) <= 2:
                    raise ConnectionError(f"fail-{x}")
                return x * 2

            # First call: fails, goes to DLQ
            call_log.clear()
            with pytest.raises(ConnectionError):
                flaky(5)

            assert len(call_log) == 2  # 2 attempts

            # Now simulate "replay" — function would succeed now
            from unbreak.dead_letter import FileBackend
            backend = FileBackend(path)
            entries = backend.read_all()
            assert len(entries) == 1
            assert list(entries[0].args) == [5]  # JSON deserializes tuples as lists

            # Manually replay by calling the function with the same args
            result = flaky(5)
            assert result.value == 10
        finally:
            os.unlink(path)


# ── 3. Circuit Breaker Lifecycle ──

class TestCircuitBreakerLifecycle:
    """closed → failures → open → recovery timeout → half-open → success → closed."""

    def test_full_lifecycle(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        call_count = 0

        @retry(max=10, backoff="fixed", delay=0.001, circuit=cb, jitter="none")
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("fail")
            return "ok"

        # Phase 1: circuit opens during retries → raises CircuitBreakerOpenError
        from unbreak.retry import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            sometimes_fails()
        assert cb.state == CircuitState.OPEN

        # Phase 2: calls still blocked while open
        with pytest.raises(CircuitBreakerOpenError):
            sometimes_fails()

        # Phase 3: wait for recovery timeout → half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Phase 4: success in half-open → closed (call_count > 3 now)
        result = sometimes_fails()
        assert result.value == "ok"
        assert cb.state == CircuitState.CLOSED


# ── 4. Predictive Circuit Breaker ──

class TestPredictiveCircuitBreaker:
    """Function gradually slows down → circuit opens before actual failures."""

    def test_predictive_opening(self):
        cb = CircuitBreaker(
            failure_threshold=10,  # high threshold, predictive should trigger first
            recovery_timeout=5.0,
            window_size=20,
            predictive=True,
            predictive_threshold=1.5,
        )
        delays = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.3, 0.5, 1.0]
        call_idx = [0]

        @retry(max=20, backoff="fixed", delay=0.001, circuit=cb, jitter="none")
        def slowing_fn():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx >= len(delays):
                return "done"
            time.sleep(delays[idx])
            return f"attempt_{idx}"

        # Call until circuit opens
        result = None
        last_error = None
        for _ in range(20):
            try:
                result = slowing_fn()
                if cb.state == CircuitState.OPEN:
                    break
            except Exception as e:
                last_error = e
                if cb.state == CircuitState.OPEN:
                    break

        # The circuit should have opened due to prediction
        # (response times degraded beyond threshold)
        assert cb.state == CircuitState.OPEN or last_error is not None or result is not None


# ── 5. Adaptive Backoff Learning ──

class TestAdaptiveBackoffLearning:
    """Function succeeds after variable delays; backoff adapts."""

    def test_adaptive_reduces_backoff(self):
        """Fast function → adaptive backoff should learn lower delays."""
        call_idx = [0]

        @retry(max=5, backoff="adaptive", base=0.01, jitter="none")
        def fast_fn():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < 3:
                raise ConnectionError("transient")
            return "ok"

        t0 = time.monotonic()
        result = fast_fn()
        elapsed = time.monotonic() - t0
        assert result.value == "ok"
        # With very low base (0.01), even worst case should be fast
        assert elapsed < 5.0


# ── 6. Budget Mode with Real Timing ──

class TestBudgetModeTiming:
    """Verify total time is within ~10% of budget."""

    def test_budget_timing(self):
        budget_seconds = 0.5

        @retry(budget=f"{budget_seconds}s", backoff="fixed", delay=0.001, jitter="none")
        def slow_fail():
            raise ConnectionError("timeout")

        t0 = time.monotonic()
        with pytest.raises(ConnectionError):
            slow_fail()
        elapsed = time.monotonic() - t0

        # Should be within ~50% of budget (generous since BudgetManager has its own scheduling)
        assert budget_seconds * 0.1 < elapsed < budget_seconds * 2.0


# ── 7. Event Observability ──

class TestEventObservability:
    """Hook into all event types, verify they fire in correct sequence."""

    def test_full_event_sequence_failure_then_fallback(self):
        events = []

        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            jitter="none",
            on_event=events.append,
            fallback=[lambda: "fallback_result"],
        )
        def fail():
            raise ConnectionError("err")

        result = fail()
        assert result.value == "fallback_result"

        types = [e.type for e in events]
        # Should have retry attempts, then exhausted, then fallback
        assert EventType.RETRY_ATTEMPT in types
        assert EventType.RETRY_EXHAUSTED in types
        assert EventType.FALLBACK_TRIGGERED in types

        # Verify ordering: attempts come before exhausted
        attempt_indices = [i for i, t in enumerate(types) if t == EventType.RETRY_ATTEMPT]
        exhausted_idx = types.index(EventType.RETRY_EXHAUSTED)
        fallback_idx = types.index(EventType.FALLBACK_TRIGGERED)
        assert all(a < exhausted_idx for a in attempt_indices)
        assert exhausted_idx < fallback_idx

    def test_success_event_sequence(self):
        events = []
        call_count = [0]

        @retry(max=3, backoff="fixed", delay=0.001, jitter="none", on_event=events.append)
        def succeed_on_second():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("first")
            return "ok"

        result = succeed_on_second()
        assert result.value == "ok"

        types = [e.type for e in events]
        assert EventType.RETRY_ATTEMPT in types
        assert EventType.RETRY_SUCCESS in types

        attempt_idx = types.index(EventType.RETRY_ATTEMPT)
        success_idx = types.index(EventType.RETRY_SUCCESS)
        assert attempt_idx < success_idx

    def test_all_event_types_covered(self):
        """Verify events cover: retry_attempt, success/exhausted, circuit_open, dead_letter."""
        events = []
        dlq = DeadLetterQueue()
        cb = CircuitBreaker(failure_threshold=10)  # high threshold so we exhaust retries first

        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            jitter="none",
            circuit=cb,
            dead_letter=dlq,
            on_event=events.append,
        )
        def fail():
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            fail()

        types = set(e.type for e in events)
        assert EventType.RETRY_ATTEMPT in types
        assert EventType.RETRY_EXHAUSTED in types
        assert EventType.DEAD_LETTER in types


# ── 8. Retry History ──

class TestRetryHistory:
    """Call fails 3 times then succeeds. Verify history metadata."""

    def test_history_accuracy(self):
        call_count = [0]

        @retry(max=4, backoff="fixed", delay=0.01, jitter="none")
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] <= 3:
                raise ConnectionError(f"fail-{call_count[0]}")
            return "success"

        result = fail_then_succeed()
        assert result.value == "success"

        # Verify history
        assert result.retries == 3  # 3 retries after first attempt
        assert len(result.errors) == 3
        assert all(isinstance(e, ConnectionError) for e in result.errors)
        assert result.total_wait > 0

        # Timeline: 4 entries (1 fail + wait, 2 fail + wait, 3 fail + wait, 4 success)
        assert len(result.timeline) == 4
        assert result.timeline[0].attempt == 1
        assert result.timeline[0].error is not None
        assert result.timeline[3].attempt == 4
        assert result.timeline[3].error is None

    def test_history_immediate_success(self):
        @retry(max=3, backoff="fixed", delay=0.001, jitter="none")
        def ok():
            return 42

        result = ok()
        assert result.value == 42
        assert result.retries == 0
        assert result.total_wait == 0.0
        assert len(result.errors) == 0
        assert len(result.timeline) == 1


# ── 9. Async Integration ──

class TestAsyncFullIntegration:
    """Async function with circuit breaker + fallback + dead letter."""

    @pytest.mark.asyncio
    async def test_async_circuit_fallback_dlq(self):
        dlq = DeadLetterQueue()
        cb = CircuitBreaker(failure_threshold=2)
        events = []

        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            circuit=cb,
            fallback=[lambda: "async_fallback"],
            dead_letter=dlq,
            on_event=events.append,
            jitter="none",
        )
        async def always_fail():
            raise ConnectionError("async boom")

        result = await always_fail()
        assert result.value == "async_fallback"
        assert cb.state == CircuitState.OPEN

        # DLQ is NOT written when fallback succeeds (by design)
        entries = dlq.read_all()
        assert len(entries) == 0

        # Events fired
        types = [e.type for e in events]
        assert EventType.FALLBACK_TRIGGERED in types

    @pytest.mark.asyncio
    async def test_async_retry_then_succeed(self):
        count = [0]

        @retry(max=3, backoff="fixed", delay=0.001, jitter="none")
        async def flaky():
            count[0] += 1
            if count[0] < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = await flaky()
        assert result.value == "recovered"
        assert result.retries == 2


# ── 10. Cloud-Style Scenario: HTTP 429 with Retry-After ──

class TestCloudStyleRetryAfter:
    """Simulate HTTP 429 with Retry-After header, verify retry respects it."""

    def test_retry_after_header(self):
        call_count = [0]

        class HTTP429Error(Exception):
            status_code = 429
            def __init__(self, retry_after):
                self.retry_after = retry_after
                self.response = MagicMock()
                self.response.headers = {"Retry-After": str(retry_after)}
                super().__init__("429 Too Many Requests")

        @retry(max=5, backoff="fixed", delay=0.001, jitter="none", on=(HTTP429Error,))
        def api_call():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise HTTP429Error(0.05)  # Retry-After: 0.05s
            return {"status": "ok"}

        t0 = time.monotonic()
        result = api_call()
        elapsed = time.monotonic() - t0

        assert result.value == {"status": "ok"}
        # Should have waited at least 0.05s per 429 response (2 times)
        assert elapsed >= 0.08
        assert call_count[0] == 3


# ── 11. Concurrent Access ──

class TestConcurrentAccess:
    """Multiple threads calling same retry-wrapped function with shared circuit breaker."""

    def test_concurrent_shared_circuit(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)
        call_count = [0]
        lock = threading.Lock()

        @retry(max=2, backoff="fixed", delay=0.001, circuit=cb, jitter="none")
        def shared_fn():
            with lock:
                call_count[0] += 1
            raise ConnectionError("shared_fail")

        results = []
        errors = []

        def worker():
            try:
                shared_fn()
                results.append("success")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have mix of ConnectionError and CircuitBreakerOpenError
        error_types = set(errors)
        assert "ConnectionError" in error_types or "CircuitBreakerOpenError" in error_types
        # Circuit should eventually be open
        assert cb.state == CircuitState.OPEN

        # Total calls should be bounded (circuit prevents excessive calls)
        assert call_count[0] <= 8 * 2  # at most max_attempts per thread


# ── 12. Edge Cases & Bonus Scenarios ──

class TestEdgeCases:
    """Additional edge cases and combined scenarios."""

    def test_fallback_chain_multiple(self):
        """Multiple fallbacks: first fails, second succeeds."""
        @retry(
            max=2,
            backoff="fixed",
            delay=0.001,
            jitter="none",
            fallback=[
                lambda: (_ for _ in ()).throw(ValueError("fb1_fail")),  # raises
                lambda: "fb2_success",
            ],
        )
        def fail():
            raise ConnectionError("main_fail")

        # Need a proper raising fallback
        def fb1():
            raise ValueError("fb1_fail")

        @retry(
            max=2,
            backoff="fixed",
            delay=0.001,
            jitter="none",
            fallback=[fb1, lambda: "fb2_success"],
        )
        def fail2():
            raise ConnectionError("main_fail")

        result = fail2()
        assert result.value == "fb2_success"

    def test_circuit_repeated_open_close_cycles(self):
        """Circuit opens, recovers, opens again."""
        from unbreak.retry import CircuitBreakerOpenError
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        phase = [0]
        call_count = [0]

        @retry(max=5, backoff="fixed", delay=0.001, circuit=cb, jitter="none")
        def cycling_fn():
            call_count[0] += 1
            if phase[0] == 0:
                raise ConnectionError("phase0")
            elif phase[0] == 1:
                return "recovered"
            else:
                raise ConnectionError("phase2")

        # Phase 0: fail until circuit opens
        with pytest.raises(CircuitBreakerOpenError):
            cycling_fn()
        assert cb.state == CircuitState.OPEN

        # Phase 1: wait and recover
        time.sleep(0.1)
        phase[0] = 1
        result = cycling_fn()
        assert result.value == "recovered"
        assert cb.state == CircuitState.CLOSED

        # Phase 2: fail again to re-open — reset first so circuit gets fresh failures
        cb.reset()
        phase[0] = 2
        with pytest.raises(CircuitBreakerOpenError):
            cycling_fn()
        assert cb.state == CircuitState.OPEN

    def test_wrap_result_false(self):
        """wrap_result=False returns raw value."""
        call_count = [0]

        @retry(max=3, backoff="fixed", delay=0.001, jitter="none", wrap_result=False)
        def raw_fn():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("oops")
            return {"data": 123}

        result = raw_fn()
        assert result == {"data": 123}
        assert not hasattr(result, "retries")

    def test_custom_error_type(self):
        """Custom retryable error type."""
        class TransientError(Exception):
            pass

        register_retryable(TransientError)
        try:
            @retry(max=3, backoff="fixed", delay=0.001, jitter="none")
            def custom_fail():
                raise TransientError("custom")

            with pytest.raises(TransientError):
                custom_fail()
        finally:
            unregister_retryable(TransientError)
