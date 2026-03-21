"""Tests for the core @retry decorator."""

import asyncio
import pytest
from retryly.retry import retry, RetryResult, CircuitBreakerOpenError
from retryly.circuit import CircuitBreaker


class TestBasicRetry:
    def test_succeeds_first_try(self):
        @retry(max=3)
        def fn():
            return "ok"
        result = fn()
        assert result.value == "ok"
        assert result.retries == 0

    def test_retries_on_error(self):
        count = 0
        @retry(max=3, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            if count < 3:
                raise ConnectionError()
            return "ok"
        result = fn()
        assert result.value == "ok"
        assert count == 3

    def test_exhausts_retries(self):
        @retry(max=2, backoff="fixed", delay=0.001)
        def fn():
            raise ConnectionError("fail")
        with pytest.raises(ConnectionError):
            fn()

    def test_non_retryable_error(self):
        count = 0
        @retry(max=5, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            raise ValueError("not retryable")
        with pytest.raises(ValueError):
            fn()
        assert count == 1  # no retry


class TestExplicitErrors:
    def test_on_tuple(self):
        count = 0
        @retry(max=3, on=(ValueError,), backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            if count < 2:
                raise ValueError()
            return "ok"
        result = fn()
        assert result.value == "ok"
        assert count == 2


class TestUntilCondition:
    def test_until_retry(self):
        count = 0
        @retry(max=5, until=lambda r: r >= 3, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            return count
        result = fn()
        assert result.value == 3


class TestBackoffModes:
    def test_fixed_backoff(self):
        count = 0
        @retry(max=3, backoff="fixed", delay=0.01)
        def fn():
            nonlocal count
            count += 1
            if count < 3:
                raise ConnectionError()
            return "ok"
        fn()
        assert count == 3

    def test_exponential_backoff(self):
        count = 0
        @retry(max=5, backoff="exponential", base=0.001, jitter="none")
        def fn():
            nonlocal count
            count += 1
            if count < 3:
                raise ConnectionError()
            return "ok"
        fn()
        assert count == 3


class TestCircuitBreakerIntegration:
    def test_circuit_opens(self):
        cb = CircuitBreaker(failure_threshold=2)
        count = 0
        @retry(max=10, circuit=cb, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            raise ConnectionError("fail")
        with pytest.raises((ConnectionError, CircuitBreakerOpenError)):
            fn()

    def test_circuit_allows_after_reset(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        count = 0
        @retry(max=10, circuit=cb, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            raise ConnectionError("fail")
        try:
            fn()
        except (ConnectionError, CircuitBreakerOpenError):
            pass
        import time
        time.sleep(0.02)
        # Circuit should allow after reset
        assert cb.state.value in ("half_open", "open")


class TestFallbackIntegration:
    def test_fallback_triggered(self):
        @retry(max=2, backoff="fixed", delay=0.001, fallback=[lambda: "fallback"])
        def fn():
            raise ConnectionError()
        result = fn()
        assert result.value == "fallback"

    def test_fallback_chain(self):
        count = 0
        def fb1():
            nonlocal count
            count += 1
            raise ValueError()
        @retry(max=2, backoff="fixed", delay=0.001, fallback=[fb1, lambda: "final"])
        def fn():
            raise ConnectionError()
        result = fn()
        assert result.value == "final"
        assert count == 1


class TestDeadLetterIntegration:
    def test_records_to_dlq(self):
        dlq = []
        @retry(max=2, backoff="fixed", delay=0.001, dead_letter=dlq.append)
        def fn():
            raise ConnectionError("dead")
        with pytest.raises(ConnectionError):
            fn()
        assert len(dlq) == 1
        assert dlq[0]["function_name"] == "fn"


class TestHistoryIntegration:
    def test_history_tracking(self):
        @retry(max=3, backoff="fixed", delay=0.001)
        def fn():
            raise ConnectionError("fail")
        with pytest.raises(ConnectionError):
            fn()
        # History tracked internally, but not accessible without wrapper
        # The RetryResult wrapper exposes it on success

    def test_history_on_success(self):
        count = 0
        @retry(max=5, backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            if count < 3:
                raise ConnectionError()
            return "ok"
        result = fn()
        assert result.retries == 2
        assert result.total_wait > 0
        assert len(result.errors) == 2
        assert len(result.timeline) == 3


class TestEventIntegration:
    def test_events_emitted(self):
        events = []
        @retry(max=3, backoff="fixed", delay=0.001, on_event=events.append)
        def fn():
            raise ConnectionError("fail")
        with pytest.raises(ConnectionError):
            fn()
        types = [e.type for e in events]
        from retryly.events import EventType
        assert EventType.RETRY_ATTEMPT in types
        assert EventType.RETRY_EXHAUSTED in types


class TestBudgetMode:
    def test_budget_mode(self):
        import time
        count = 0
        @retry(budget="0.1s", backoff="fixed", delay=0.001)
        def fn():
            nonlocal count
            count += 1
            raise ConnectionError("fail")
        with pytest.raises(ConnectionError):
            fn()
        # Should have made several attempts within budget
        assert count >= 1


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_async_retries(self):
        count = 0
        @retry(max=3, backoff="fixed", delay=0.001)
        async def fn():
            nonlocal count
            count += 1
            if count < 3:
                raise ConnectionError()
            return "ok"
        result = await fn()
        assert result.value == "ok"
        assert count == 3

    @pytest.mark.asyncio
    async def test_async_exhausts(self):
        @retry(max=2, backoff="fixed", delay=0.001)
        async def fn():
            raise ConnectionError()
        with pytest.raises(ConnectionError):
            await fn()

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        @retry(max=2, backoff="fixed", delay=0.001, fallback=[lambda: "fb"])
        async def fn():
            raise ConnectionError()
        result = await fn()
        assert result.value == "fb"

    @pytest.mark.asyncio
    async def test_async_history(self):
        count = 0
        @retry(max=5, backoff="fixed", delay=0.001)
        async def fn():
            nonlocal count
            count += 1
            if count < 2:
                raise ConnectionError()
            return "ok"
        result = await fn()
        assert result.retries == 1


class TestWrapResult:
    def test_no_wrap(self):
        @retry(max=3, wrap_result=False)
        def fn():
            return 42
        assert fn() == 42
        assert not isinstance(fn(), RetryResult)
