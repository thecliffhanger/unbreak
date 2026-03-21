"""Integration tests combining multiple features."""

import asyncio
import tempfile
import os
import pytest
from retryly.retry import retry, CircuitBreakerOpenError
from retryly.circuit import CircuitBreaker
from retryly.events import EventType


class TestRetryCircuitFallbackDLQ:
    """Integration: retry + circuit breaker + fallback + dead letter."""

    def test_full_pipeline_success(self):
        count = 0
        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            circuit=True,
            fallback=[lambda: "fallback"],
        )
        def fn():
            nonlocal count
            count += 1
            raise ConnectionError()
        result = fn()
        assert result.value == "fallback"

    def test_circuit_blocks_all(self):
        cb = CircuitBreaker(failure_threshold=1)
        dlq = []
        @retry(
            max=5,
            backoff="fixed",
            delay=0.001,
            circuit=cb,
            dead_letter=dlq.append,
        )
        def fn():
            raise ConnectionError()
        with pytest.raises(CircuitBreakerOpenError):
            fn()
        # Second call should be blocked by circuit
        with pytest.raises(CircuitBreakerOpenError):
            fn()

    def test_events_with_all_features(self):
        events = []
        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            on_event=events.append,
            fallback=[lambda: "ok"],
        )
        def fn():
            raise ConnectionError()
        result = fn()
        assert result.value == "ok"
        types = [e.type for e in events]
        assert EventType.RETRY_ATTEMPT in types
        assert EventType.FALLBACK_TRIGGERED in types

    def test_file_dlq_integration(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            @retry(
                max=2,
                backoff="fixed",
                delay=0.001,
                dead_letter=f"file://{path}",
            )
            def fn():
                raise ConnectionError("recorded")
            with pytest.raises(ConnectionError):
                fn()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            import json
            data = json.loads(lines[0])
            assert data["function_name"] == "fn"
        finally:
            os.unlink(path)

    def test_until_with_fallback(self):
        count = 0
        @retry(
            max=3,
            until=lambda r: r >= 10,
            backoff="fixed",
            delay=0.001,
            fallback=[lambda: 10],
        )
        def fn():
            nonlocal count
            count += 1
            return count
        result = fn()
        assert result.value == 10


class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_async_full_pipeline(self):
        count = 0
        events = []
        @retry(
            max=3,
            backoff="fixed",
            delay=0.001,
            on_event=events.append,
            fallback=[lambda: "async_fb"],
        )
        async def fn():
            nonlocal count
            count += 1
            raise ConnectionError()
        result = await fn()
        assert result.value == "async_fb"
        types = [e.type for e in events]
        assert EventType.FALLBACK_TRIGGERED in types

    @pytest.mark.asyncio
    async def test_async_budget(self):
        count = 0
        @retry(budget="0.1s", backoff="fixed", delay=0.001)
        async def fn():
            nonlocal count
            count += 1
            raise ConnectionError()
        with pytest.raises(ConnectionError):
            await fn()
        assert count >= 1
