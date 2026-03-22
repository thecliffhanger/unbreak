"""Tests for fallback chains."""

import pytest
from unbreak.fallback import run_fallback_chain, run_fallback_chain_async


class TestFallbackChain:
    def test_first_succeeds(self):
        assert run_fallback_chain([lambda: "a", lambda: "b"]) == "a"

    def test_falls_through(self):
        def fail(*a, **kw):
            raise ValueError("nope")
        assert run_fallback_chain([fail, lambda: "ok"]) == "ok"

    def test_all_fail_raises_last(self):
        def fail1(*a, **kw):
            raise ValueError("e1")
        def fail2(*a, **kw):
            raise TypeError("e2")
        from unbreak.fallback import FallbackChainError
        with pytest.raises(FallbackChainError):
            run_fallback_chain([fail1, fail2])

    def test_empty_raises(self):
        with pytest.raises(RuntimeError, match="No fallbacks"):
            run_fallback_chain([])

    def test_with_args_kwargs(self):
        def fn(x, y=10):
            return x + y
        assert run_fallback_chain([fn], args=(5,), kwargs={"y": 3}) == 8

    def test_lambda_fallback(self):
        def fail(*a, **kw):
            raise ConnectionError()
        assert run_fallback_chain([fail, lambda: "default"]) == "default"


@pytest.mark.asyncio
async def test_async_fallback_chain():
    async def afail(*a, **kw):
        raise ValueError("async fail")
    async def aok(*a, **kw):
        return "async ok"
    result = await run_fallback_chain_async([afail, aok])
    assert result == "async ok"

@pytest.mark.asyncio
async def test_async_mixed_fallback():
    async def afail(*a, **kw):
        raise ValueError()
    assert await run_fallback_chain_async([afail, lambda: "sync ok"]) == "sync ok"
