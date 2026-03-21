"""Fallback chains for exhausted retries."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence


class FallbackChainError(Exception):
    """Raised when all fallbacks in a chain fail."""

    def __init__(self, fallback_errors: list[Exception]) -> None:
        self.fallback_errors = fallback_errors
        names = [type(e).__name__ for e in fallback_errors]
        super().__init__(f"All {len(fallback_errors)} fallback(s) failed: {', '.join(names)}")


async def run_fallback_chain_async(
    fallbacks: Sequence[Callable[..., Any]],
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    """Run fallback functions in order (async version). Returns first success."""
    kwargs = kwargs or {}
    errors: list[Exception] = []
    for fb in fallbacks:
        try:
            if inspect.iscoroutinefunction(fb):
                return await fb(*args, **kwargs)
            else:
                return fb(*args, **kwargs)
        except Exception as e:
            errors.append(e)
            continue
    if errors:
        raise FallbackChainError(errors)
    raise RuntimeError("No fallbacks provided")


def run_fallback_chain(
    fallbacks: Sequence[Callable[..., Any]],
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    """Run fallback functions in order. Returns first success or raises last error."""
    kwargs = kwargs or {}
    errors: list[Exception] = []
    for fb in fallbacks:
        try:
            return fb(*args, **kwargs)
        except Exception as e:
            errors.append(e)
            continue
    if errors:
        raise FallbackChainError(errors)
    raise RuntimeError("No fallbacks provided")
