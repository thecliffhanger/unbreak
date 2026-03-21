"""Fallback chains for exhausted retries."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence


async def run_fallback_chain_async(
    fallbacks: Sequence[Callable[..., Any]],
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    """Run fallback functions in order (async version). Returns first success."""
    kwargs = kwargs or {}
    last_error: BaseException | None = None
    for fb in fallbacks:
        try:
            if inspect.iscoroutinefunction(fb):
                return await fb(*args, **kwargs)
            else:
                return fb(*args, **kwargs)
        except Exception as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No fallbacks provided")


def run_fallback_chain(
    fallbacks: Sequence[Callable[..., Any]],
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    """Run fallback functions in order. Returns first success or raises last error."""
    kwargs = kwargs or {}
    last_error: BaseException | None = None
    for fb in fallbacks:
        try:
            return fb(*args, **kwargs)
        except Exception as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No fallbacks provided")
