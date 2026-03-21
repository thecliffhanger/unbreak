"""Smart error detection for HTTP, DB, network, and custom error types."""

from __future__ import annotations

import re
import socket
from typing import Callable, Type

# HTTP status codes that are retryable
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# Network errors
_RETRYABLE_NETWORK_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    socket.timeout,
    TimeoutError,
    BrokenPipeError,
)

# DB error patterns
_DB_ERROR_PATTERNS: tuple[str, ...] = (
    r"operational error",
    r"interface error",
    r"database is locked",
    r"disk i/o error",
    r"connection.*refused",
    r"connection.*reset",
    r"too many connections",
    r"could not connect",
    r"connection.*timed? ?out",
    r"temporary failure",
    r"cluster is down",
    r"loading.*error",
    r"read only",
    r"readonly",
)

_DB_COMPILED = [re.compile(p, re.IGNORECASE) for p in _DB_ERROR_PATTERNS]

# User-registered custom error types
_CUSTOM_RETRYABLE: set[type[Exception]] = set()
_CUSTOM_RETRYABLE_CHECKS: list[Callable[[Exception], bool]] = []


def is_retryable(error: BaseException) -> bool:
    """Determine if an error is retryable.

    Checks HTTP status codes (via attributes), network error types,
    database error patterns, and user-registered custom types.
    """
    # Check if it's a retryable network error
    if isinstance(error, _RETRYABLE_NETWORK_ERRORS):
        return True

    # Check custom registered types
    for err_type in _CUSTOM_RETRYABLE:
        if isinstance(error, err_type):
            return True

    # Check custom predicates
    for check in _CUSTOM_RETRYABLE_CHECKS:
        try:
            if check(error):
                return True
        except Exception:
            pass

    # Check HTTP status
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status is not None:
        if isinstance(status, int) and status in _RETRYABLE_HTTP_STATUS:
            return True

    # Check DB error patterns
    msg = str(error)
    if msg:
        for pattern in _DB_COMPILED:
            if pattern.search(msg):
                return True

    return False


def get_retry_after(error: BaseException) -> float | None:
    """Extract Retry-After seconds from an HTTP error if available."""
    obj = error
    if hasattr(obj, "response"):
        obj = obj.response  # type: ignore
    for attr in ("headers", "Headers"):
        headers = getattr(obj, attr, None)
        if headers is not None:
            if isinstance(headers, dict):
                ra = headers.get("Retry-After") or headers.get("retry-after")
            else:
                ra = headers.get("Retry-After", None)  # type: ignore
            if ra is not None:
                try:
                    return float(ra)
                except (ValueError, TypeError):
                    pass
            # X-RateLimit-Reset (unix timestamp)
            rlr = (headers.get("X-RateLimit-Reset") if isinstance(headers, dict)
                    else getattr(headers, "get", lambda k: None)("X-RateLimit-Reset"))
            if rlr is not None:
                try:
                    import time
                    wait = float(rlr) - time.time()
                    if wait > 0:
                        return wait
                except (ValueError, TypeError):
                    pass
    return None


def register_retryable(error_type: type[Exception]) -> None:
    """Register a custom error type as retryable."""
    _CUSTOM_RETRYABLE.add(error_type)


def register_retryable_check(check: Callable[[Exception], bool]) -> None:
    """Register a custom predicate function to determine retryability."""
    _CUSTOM_RETRYABLE_CHECKS.append(check)


def unregister_retryable(error_type: type[Exception]) -> None:
    """Remove a custom error type from the retryable set."""
    _CUSTOM_RETRYABLE.discard(error_type)


def clear_custom_retryable() -> None:
    """Clear all custom retryable registrations (for testing)."""
    _CUSTOM_RETRYABLE.clear()
    _CUSTOM_RETRYABLE_CHECKS.clear()
