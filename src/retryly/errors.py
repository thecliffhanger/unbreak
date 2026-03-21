"""Smart error detection for HTTP, DB, network, and custom error types."""

from __future__ import annotations

import re
import socket
import threading
from typing import Callable, Type


class _RetryableRegistry:
    """Thread-safe registry for custom retryable error types and checks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._types: set[type[Exception]] = set()
        self._checks: list[Callable[[Exception], bool]] = []

    def add_type(self, error_type: type[Exception]) -> None:
        with self._lock:
            self._types.add(error_type)

    def remove_type(self, error_type: type[Exception]) -> None:
        with self._lock:
            self._types.discard(error_type)

    def add_check(self, check: Callable[[Exception], bool]) -> None:
        with self._lock:
            self._checks.append(check)

    def is_retryable(self, error: BaseException) -> bool:
        with self._lock:
            types = set(self._types)
            checks = list(self._checks)
        for err_type in types:
            if isinstance(error, err_type):
                return True
        for check in checks:
            try:
                if check(error):
                    return True
            except Exception:
                pass
        return False

    def clear(self) -> None:
        with self._lock:
            self._types.clear()
            self._checks.clear()


_registry = _RetryableRegistry()

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


def is_retryable(error: BaseException) -> bool:
    """Determine if an error is retryable.

    Checks HTTP status codes (via attributes), network error types,
    database error patterns, and user-registered custom types.
    """
    # Check if it's a retryable network error
    if isinstance(error, _RETRYABLE_NETWORK_ERRORS):
        return True

    # Check custom registered types and predicates
    if _registry.is_retryable(error):
        return True

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
    _registry.add_type(error_type)


def register_retryable_check(check: Callable[[Exception], bool]) -> None:
    """Register a custom predicate function to determine retryability."""
    _registry.add_check(check)


def unregister_retryable(error_type: type[Exception]) -> None:
    """Remove a custom error type from the retryable set."""
    _registry.remove_type(error_type)


def clear_custom_retryable() -> None:
    """Clear all custom retryable registrations (for testing)."""
    _registry.clear()
