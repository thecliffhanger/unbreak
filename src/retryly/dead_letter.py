"""Dead letter queue with pluggable backends (in-memory, file, custom)."""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Union


logger = logging.getLogger(__name__)


@dataclass
class DeadLetterEntry:
    """Record of a failed call."""
    function_name: str
    args: tuple
    kwargs: dict
    error: str
    error_type: str
    timestamp: float
    retry_count: int
    total_wait_time: float


class DeadLetterBackend:
    """Base for dead letter backends."""

    def write(self, entry: DeadLetterEntry) -> None:
        raise NotImplementedError

    def read_all(self) -> list[DeadLetterEntry]:
        raise NotImplementedError


class InMemoryBackend(DeadLetterBackend):
    """In-memory dead letter queue."""

    def __init__(self) -> None:
        self._entries: list[DeadLetterEntry] = []
        self._lock = threading.Lock()

    def write(self, entry: DeadLetterEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def read_all(self) -> list[DeadLetterEntry]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class FileBackend(DeadLetterBackend):
    """File-based dead letter queue (JSONL format)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, entry: DeadLetterEntry) -> None:
        data = {
            "function_name": entry.function_name,
            "args": entry.args,
            "kwargs": entry.kwargs,
            "error": entry.error,
            "error_type": entry.error_type,
            "timestamp": entry.timestamp,
            "retry_count": entry.retry_count,
            "total_wait_time": entry.total_wait_time,
        }
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")

    def read_all(self) -> list[DeadLetterEntry]:
        entries: list[DeadLetterEntry] = []
        if not self.path.exists():
            return entries
        with self._lock:
            with open(self.path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            # Convert lists back to tuples for args
                            if "args" in d and isinstance(d["args"], list):
                                d["args"] = tuple(d["args"])
                            entries.append(DeadLetterEntry(**{k: d[k] for k in DeadLetterEntry.__dataclass_fields__}))
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning("Skipping corrupt JSONL line %d: %s", line_num, e)
        return entries


class DeadLetterQueue:
    """Dead letter queue with pluggable backend.

    Args:
        backend: A backend instance, a "file:///path" string, or a callable.
                 If callable, it receives a DeadLetterEntry dict.
    """

    def __init__(
        self,
        backend: Union[DeadLetterBackend, str, Callable[[dict], None], None] = None,
    ) -> None:
        if backend is None:
            self._backend: DeadLetterBackend = InMemoryBackend()
        elif isinstance(backend, str):
            if backend.startswith("file://"):
                path = backend[len("file://"):]
                self._backend = FileBackend(path)
            else:
                self._backend = InMemoryBackend()
        elif callable(backend) and not isinstance(backend, DeadLetterBackend):
            # Wrap callable as backend
            self._backend = _CallableBackend(backend)
        else:
            self._backend = backend

    def record(
        self,
        function_name: str,
        args: tuple,
        kwargs: dict,
        error: BaseException,
        retry_count: int,
        total_wait_time: float,
    ) -> DeadLetterEntry:
        """Record a failed call to the dead letter queue."""
        entry = DeadLetterEntry(
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            error=str(error),
            error_type=type(error).__name__,
            timestamp=time.time(),
            retry_count=retry_count,
            total_wait_time=total_wait_time,
        )
        self._backend.write(entry)
        return entry

    def read_all(self) -> list[DeadLetterEntry]:
        return self._backend.read_all()


class _CallableBackend(DeadLetterBackend):
    def __init__(self, fn: Callable[[dict], None]) -> None:
        self._fn = fn

    def write(self, entry: DeadLetterEntry) -> None:
        data = {
            "function_name": entry.function_name,
            "args": entry.args,
            "kwargs": entry.kwargs,
            "error": entry.error,
            "error_type": entry.error_type,
            "timestamp": entry.timestamp,
            "retry_count": entry.retry_count,
            "total_wait_time": entry.total_wait_time,
        }
        self._fn(data)

    def read_all(self) -> list[DeadLetterEntry]:
        return []


def replay(
    path: str,
    registry: dict[str, Callable[..., Any]] | None = None,
) -> list[tuple[str, Any | Exception]]:
    """Read dead letter entries from a file and attempt to replay them.

    Args:
        path: Path to the JSONL dead letter file.
        registry: Optional mapping of function names to callables. If a function
                  name is in the registry, it will be called with the stored args/kwargs.
                  If not in the registry, the entry is skipped.

    Returns:
        List of (function_name, result_or_error) tuples.
    """
    backend = FileBackend(path)
    entries = backend.read_all()
    results: list[tuple[str, Any | Exception]] = []
    for entry in entries:
        if registry and entry.function_name in registry:
            fn = registry[entry.function_name]
            try:
                result = fn(*entry.args, **entry.kwargs)
                results.append((entry.function_name, result))
            except Exception as e:
                results.append((entry.function_name, e))
        else:
            results.append((entry.function_name, None))
    return results
