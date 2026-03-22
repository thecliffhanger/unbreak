"""Coordinated jitter for distributed thundering herd prevention."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Optional


def apply_jitter(delay: float, jitter: str | float | None = None) -> float:
    """Apply jitter to a delay value.

    Args:
        delay: Base delay in seconds.
        jitter: Jitter mode:
            - None or "none": no jitter
            - "full": random jitter in [0, delay]
            - "equal": random jitter in [delay*0.5, delay*1.5]
            - "decorrelated": decorrelated jitter using previous delay
            - float: max jitter magnitude added uniformly from [-jitter, +jitter]
    """
    if jitter is None or jitter == "none":
        return delay
    if isinstance(jitter, float):
        import random
        return max(0.001, delay + (random.random() * 2 - 1) * jitter)
    if jitter == "full":
        import random
        return max(0.001, random.random() * delay)
    if jitter == "equal":
        import random
        return max(0.001, delay * (0.5 + random.random()))
    if jitter == "decorrelated":
        import random
        base = delay * 0.5
        return max(0.001, base + random.random() * (delay * 2 - base))
    raise ValueError(f"Unknown jitter mode: {jitter!r}")


def coordinated_jitter(
    delay: float,
    attempt: int,
    key: str = "default",
    window_seconds: int = 10,
) -> float:
    """Deterministic coordinated jitter for distributed retry spreading.

    Uses hash of (key + attempt + time_window) to spread retries across
    the delay range deterministically, so distributed instances don't
    all retry at the same time.

    Args:
        delay: Base delay in seconds.
        attempt: Current attempt number (1-indexed).
        key: Service/key name for coordination.
        window_seconds: Time window size for hash bucketing.
    """
    now = int(time.time() / window_seconds)
    seed = f"{key}:{attempt}:{now}"
    hash_val = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    # Normalize to [0, 1]
    fraction = (hash_val % 10000) / 10000.0
    # Spread across [0.1*delay, delay]
    return max(0.001, delay * (0.1 + 0.9 * fraction))
