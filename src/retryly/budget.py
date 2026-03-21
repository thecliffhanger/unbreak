"""Time-budget mode for retries."""

from __future__ import annotations

import re
import time
from datetime import timedelta
from typing import Union


def parse_budget(budget: Union[str, timedelta, float, int]) -> timedelta:
    """Parse a budget value into a timedelta.

    Accepts: timedelta, numeric (seconds), or string like '30s', '500ms', '1m'.
    """
    if isinstance(budget, timedelta):
        return budget
    if isinstance(budget, (int, float)):
        return timedelta(seconds=float(budget))
    if isinstance(budget, str):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", budget.strip())
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            if unit == "ms":
                return timedelta(seconds=val / 1000)
            if unit == "s":
                return timedelta(seconds=val)
            if unit == "m":
                return timedelta(minutes=val)
            if unit == "h":
                return timedelta(hours=val)
        raise ValueError(f"Cannot parse budget: {budget!r}. Use '30s', '500ms', '1m', etc.")
    raise TypeError(f"Invalid budget type: {type(budget)}")


class BudgetManager:
    """Manages time-budget-based retry scheduling.

    Distributes wait times so earlier attempts are shorter and later ones longer,
    maximizing the chance of success within the budget window.
    """

    def __init__(self, total_budget: timedelta) -> None:
        self.total_budget = total_budget
        self.total_seconds = total_budget.total_seconds()
        self.started_at: float | None = None

    def start(self) -> None:
        self.started_at = time.monotonic()

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed())

    def is_exhausted(self) -> bool:
        return self.elapsed() >= self.total_seconds

    def compute_wait(self, attempt: int, max_attempts: int = 10) -> float:
        """Compute wait time for *attempt*, distributing budget optimally.

        Earlier attempts get shorter waits; later attempts get longer waits,
        since we want to maximize total attempts within the budget.
        """
        if self.started_at is None:
            self.start()
        rem = self.remaining()
        if rem <= 0:
            return 0.0
        # Give remaining attempts equal share, but weight later ones more
        # Use exponential distribution: each attempt gets proportionally more
        remaining_attempts = max(1, max_attempts - attempt + 1)
        # Share budget with bias toward longer waits for later attempts
        share = rem / (remaining_attempts + 1)  # +1 to leave room
        # Apply exponential growth to the share
        factor = 1.0 + (attempt - 1) * 0.5
        wait = min(share * factor, rem * 0.8)
        return max(0.001, wait)
