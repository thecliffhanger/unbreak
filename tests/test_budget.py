"""Tests for time budget mode."""

import time
import pytest
from datetime import timedelta
from unbreak.budget import parse_budget, BudgetManager


class TestParseBudget:
    def test_timedelta(self):
        td = timedelta(seconds=30)
        assert parse_budget(td) == td

    def test_string_seconds(self):
        assert parse_budget("30s") == timedelta(seconds=30)

    def test_string_ms(self):
        assert parse_budget("500ms") == timedelta(seconds=0.5)

    def test_string_minutes(self):
        assert parse_budget("2m") == timedelta(minutes=2)

    def test_numeric(self):
        assert parse_budget(30) == timedelta(seconds=30)
        assert parse_budget(1.5) == timedelta(seconds=1.5)

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            parse_budget("abc")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            parse_budget([])


class TestBudgetManager:
    def test_remaining_decreases(self):
        bm = BudgetManager(timedelta(seconds=1))
        bm.start()
        time.sleep(0.1)
        assert bm.remaining() < 0.95
        assert bm.remaining() > 0.0

    def test_exhausted(self):
        bm = BudgetManager(timedelta(seconds=0.05))
        bm.start()
        time.sleep(0.06)
        assert bm.is_exhausted() is True

    def test_compute_wait(self):
        bm = BudgetManager(timedelta(seconds=10))
        bm.start()
        w1 = bm.compute_wait(1)
        w2 = bm.compute_wait(2)
        assert w1 > 0
        assert w2 > 0

    def test_not_started(self):
        bm = BudgetManager(timedelta(seconds=10))
        assert bm.remaining() == 10.0
        assert bm.is_exhausted() is False
