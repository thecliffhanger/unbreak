"""Tests for retry history."""

import time
import pytest
from unbreak.history import RetryHistory, RetryTimeline


class TestRetryHistory:
    def test_empty_history(self):
        h = RetryHistory()
        assert h.retries == 0
        assert h.total_wait == 0.0
        assert h.errors == []
        assert h.timeline == []

    def test_record_attempts(self):
        h = RetryHistory()
        h.record(1, 1.0, ConnectionError("fail"))
        h.record(2, 2.0, ConnectionError("fail"))
        h.record(3, 0.0, None)
        assert h.retries == 2  # attempts after first
        assert h.total_wait == 3.0
        assert len(h.errors) == 2
        assert len(h.timeline) == 3

    def test_timeline_entries(self):
        h = RetryHistory()
        h.record(1, 0.5, ValueError("e"))
        entry = h.timeline[0]
        assert entry.attempt == 1
        assert entry.wait_time == 0.5
        assert isinstance(entry.error, ValueError)

    def test_elapsed(self):
        h = RetryHistory()
        time.sleep(0.01)
        assert h.elapsed > 0
