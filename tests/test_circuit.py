"""Tests for circuit breaker."""

import time
import pytest
from retryly.circuit import CircuitBreaker, CircuitState


class TestCircuitBreakerBasics:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, window_size=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_success_resets_window(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Only 2 failures in window now
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow() is True  # one permit

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_sliding_window(self):
        cb = CircuitBreaker(failure_threshold=3, window_size=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Fill window with successes
        cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestPredictiveCircuitBreaker:
    def test_no_predict_without_flag(self):
        cb = CircuitBreaker(predictive=False)
        assert cb.record_response_time(100.0) is False

    def test_predictive_needs_warmup(self):
        cb = CircuitBreaker(predictive=True, predictive_threshold=2.0)
        for _ in range(3):
            assert cb.record_response_time(0.1) is False

    def test_detects_slowdown(self):
        cb = CircuitBreaker(predictive=True, predictive_threshold=1.0)
        # Train with fast responses
        for _ in range(20):
            cb.record_response_time(0.001)
        # Sudden slowdown
        result = cb.record_response_time(1.0)
        assert result is True


class TestPatternLearning:
    def test_reduces_threshold(self):
        cb = CircuitBreaker(failure_threshold=5, window_size=10)
        # Train that attempt 3 always fails
        for _ in range(5):
            cb.record_attempt(3, False)
            cb.record_attempt(1, True)
            cb.record_attempt(2, True)
        # Now threshold should be reduced
        threshold = cb._learned_threshold()
        assert threshold < 5
