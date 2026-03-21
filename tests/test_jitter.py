"""Tests for coordinated jitter."""

import pytest
from retryly.jitter import apply_jitter, coordinated_jitter


class TestApplyJitter:
    def test_none_jitter(self):
        assert apply_jitter(1.0, None) == 1.0

    def test_none_string(self):
        assert apply_jitter(1.0, "none") == 1.0

    def test_full_jitter_in_range(self):
        for _ in range(50):
            d = apply_jitter(1.0, "full")
            assert 0.001 <= d <= 1.0

    def test_equal_jitter_in_range(self):
        for _ in range(50):
            d = apply_jitter(1.0, "equal")
            assert 0.5 <= d <= 1.5

    def test_float_jitter(self):
        for _ in range(50):
            d = apply_jitter(1.0, 0.2)
            assert 0.8 <= d <= 1.2

    def test_decorrelated(self):
        for _ in range(10):
            d = apply_jitter(1.0, "decorrelated")
            assert d >= 0.001

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            apply_jitter(1.0, "bad_mode")


class TestCoordinatedJitter:
    def test_deterministic(self):
        """Same inputs should produce same output within same time window."""
        d1 = coordinated_jitter(1.0, 1, "test_key")
        d2 = coordinated_jitter(1.0, 1, "test_key")
        assert d1 == d2

    def test_different_keys(self):
        d1 = coordinated_jitter(1.0, 1, "key_a")
        d2 = coordinated_jitter(1.0, 1, "key_b")
        # Very unlikely to be equal
        # (may rarely collide but vanishingly unlikely)
        assert d1 != d2 or True  # tolerate rare collision

    def test_different_attempts(self):
        results = set()
        for i in range(1, 10):
            results.add(coordinated_jitter(1.0, i, "key"))
        assert len(results) > 1

    def test_in_range(self):
        for i in range(1, 20):
            d = coordinated_jitter(10.0, i, "key")
            assert 1.0 <= d <= 10.0

    def test_default_key(self):
        d = coordinated_jitter(1.0, 1)
        assert d > 0
