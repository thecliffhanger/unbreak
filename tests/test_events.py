"""Tests for event emission."""

import pytest
from retryly.events import EventType, RetryEvent, EventDispatcher


class TestRetryEvent:
    def test_creation(self):
        e = RetryEvent(type=EventType.RETRY_ATTEMPT, function_name="fn", attempt=2)
        assert e.type == EventType.RETRY_ATTEMPT
        assert e.function_name == "fn"
        assert e.attempt == 2
        assert e.timestamp > 0

    def test_error_type_auto(self):
        err = ValueError("bad")
        e = RetryEvent(type=EventType.RETRY_EXHAUSTED, function_name="fn", error=err)
        assert e.error_type == "ValueError"


class TestEventDispatcher:
    def test_callback_receives_event(self):
        received = []
        def handler(event):
            received.append(event)

        d = EventDispatcher(handler)
        e = RetryEvent(type=EventType.RETRY_ATTEMPT, function_name="fn", attempt=1)
        d.emit(e)
        assert len(received) == 1
        assert received[0].type == EventType.RETRY_ATTEMPT

    def test_handler_error_doesnt_propagate(self):
        def bad_handler(event):
            raise RuntimeError("handler error")

        d = EventDispatcher(bad_handler)
        d.emit(RetryEvent(type=EventType.RETRY_ATTEMPT, function_name="fn"))  # no error

    def test_multiple_callbacks(self):
        received = []
        d = EventDispatcher()
        d.add_callback(lambda e: received.append("a"))
        d.add_callback(lambda e: received.append("b"))
        d.emit(RetryEvent(type=EventType.RETRY_ATTEMPT, function_name="fn"))
        assert received == ["a", "b"]

    def test_no_callbacks(self):
        d = EventDispatcher()
        d.emit(RetryEvent(type=EventType.RETRY_ATTEMPT, function_name="fn"))  # no error
