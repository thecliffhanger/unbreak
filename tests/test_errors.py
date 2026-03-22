"""Tests for smart error detection."""

import socket
import pytest
from unbreak.errors import is_retryable, register_retryable, unregister_retryable, clear_custom_retryable, get_retry_after


class TestNetworkErrors:
    def test_connection_error(self):
        assert is_retryable(ConnectionError()) is True

    def test_connection_refused(self):
        assert is_retryable(ConnectionRefusedError()) is True

    def test_connection_reset(self):
        assert is_retryable(ConnectionResetError()) is True

    def test_socket_timeout(self):
        assert is_retryable(socket.timeout()) is True

    def test_timeout_error(self):
        assert is_retryable(TimeoutError()) is True

    def test_broken_pipe(self):
        assert is_retryable(BrokenPipeError()) is True

    def test_generic_error_not_retryable(self):
        assert is_retryable(ValueError("bad")) is False


class TestHTTPErrors:
    def test_429_retryable(self):
        class HTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
        assert is_retryable(HTTPError(429)) is True

    def test_500_retryable(self):
        class HTTPError(Exception):
            status_code = 500
        assert is_retryable(HTTPError()) is True

    def test_502_retryable(self):
        class HTTPError(Exception):
            status_code = 502
        assert is_retryable(HTTPError()) is True

    def test_503_retryable(self):
        class HTTPError(Exception):
            status_code = 503
        assert is_retryable(HTTPError()) is True

    def test_504_retryable(self):
        class HTTPError(Exception):
            status_code = 504
        assert is_retryable(HTTPError()) is True

    def test_404_not_retryable(self):
        class HTTPError(Exception):
            status_code = 404
        assert is_retryable(HTTPError()) is False

    def test_200_not_error(self):
        assert is_retryable(Exception()) is False


class TestDBErrors:
    def test_database_locked(self):
        assert is_retryable(Exception("database is locked")) is True

    def test_operational_error(self):
        assert is_retryable(Exception("Operational Error: could not connect")) is True

    def test_too_many_connections(self):
        assert is_retryable(Exception("FATAL: too many connections")) is True

    def test_disk_io_error(self):
        assert is_retryable(Exception("disk I/O error")) is True


class TestRetryAfter:
    def test_retry_after_header(self):
        class Response:
            headers = {"Retry-After": "5"}
        class Error(Exception):
            response = Response()
        assert get_retry_after(Error()) == 5.0

    def test_no_retry_after(self):
        assert get_retry_after(ValueError("nope")) is None


class TestCustomErrors:
    def setup_method(self):
        clear_custom_retryable()

    def test_register_custom(self):
        class MyError(Exception):
            pass
        register_retryable(MyError)
        assert is_retryable(MyError()) is True

    def test_unregister_custom(self):
        class MyError(Exception):
            pass
        register_retryable(MyError)
        unregister_retryable(MyError)
        assert is_retryable(MyError()) is False

    def teardown_method(self):
        clear_custom_retryable()
