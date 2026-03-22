"""Tests for dead letter queue."""

import json
import os
import pytest
import tempfile
from unbreak.dead_letter import DeadLetterQueue, InMemoryBackend, FileBackend, replay


class TestInMemoryBackend:
    def test_write_and_read(self):
        dlq = DeadLetterQueue()
        dlq.record("my_func", (1, 2), {"x": 3}, ValueError("bad"), 3, 1.5)
        entries = dlq.read_all()
        assert len(entries) == 1
        assert entries[0].function_name == "my_func"
        assert entries[0].retry_count == 3
        assert entries[0].error_type == "ValueError"

    def test_multiple_entries(self):
        dlq = DeadLetterQueue()
        dlq.record("f1", (), {}, ValueError("a"), 1, 0.0)
        dlq.record("f2", (), {}, TypeError("b"), 2, 0.5)
        assert len(dlq.read_all()) == 2


class TestFileBackend:
    def test_file_write_and_read(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            dlq = DeadLetterQueue(f"file://{path}")
            dlq.record("func", (1,), {}, RuntimeError("err"), 5, 2.0)
            entries = dlq.read_all()
            assert len(entries) == 1
            assert entries[0].function_name == "func"

            # Verify file format
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["function_name"] == "func"
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            os.unlink(path)
            dlq = DeadLetterQueue(f"file://{path}")
            assert dlq.read_all() == []
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestCallableBackend:
    def test_callable_backend(self):
        collected = []
        def handler(entry):
            collected.append(entry)
        dlq = DeadLetterQueue(handler)
        dlq.record("fn", (), {}, ValueError("e"), 1, 0.0)
        assert len(collected) == 1
        assert collected[0]["function_name"] == "fn"


class TestReplay:
    def test_replay_reads_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            dlq = DeadLetterQueue(f"file://{path}")
            dlq.record("fn", (), {}, ValueError("e"), 1, 0.0)
            results = replay(path)
            assert len(results) == 1
            # Without registry, returns (function_name, None)
            assert results[0] == ("fn", None)
            # With registry, function is called
            results2 = replay(path, registry={"fn": lambda: "replayed"})
            assert len(results2) == 1
            assert results2[0] == ("fn", "replayed")
        finally:
            os.unlink(path)
