# retryly Code Review

## Summary

Well-structured retry library with adaptive backoff, circuit breaker, fallback chains, dead letter queues, and budget mode. The architecture is clean and the feature set is impressive for v0.1. Below are the issues found, ordered by severity.

---

## 🔴 Critical Issues

### 1. `BudgetManager.start()` called every iteration when budget is set

**File:** `retry/retry.py` — `_do_retry()` and `_do_retry_async()`

```python
if bm:
    if bm.is_exhausted():
        break
    bm.start()
```

`bm.start()` is called on **every** loop iteration when the budget isn't exhausted, resetting `started_at` each time. This means `elapsed()` only measures time since the last iteration, not total time. The budget will effectively never exhaust.

**Fix:** Only call `bm.start()` once before the loop (or check if `started_at is None`).

### 2. Async circuit breaker uses `time.sleep` — blocking the event loop

**File:** `retry/retry.py` — `_do_retry_async()` line with `await asyncio.sleep(wait)`

The async path correctly uses `await asyncio.sleep(wait)` for the retry delay, but if `backoff_strategy.compute()` or `jitter` functions are slow, they block the event loop. More critically, the `CircuitBreaker.record_response_time()` does not hold the GIL in a way that's async-safe — while `threading.Lock` works in asyncio single-threaded mode, the `time.monotonic()` calls and lock acquisitions add unnecessary overhead. **This is acceptable but not ideal.** The real issue is: `_should_stop` and `_compute_wait` are sync functions that could do blocking work in the async path.

### 3. `record_attempt()` not thread-safe

**File:** `circuit.py` — `record_attempt()`

```python
def record_attempt(self, attempt: int, success: bool) -> None:
    self._attempt_total_counts[attempt] = self._attempt_total_counts.get(attempt, 0) + 1
    if not success:
        self._attempt_fail_counts[attempt] = self._attempt_fail_counts.get(attempt, 0) + 1
```

This method accesses `_attempt_total_counts` and `_attempt_fail_counts` without holding `self._lock`, while `_learned_threshold()` reads them under the lock. Race condition between concurrent calls.

**Fix:** Acquire `self._lock` at the start of `record_attempt()`.

### 4. `record_response_time()` not thread-safe

**File:** `circuit.py` — `record_response_time()`

Multiple fields (`_response_times`, `_response_time_ema`, `_response_time_var`) are read/written without any lock. If `predictive=True`, concurrent calls to the same `CircuitBreaker` will corrupt state.

**Fix:** Acquire `self._lock` for all shared state access in this method.

### 5. `replay()` is a stub that can never work

**File:** `dead_letter.py` — `replay()`

```python
def replay(path: str) -> list[Any]:
    for entry in entries:
        try:
            results.append(("replay_not_implemented", entry.function_name))
        except Exception as e:
            results.append(e)
```

The docstring says "attempt to replay them" but the implementation just returns placeholder tuples. The `try/except` never catches anything. This is misleading — the function appears in `__all__` and the public API but does nothing useful.

**Fix:** Either implement actual replay (e.g., accept a registry mapping function names to callables) or mark it with `@deprecated` / remove from `__all__`.

---

## 🟡 Important Issues

### 6. `_learned_threshold()` can lower threshold below 1

**File:** `circuit.py`

```python
threshold = max(2, threshold - 1)
```

This is called for **each** attempt number with >80% failure rate. If multiple attempts qualify, threshold gets decremented multiple times in a single call (the loop has no `break`), potentially reducing it by more than intended.

**Fix:** Track the reduction and apply it once, or break after first match.

### 7. Global mutable state in `errors.py` is not thread-safe

**File:** `errors.py`

```python
_CUSTOM_RETRYABLE: set[type[Exception]] = set()
_CUSTOM_RETRYABLE_CHECKS: list[Callable[[Exception], bool]] = []
```

These module-level globals are modified by `register_retryable()`/`unregister_retryable()`/`clear_custom_retryable()` and read by `is_retryable()` with no synchronization. Concurrent registration + checking can cause issues (e.g., iterating a list being appended to).

**Fix:** Add a `threading.Lock` around access, or note that registration should happen at startup only.

### 8. `FileBackend.read_all()` not robust against partial/corrupt lines

**File:** `dead_letter.py`

```python
for line in f:
    line = line.strip()
    if line:
        d = json.loads(line)
```

If the process crashes mid-write, a partial JSON line will cause `json.JSONDecodeError` and abort the entire read, losing all subsequent valid entries.

**Fix:** Wrap `json.loads(line)` in try/except, skip/log corrupt lines.

### 9. Duplicate timeline recording on `until` condition not met

**File:** `retry/retry.py` — sync `_do_retry()`

```python
if until and not until(result):
    # ...
    if _should_stop(attempt, max, bm):
        history.record(attempt, 0.0, None)  # <-- first record
        break
    wait = _compute_wait(...)
    total_wait += wait
    history.record(attempt, wait, None)  # <-- second record (overwritten by append)
    time.sleep(wait)
    continue
```

When `_should_stop` is False, the timeline records the attempt **twice**: once with `wait_time=0.0` and again with the actual wait. The first record is wasted. The async path doesn't have the first `history.record` call, creating an inconsistency.

**Fix:** Only record once per attempt. Move the `history.record` after the `wait` computation, remove the duplicate.

### 10. `stats.consecutive_failures` is actually `failures_in_window`

**File:** `circuit.py` — `CircuitStats`

```python
consecutive_failures=self._failures_in_window(),
```

This counts failures in the sliding window, not consecutive failures. The field name is misleading.

**Fix:** Rename to `window_failures` or implement actual consecutive failure tracking.

### 11. `_do_retry` / `_do_retry_async` are large duplicated functions

**File:** `retry/retry.py`

These two functions are ~100 lines each and nearly identical. Any bug fix must be applied to both. This is a maintenance hazard.

**Fix:** Extract shared logic into a common helper, or use a thin async/sync adapter pattern.

### 12. No input validation on `ExponentialBackoff` and `FixedBackoff`

**File:** `backoff.py`

Negative or zero `base`, `delay`, `factor`, `max_delay` will produce nonsensical results silently (e.g., `max(0, -5)` = 0 with no wait, or negative factors).

**Fix:** Validate in `__init__` and raise `ValueError` for non-positive values.

### 13. `CircuitBreakerOpenError` not exported

**File:** `retry/retry.py` defines it, `__init__.py` doesn't export it. Users can't catch it by name without importing from `retryly.retry`.

### 14. `AdaptiveBackoff.compute()` uses `id(self)` — shared across all functions

**File:** `backoff.py`

```python
def compute(self, attempt: int) -> float:
    state = self._get_state(id(self))
```

When used without `compute_for()`, all functions sharing the same `AdaptiveBackoff` instance share EMA state. The `compute()` method is effectively useless for multi-function use.

---

## 🟢 Minor Issues

### 15. `EventDispatcher.emit` silently swallows handler exceptions

**File:** `events.py`

```python
except Exception:
    pass  # Don't let event handlers break the retry loop
```

This is intentional but makes debugging hard. Consider logging at minimum.

### 16. `clear_custom_retryable()` not exported

**File:** `errors.py` defines it, `__init__.py` doesn't export it. Useful for testing but not accessible from the public API.

### 17. `register_retryable_check()` not exported

Same as above — defined in `errors.py` but not in `__init__.py` or `__all__`.

### 18. `RetryHistory.retries` off-by-one semantic ambiguity

```python
@property
def retries(self) -> int:
    return max(0, len(self._timeline) - 1)
```

If there's 1 timeline entry (first attempt failed), `retries = 0`. But a "retry" implies a second attempt. The property name could confuse users — it returns "retries performed so far" which equals `attempts - 1`.

### 19. `coordinated_jitter` only spreads forward, not symmetrically

**File:** `jitter.py`

```python
return max(0.001, delay * (0.1 + 0.9 * fraction))
```

Range is `[0.1*delay, delay]`. For a 10s delay, retries are spread across [1s, 10s]. Missing the `[delay, 1.5*delay]` upper range that some use cases might want. Consider documenting or making configurable.

### 20. `__all__` missing `get_backoff` and `parse_budget`

Both are public utility functions but not listed in `__all__`.

### 21. No `__repr__` on `CircuitBreaker` or `BudgetManager`

Makes debugging harder. Consider adding state-aware `__repr__`.

### 22. Dead letter `args` tuple serializes as JSON array, which round-trips as list

**File:** `dead_letter.py` — `FileBackend.read_all()`

`json.dumps({"args": (1, 2)})` → `{"args": [1, 2]}` → `args` comes back as `list`, not `tuple`. The `DeadLetterEntry` dataclass expects `tuple`.

**Fix:** Cast back to tuple in `read_all()`: `tuple(d.get("args", []))`.

### 23. Missing docstrings on several public functions

`get_retry_after()`, `clear_custom_retryable()`, `register_retryable_check()` lack docstrings.

### 24. `stats.last_failure_time` and `stats.last_success_time` are never set

**File:** `circuit.py` — `CircuitStats` dataclass has these fields but `CircuitBreaker.stats` never populates them.

---

## Observability Notes

- `CIRCUIT_HALF_OPEN` event type exists but is never emitted (transition happens inside `_maybe_transition` which has no dispatch).
- Predictive circuit opening emits `CIRCUIT_OPEN` but doesn't actually change state — it just calls `record_failure()`. The event says "circuit open" but the state may still be CLOSED.
- `CIRCUIT_CLOSED` event only fires in sync path, not async path (missing the state transition check in async `_do_retry_async`).

---

## Test Coverage Gaps

- No tests for thread safety of circuit breaker or adaptive backoff
- No tests for `replay()` (understandable given it's a stub)
- No tests for partial/corrupt JSONL in `FileBackend`
- No tests for `clear_custom_retryable()` or `register_retryable_check()`
- No async integration tests for circuit breaker + budget together
