# retryly

Smart retry library with adaptive backoff, circuit breaker, fallback chains, and dead letter queues.

## Quick Start

```python
import retryly

@retryly.retry(max=5, backoff="exponential")
def call_api():
    ...

# With circuit breaker
@retryly.retry(max=5, circuit=True)
def call_api():
    ...

# Time budget mode
@retryly.retry(budget="30s")
def call_api():
    ...

# Fallback chain
@retryly.retry(max=3, fallback=[redis_get, db_query, lambda: "default"])
def call_api():
    ...
```

## Features

- **Backoff strategies**: fixed, exponential, adaptive (learns from response times)
- **Circuit breaker**: sliding window, predictive degradation detection
- **Smart error detection**: HTTP, DB, network, custom
- **Time budget mode**: retries within a time window
- **Fallback chains**: cascading fallbacks
- **Dead letter queue**: in-memory, file (JSONL), custom backends
- **Coordinated jitter**: distributed thundering herd prevention
- **Observability**: structured event callbacks
- **Retry history**: per-call timeline tracking
- **Async support**: full asyncio support
- **Zero dependencies**: stdlib only
