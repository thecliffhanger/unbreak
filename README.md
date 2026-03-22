# unbreak

[![PyPI version](https://badge.fury.io/py/unbreak.svg)](https://pypi.org/project/unbreak)
[![Python versions](https://img.shields.io/pypi/pyversions/unbreak.svg)](https://pypi.org/project/unbreak)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Smart retry library with adaptive backoff, circuit breaker, fallback chains, and dead letter queues.

## Install

```bash
pip install unbreak
```

## Quick Start

```python
import unbreak

@unbreak.retry(max=5, backoff="exponential")
def call_api():
    ...

# With circuit breaker
@unbreak.retry(max=5, circuit=True)
def call_api():
    ...

# Time budget mode
@unbreak.retry(budget="30s")
def call_api():
    ...

# Fallback chain
@unbreak.retry(max=3, fallback=[redis_get, db_query, lambda: "default"])
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

## License

MIT

---

Part of the [thecliffhanger](https://github.com/thecliffhanger) open source suite.
