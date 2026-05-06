"""
queue package

This package contains the async queue layer.

It will handle:
- slow tool dispatch
- Redis-backed job storage
- idempotency
- retries
- dead-letter queue
"""