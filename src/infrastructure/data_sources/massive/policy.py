"""Deployment policy, distinct from Massive's subscription entitlements.

Paid Currencies has no five-request/minute quota. These configurable ceilings
bound our own backfill load; cooldowns and Redis coordination still fail closed.
"""
import os
from infrastructure.database.redis.rate_limiter import RedisRateLimiter


def positive_setting(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def rest_budget(redis_client=None):
    return RedisRateLimiter(
        max_per_minute=positive_setting("MASSIVE_REST_REQUESTS_PER_MINUTE", 600),
        max_per_second=positive_setting("MASSIVE_REST_REQUESTS_PER_SECOND", 10),
        key_prefix="massive_rl", max_wait_seconds=30, redis_client=redis_client)


def websocket_budget(redis_client):
    # Reconnects, authentication and subscription messages all consume this
    # shared application budget, including unsuccessful attempts.
    return RedisRateLimiter(max_per_minute=30, max_per_second=2,
        key_prefix="massive_ws_control", max_wait_seconds=10,
        redis_client=redis_client)
