"""Translate provider throttling into a shared cooldown without exposing URLs."""
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred


async def defer_if_throttled(limiter, status, headers):
    if status not in (418, 429):
        return
    delay = 180 if status == 418 else 60
    raw = next((v for k, v in (headers or {}).items() if k.lower() == "retry-after"), None)
    if raw is not None:
        try:
            parsed = float(raw)
        except (ValueError, TypeError):
            try:
                until = parsedate_to_datetime(raw)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                parsed = (until - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                parsed = float("nan")
        if math.isfinite(parsed):
            delay = max(1, parsed)
    await limiter.defer(delay)
    raise ProviderRequestDeferred("Provider requested a cooldown; market-data request deferred.", delay)
