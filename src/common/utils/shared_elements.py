from datetime import datetime, timedelta, timezone


INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 240,
    "1d": 1440,
    "3d": 1440,
    "1w": 10080, 
    "1M": 43200
}


INTERVAL_RANGES = {
    "1m": timedelta(days=90),    # 3 months
    "5m": timedelta(days=180),   # 6 months
    "15m": timedelta(days=365),  # 1 year
    "30m": timedelta(days=730),  # 2 years
    "1h": timedelta(days=730),   # 2 years
    "2h": timedelta(days=730),   # 2 years
    "4h": timedelta(days=1825),  # 5 years
    "6h": timedelta(days=1825),  # 5 years
    "1d": timedelta(days=1825),
    "3d": timedelta(days=1825),
    "1w": timedelta(days=1825),
    "1M": timedelta(days=1825)
}

def calculate_start_time(interval: str) -> datetime:
    return datetime.now(timezone.utc) - INTERVAL_RANGES.get(interval, timedelta(days=90))