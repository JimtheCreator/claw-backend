"""Separate downward entry ladder; never substitute the requested interval."""
EXECUTION_LADDER = {
    '1m': [], '5m': ['1m'], '15m': ['5m', '1m'], '30m': ['5m', '1m'],
    '1h': ['15m', '5m', '1m'], '2h': ['30m', '15m', '5m'],
    '4h': ['1h', '15m', '5m'], '6h': ['1h', '15m', '5m'],
    '1d': ['4h', '1h', '15m'], '3d': ['1d', '4h', '1h'],
    '1w': ['1d', '4h', '1h'], '1M': ['1w', '1d', '4h'],
}


def get_execution_chain(interval):
    return list(EXECUTION_LADDER.get(interval, []))
