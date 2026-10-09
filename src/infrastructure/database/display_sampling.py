"""Legacy chart sampling sizes, shared across the storage transition.

These are display samples of existing candles, not new OHLCV intervals.
Analyzer and scanner inputs must continue to read the original candles.
"""


def display_window_seconds(date_range):
    seconds = max(int(date_range.total_seconds() / 300), 60)
    unit = 60 if seconds < 3600 else 3600 if seconds < 86400 else 86400
    return seconds // unit * unit
