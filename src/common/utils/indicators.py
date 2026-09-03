import pandas as pd


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Standard ATR: rolling mean of True Range, where True Range is the
    largest of (high-low), |high-prev_close|, |low-prev_close|.

    Uses min_periods=1 so it returns a usable (if noisier) estimate on
    short series instead of a run of NaNs - callers that need a
    volatility-scaled tolerance shouldn't have to special-case a cold
    start themselves.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    true_range = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=1).mean()