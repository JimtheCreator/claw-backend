from typing import Optional, Set
import pandas as pd

from common.logger import logger


def prepare_ohlcv(df: pd.DataFrame, required_columns: Set[str], engine_name: str) -> Optional[pd.DataFrame]:
    """
    Shared candle-cleaning pass for analysis engines: validates required
    columns are present, coerces timestamp, sorts chronologically, drops
    duplicate timestamps, drops rows with NaNs in the required columns, and
    resets the index to a clean 0..n-1 range.

    Every engine that indexes into candles positionally (swing detection,
    market structure, order blocks, liquidity, etc.) MUST clean through
    this function with the SAME raw input DataFrame. Two engines cleaning
    independently, even with identical-looking logic, is a duplicated-
    source-of-truth risk - if the cleaning rules ever diverge, positional
    indices between engines silently stop lining up and every downstream
    "swing at index i" reference points at the wrong candle. Route
    everything through here instead.

    Returns None (never raises) on missing/invalid data so callers can
    degrade gracefully rather than crash a live analysis pipeline.
    """
    if df is None or len(df) == 0:
        logger.warning(f"[{engine_name}] Empty or missing OHLCV data.")
        return None

    missing = required_columns - set(df.columns)
    if missing:
        logger.error(f"[{engine_name}] Missing required columns: {missing}")
        return None

    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])
    cleaned = cleaned.sort_values("timestamp")
    cleaned = cleaned.drop_duplicates(subset="timestamp")
    cleaned = cleaned.dropna(subset=list(required_columns))
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        logger.warning(f"[{engine_name}] No valid rows left after cleaning.")
        return None

    return cleaned