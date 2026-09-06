"""Use settled candles for structural confirmation, including monthly bars."""
import re

import pandas as pd


def closed_candles(candles, interval, as_of=None):
    frame = candles.copy()
    if frame.empty:
        return frame
    match = re.fullmatch(r"([1-9]\d*)([mhdwM])", interval)
    if not match:
        raise ValueError(f"Unsupported analysis interval: {interval}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    count, unit = int(match[1]), match[2]
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if unit == "M":
        close_time = frame.timestamp.map(lambda t: t + pd.DateOffset(months=count))
    else:
        close_time = frame.timestamp + pd.Timedelta(**{{"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]: count})
    return frame.loc[close_time <= now].reset_index(drop=True)
