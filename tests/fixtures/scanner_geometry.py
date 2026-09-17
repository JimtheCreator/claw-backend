"""Deterministic geometry examples; synthetic labels are not market accuracy evidence."""
import math

from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import LOOKBACK, utc_iso


def geometry_rows(recipe, *, scale=1.0, cutoff=1789646400, interval="15m"):
    kind = recipe["kind"]
    tails = {
        "engulfing": [[101, 101.2, 100.1, 100.3], [100, 102.7, 99.8, 102.5]],
        "hammer": [[100, 101.1, 96, 101]],
        "standard_doji": [[100, 102, 98, 100]],
        "dragonfly_doji": [[100, 100, 96, 100]],
        "gravestone_doji": [[100, 104, 100, 100]],
        "morning_star": [[105, 106, 100, 100.5], [99, 100, 98, 99], [100.5, 105, 100, 104.5]],
    }
    if kind not in {"envelope", "rectangle", "flat", "trend", "pivots", "flag"} | tails.keys():
        raise ValueError(f"Unknown geometry recipe: {kind}")
    if kind == "pivots":
        pivots = recipe["points"]
        assert pivots[0][0] == 0 and pivots[-1][0] == LOOKBACK - 1
        assert all(a[0] < b[0] for a, b in zip(pivots, pivots[1:]))
        segment = 0
    rows = []
    for i in range(LOOKBACK):
        if kind == "envelope":
            origin = recipe.get("origin", 150)
            upper = 104 + recipe["upper_slope"] * (i - origin)
            lower = 96 + recipe["lower_slope"] * (i - origin)
            price = (upper + lower) / 2 + (upper - lower) / 2 * math.sin(2 * math.pi * (i - 2) / 20)
            values = [price, price + .05, price - .05, price, 1000.0]
        elif kind == "rectangle":
            price = 100 + .9 * math.sin(2 * math.pi * (i - 2) / 20)
            high = 101 if price > 100.6 else price + .05
            low = 99 if price < 99.4 else price - .05
            values = [price, high, low, price, 1000.0]
        elif kind == "flat":
            values = [100.0, 100.0, 100.0, 100.0, 1000.0]
        elif kind == "pivots":
            while i > pivots[segment + 1][0]:
                segment += 1
            (left, start), (right, end) = pivots[segment:segment + 2]
            price = start + (end - start) * (i - left) / (right - left)
            wick = recipe.get("wick", 0.0)
            values = [price, price + wick, price - wick, price, 1000.0]
        elif kind == "flag":
            pole_start = recipe.get("pole_start", 150)
            pole_end = recipe.get("pole_end", 210)
            gain = recipe.get("gain", 20)
            if i <= pole_end:
                price = 100 + gain * max(0, i - pole_start) / (pole_end - pole_start)
            else:
                offset = i - pole_end
                price = (100 + gain - 2 + recipe.get("slope", -.08) * offset
                         + 2 * math.cos(2 * math.pi * offset / 10))
            values = [price, price + .05, price - .05, price, 1000.0]
        else:
            price = 130 - i * .12
            values = [price, price + .1, price - .3, price - .2, 1000.0]
        rows.append(dict(zip(("open", "high", "low", "close", "volume"), values)))
    if kind in tails:
        tail = tails[kind]
        for row, values in zip(rows[-len(tail):], tail):
            row.update(zip(("open", "high", "low", "close"), values))
    if recipe.get("mirror"):
        for row in rows:
            row.update(open=200-row["open"], high=200-row["low"],
                       low=200-row["high"], close=200-row["close"])
    if recipe.get("wrong_trend"):
        for row in rows[:-1]:
            row.update(open=200-row["open"], high=200-row["low"],
                       low=200-row["high"], close=200-row["close"])
    if recipe.get("large_body"):
        rows[-1].update(open=98.5, close=101.5)
    if recipe.get("wrong_final_direction"):
        rows[-1]["open"], rows[-1]["close"] = rows[-1]["close"], rows[-1]["open"]
    step = INTERVAL_SECONDS[interval]
    for i, row in enumerate(rows):
        for field in ("open", "high", "low", "close"):
            row[field] *= scale
        row["timestamp"] = utc_iso(cutoff - (LOOKBACK - i) * step)
    return rows
