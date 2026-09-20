"""Serializable observed geometry. No targets, forecasts, or provider access."""
import math


def chart_candles(ohlcv):
    return [dict(index=i, timestamp=stamp, **{key: float(ohlcv[key][i])
            for key in ("open", "high", "low", "close")})
            for i, stamp in enumerate(ohlcv["timestamp"])]


def geometry(item, category, start, end, size):
    levels = item.get("key_levels") or {}
    if not isinstance(levels, dict):
        levels = {}
    points = []
    raw_points = levels.get("points") or {}
    for label, raw in (raw_points.items() if isinstance(raw_points, dict) else []):
        if not isinstance(raw, dict):
            continue
        try:
            index, price = int(raw["index"]), float(raw["price"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if 0 <= index < size and math.isfinite(price) and price > 0:
            points.append({"index": index, "price": price, "label": str(label)[:32]})
    points.sort(key=lambda p: (p["index"], p["label"]))
    lines = []
    explicit = levels.get("overlay_lines") or []
    if not isinstance(explicit, list):
        explicit = []
    for line in explicit[:8]:
        try:
            pair = [{"index": int(p["index"]), "price": float(p["price"])} for p in line]
            if len(pair) == 2 and all(0 <= p["index"] < size and math.isfinite(p["price"]) and p["price"] > 0 for p in pair):
                lines.append(pair)
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    name = item["pattern_name"]
    if not lines and "rectangle" in name:
        for boundary in ("support", "resistance"):
            price = levels.get(boundary)
            if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
                lines.append([{"index": x, "price": float(price)} for x in (start, size - 1)])
    if not lines and any(family in name for family in ("channel", "triangle", "wedge", "rectangle")):
        # The detector's observed pivot sets; regression reproduces its fitted
        # boundaries. Never substitute an idealized pattern template on a chart.
        for prefix in ("peak", "trough"):
            group = [p for p in points if p["label"].startswith(prefix)]
            if len(group) < 2:
                continue
            mx = sum(p["index"] for p in group) / len(group)
            my = sum(p["price"] for p in group) / len(group)
            denominator = sum((p["index"] - mx) ** 2 for p in group)
            if not denominator:
                continue
            slope = sum((p["index"] - mx) * (p["price"] - my) for p in group) / denominator
            lines.append([{"index": x, "price": my + slope * (x - mx)} for x in (start, size - 1)])
    if not lines and len(points) >= 2:
        lines = [[a, b] for a, b in zip(points, points[1:])]
    return {"start_index": start, "end_index": end, "category": category,
            "points": points[:64], "lines": lines[:64]}
