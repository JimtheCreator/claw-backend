"""Turn structural analysis into an explicit, conservative trade plan.

The engines describe facts (swings, structure, zones); this module decides
whether those facts support a *conditional* setup.  It deliberately defaults
to ``wait``: the API must not turn incomplete or conflicting evidence into a
buy/sell instruction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from math import isfinite

import pandas as pd

from common.utils.indicators import average_true_range


MIN_RISK_REWARD = 1.5


def build_trade_plan(
    candles: pd.DataFrame,
    *,
    interval: str,
    structure: Any,
    premium_discount: Any,
    liquidity: Any,
    fvg: Any,
    order_blocks: Any,
    confluence: Any,
    mtfa: Dict[str, Any],
    swings: Any = None,
) -> Dict[str, Any]:
    """Return a JSON-safe execution plan; never invent a trade on weak data."""
    current_price = _current_price(candles)
    trend = getattr(structure, "trend", None)
    base = {
        "interval": interval,
        "trend_direction": trend or "undetermined",
        "current_price": current_price,
        "action": "wait",
        "entry_level": None,
        "entry_zone": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_per_unit": None,
        "risk_reward": None,
        "wait_for_confirmation": True,
        "confirmation_required": None,
        "reason": None,
        "evidence": {"mtfa": mtfa},
        "primary_scenario": None,
    }

    if current_price is None:
        return _wait(base, "No valid closing price is available; no trade plan can be formed.")
    if trend not in {"bullish", "bearish"}:
        return _wait(base, "Market structure has no confirmed directional trend yet.")

    context, bias, explanation = _market_context(mtfa, trend)
    base["market_context"] = context
    base["context_summary"] = explanation
    if context in {"incomplete", "mixed"}:
        return _wait(base, explanation)

    # Interpret counter-trend structure as a possible pullback, NOT a reversal
    # prediction or permission to enter against the still-active local trend.
    direction = bias if context == "pullback" else trend
    scenario = _structure_watch(candles, swings, direction, liquidity, context)
    if scenario and context == "pullback":
        intermediate = [tf for tf, value in mtfa.get("htf_trends", {}).items() if value != bias]
        if intermediate:
            scenario["extra_confirmation"] = f"Also require {', '.join(intermediate)} structure to turn {bias} before entry."
            scenario["confirmation"] += " " + scenario["extra_confirmation"]
    base["primary_scenario"] = scenario
    if context == "pullback":
        return _wait(base, explanation)

    expected_zone = "discount" if trend == "bullish" else "premium"
    if not getattr(premium_discount, "range_available", False):
        return _wait(base, "The current dealing range is not confirmed, so price location cannot be assessed.")
    if getattr(premium_discount, "zone", None) != expected_zone:
        return _wait(
            base,
            f"{trend.title()} structure is present, but price is not in {expected_zone}; wait for a retrace.",
        )

    entry = _select_entry_zone(trend, current_price, confluence, order_blocks, fvg)
    if entry is None:
        return _wait(base, f"No fresh {trend} entry zone is available at or beyond current price.")

    entry_level = entry["top"] if trend == "bullish" else entry["bottom"]
    atr = _atr(candles, current_price)
    buffer = max(atr * 0.25, current_price * 0.0005)
    stop = entry["bottom"] - buffer if trend == "bullish" else entry["top"] + buffer
    target = _select_target(trend, current_price, liquidity, premium_discount, candles)
    if target is None:
        return _wait(base, "No logical opposing-liquidity target is available; risk/reward cannot be evaluated.")

    risk = abs(entry_level - stop)
    reward = (target - entry_level) if trend == "bullish" else (entry_level - target)
    if risk <= 0 or reward <= 0:
        return _wait(base, "The available target is invalid relative to the proposed entry and stop.")
    risk_reward = round(reward / risk, 2)
    if risk_reward < MIN_RISK_REWARD:
        return _wait(
            base,
            f"Best available setup has only {risk_reward}R reward/risk, below the {MIN_RISK_REWARD}R minimum.",
        )

    trigger = (
        "Wait for a candle to reject the zone and close back above the entry level "
        "with a bullish structure confirmation."
        if trend == "bullish"
        else "Wait for a candle to reject the zone and close back below the entry level "
        "with a bearish structure confirmation."
    )
    base.update({
        "action": "long" if trend == "bullish" else "short",
        "entry_level": _round(entry_level),
        "entry_zone": {"bottom": _round(entry["bottom"]), "top": _round(entry["top"]), "source": entry["source"]},
        "stop_loss": _round(stop),
        "take_profit": _round(target),
        "risk_per_unit": _round(risk),
        "risk_reward": risk_reward,
        "confirmation_required": trigger,
        "reason": "Directional structure, price location, fresh zone, target, and MTFA context meet the setup rules.",
        "evidence": {**base["evidence"], "premium_discount_zone": expected_zone},
        "primary_scenario": {
            "kind": "conditional_entry", "direction": trend,
            "title": f"{trend.title()} continuation setup",
            "trigger": _round(entry_level), "target": _round(target),
            "invalidation": _round(stop), "setup": True,
            "confirmation": trigger,
            "alternative": f"Cancel the setup if price breaches the stop at {_round(stop)}; reassess structure.",
        },
    })
    return base


def _wait(plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
    plan["reason"] = reason
    scenario = plan.get("primary_scenario")
    plan["confirmation_required"] = scenario["confirmation"] if scenario else "Wait for confirmed structure and complete context before entering."
    return plan


def _market_context(mtfa, trend):
    if not mtfa.get("enabled") or mtfa.get("context") == "no_higher_timeframe":
        return "local", trend, "Requested-timeframe structure only; no higher-timeframe confirmation."
    trends = mtfa.get("htf_trends") or {
        tf: trend if aligned is True else ("bearish" if trend == "bullish" else "bullish") if aligned is False else None
        for tf, aligned in mtfa.get("htf_trend_alignment", {}).items()
    }
    missing = set(mtfa.get("htf_requested", [])) - set(trends)
    if mtfa.get("htf_unavailable") or missing or not trends or any(t not in {"bullish", "bearish"} for t in trends.values()):
        return "incomplete", None, "Higher-timeframe evidence is incomplete. No directional forecast until that context is resolved."
    # Respect hierarchy: a nearest-HTF correction inside two aligned larger
    # frames is a nested pullback, not equivalent to the largest frames splitting.
    def duration(tf):
        units = {"m": 1, "h": 60, "d": 1440, "w": 10080, "M": 43200}
        return int(tf[:-1]) * units[tf[-1]]
    ordered = sorted(trends, key=duration)
    bias = trends[ordered[-1]]
    if len(set(trends.values())) > 1:
        if len(ordered) < 3 or trends[ordered[-2]] != bias:
            return "mixed", None, "The largest higher timeframes are split; there is no shared directional bias. Wait for structure to align, not a presumed reversal."
        return "pullback", bias, f"The larger timeframes are {bias}, but nearer structure is correcting. Watch for {bias} resumption; both local and intermediate structure must confirm."
    if bias != trend:
        return "pullback", bias, f"{trend.title()} local structure inside a {bias} higher-timeframe trend: possible pullback, not a confirmed reversal."
    return "aligned", bias, f"Local and higher-timeframe structure are {bias}. Look for continuation only after confirmation."


def _structure_watch(candles, swings, direction, liquidity, context):
    """Use an unbroken, confirmed structural pivot; never manufacture a path."""
    bullish = direction == "bullish"
    pivot_type = "high" if bullish else "low"
    price = float(candles.close.iloc[-1])
    pivots = sorted((s for s in getattr(swings, "swings", [])
                     if s.confirmed and s.type == pivot_type and 0 <= s.index < len(candles)),
                    key=lambda s: s.index, reverse=True)
    for pivot in pivots:
        level = float(pivot.price)
        after = candles.iloc[pivot.index + 1:]
        broken = (after.close > level).any() if bullish else (after.close < level).any()
        if not isfinite(level) or broken or not (level > price if bullish else level < price):
            continue
        leg = candles.iloc[pivot.index:]
        invalidation = float(leg.low.min() if bullish else leg.high.max())
        if not (invalidation < price if bullish else invalidation > price):
            continue
        target = _select_target(direction, level, liquidity, None, candles)
        crossing, holding = ("above", "holds above") if bullish else ("below", "holds below")
        kind = "pullback_reversal_watch" if context == "pullback" else "structure_break_watch"
        return {
            "kind": kind, "direction": direction, "setup": False,
            "title": f"Potential {direction} resumption" if context == "pullback" else f"{direction.title()} structure watch",
            "trigger": _round(level), "target": _round(target) if target is not None else None,
            "invalidation": _round(invalidation),
            "confirmation": f"Wait for a {crossing} {level:g} candle close, then a retest that {holding} {level:g}. Reassess entry, stop and reward/risk.",
            "alternative": f"If the {'pullback low' if bullish else 'rally high'} at {invalidation:g} breaks first, cancel this watch; the local move may extend. This alone does not reverse the higher-timeframe trend.",
        }
    return None


def _select_entry_zone(
    trend: str, price: float, confluence: Any, order_blocks: Any, fvg: Any
) -> Optional[Dict[str, float]]:
    direction = trend
    candidates = []
    for source, zones in (
        ("imbalance_order_block", getattr(confluence, "zones", [])),
        ("order_block", getattr(order_blocks, "zones", [])),
        ("fair_value_gap", getattr(fvg, "zones", [])),
    ):
        for zone in zones:
            if getattr(zone, "type", None) != direction or not _is_fresh(zone, source):
                continue
            bottom, top = float(zone.bottom), float(zone.top)
            # A long retracement zone must not be above price; the mirror applies to shorts.
            if trend == "bullish" and bottom > price:
                continue
            if trend == "bearish" and top < price:
                continue
            distance = max(price - top, 0.0) if trend == "bullish" else max(bottom - price, 0.0)
            candidates.append((0 if source == "imbalance_order_block" else 1, distance, bottom, top, source))
    if not candidates:
        return None
    _, _, bottom, top, source = min(candidates)
    return {"bottom": bottom, "top": top, "source": source}


def _is_fresh(zone: Any, source: str) -> bool:
    if source == "fair_value_gap":
        return getattr(zone, "mitigation_status", None) in {"unmitigated", "partially_mitigated"}
    if source == "order_block":
        return getattr(zone, "mitigation_status", None) == "unmitigated"
    return (
        getattr(zone, "fvg_mitigation_status", None) in {"unmitigated", "partially_mitigated"}
        and getattr(zone, "ob_mitigation_status", None) == "unmitigated"
    )


def _select_target(trend: str, price: float, liquidity: Any, premium_discount: Any, candles=None) -> Optional[float]:
    side = "buy_side" if trend == "bullish" else "sell_side"
    levels = []
    for pool in getattr(liquidity, "pools", []):
        level = float(pool.level)
        if getattr(pool, "side", None) != side or not isfinite(level):
            continue
        if candles is not None:
            after = candles.iloc[getattr(pool, "last_index", 0) + 1:]
            if ((after.high >= level).any() if trend == "bullish" else (after.low <= level).any()):
                continue
        levels.append(level)
    levels = [level for level in levels if level > price] if trend == "bullish" else [level for level in levels if level < price]
    if levels:
        return min(levels) if trend == "bullish" else max(levels)
    fallback = getattr(premium_discount, "top", None) if trend == "bullish" else getattr(premium_discount, "bottom", None)
    if fallback is not None and ((trend == "bullish" and fallback > price) or (trend == "bearish" and fallback < price)):
        return float(fallback)
    return None


def _current_price(candles: pd.DataFrame) -> Optional[float]:
    if candles is None or "close" not in candles or candles.empty:
        return None
    close = pd.to_numeric(candles["close"], errors="coerce").iloc[-1]
    return None if pd.isna(close) or not isfinite(float(close)) or close <= 0 else float(close)


def _atr(candles: pd.DataFrame, current_price: float) -> float:
    required = {"high", "low", "close"}
    if candles is None or not required.issubset(candles.columns):
        return current_price * 0.001
    atr = average_true_range(candles, period=14)
    value = atr.iloc[-1]
    return current_price * 0.001 if pd.isna(value) or value <= 0 else float(value)


def _round(value: float) -> float:
    return round(value, 8)
