"""Turn structural analysis into an explicit, conservative trade plan.

The engines describe facts (swings, structure, zones); this module decides
whether those facts support a *conditional* setup.  It deliberately defaults
to ``wait``: the API must not turn incomplete or conflicting evidence into a
buy/sell instruction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from math import isfinite
from copy import deepcopy

import pandas as pd

from common.utils.indicators import average_true_range
from core.use_cases.market_analysis.setup_evidence import rank_entry_zones, staged_targets
from core.use_cases.market_analysis.market_read import build_market_read


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
    sweeps: Any = None,
    exit_policy: str = "single",
    vwap=None, volume_profile=None, divergence=None, cvd=None, tsmom=None,
    evidence_policy: str = "smc_v2",
    cost_policy: str = "none",
    minimum_stop_bps: float = 0.0,
    execution_policy: str = "retest",
    fee_bps_per_side: float = 10.0,
    slippage_bps_per_side: float = 2.0,
) -> Dict[str, Any]:
    """Return a JSON-safe execution plan; never invent a trade on weak data."""
    if execution_policy == "next_move":
        from core.use_cases.market_analysis.next_move_plan import build_next_move_plan
        return build_next_move_plan(
            candles, interval=interval, structure=structure, premium_discount=premium_discount,
            liquidity=liquidity, fvg=fvg, order_blocks=order_blocks, confluence=confluence,
            mtfa=mtfa, swings=swings, sweeps=sweeps, vwap=vwap, volume_profile=volume_profile,
            divergence=divergence, cvd=cvd, tsmom=tsmom, evidence_policy=evidence_policy,
            cost_policy=cost_policy, minimum_stop_bps=minimum_stop_bps,
            fee_bps_per_side=fee_bps_per_side, slippage_bps_per_side=slippage_bps_per_side,
        )
    if execution_policy != "retest":
        raise ValueError(f"Unsupported execution policy: {execution_policy}")
    if exit_policy not in {"single", "staged", "staged_no_be"}:
        raise ValueError(f"Unsupported exit policy: {exit_policy}")
    if evidence_policy not in {"smc_v2", "indicators_v1"}:
        raise ValueError(f"Unsupported evidence policy: {evidence_policy}")
    if cost_policy not in {"none", "minimum_stop_bps"}:
        raise ValueError(f"Unsupported cost policy: {cost_policy}")
    minimum_stop_bps = float(minimum_stop_bps)
    if not isfinite(minimum_stop_bps) or minimum_stop_bps < 0:
        raise ValueError("minimum_stop_bps must be a finite non-negative number")
    current_price = _current_price(candles)
    # Request boundary: disabled MTFA must not echo stale caller-owned context.
    mtfa = deepcopy(mtfa) if mtfa.get("enabled") is True else {"enabled": False, "context": "disabled"}
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
        "policy_version": "evidence-v2" if evidence_policy == "smc_v2" else "indicators-v1",
        "validation_status": "experimental_not_validated",
        "targets": [],
        "management": None,
        "chart_evidence": [],
        # An entry rejection must not discard the market analysis itself.
        "market_read": build_market_read(candles, interval, structure, swings) if current_price is not None else None,
    }

    if current_price is None:
        return _wait(base, "No valid closing price is available; no trade plan can be formed.")
    if trend not in {"bullish", "bearish"}:
        return _wait(base, "Market structure has no confirmed directional trend yet.")

    # A market scenario and permission to enter are different outputs. Keep a
    # local conditional scenario even if an entry/context gate returns early.
    # This does NOT populate entry_level/stop_loss/take_profit or change action.
    base["forecast_scenario"] = _structure_watch(candles, swings, trend, liquidity, "local")

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
    if scenario:
        base["forecast_scenario"] = deepcopy(scenario)
    if context == "pullback":
        return _wait(base, explanation)

    expected_zone = "discount" if trend == "bullish" else "premium"
    if not getattr(premium_discount, "range_available", False):
        return _wait(base, "The current dealing range is not confirmed, so price location cannot be assessed.")
    candidates = rank_entry_zones(candles, trend, order_blocks=order_blocks, fvg=fvg,
                                 confluence=confluence, structure=structure, sweeps=sweeps,
                                 swings=swings, mtfa=mtfa, vwap=vwap, volume_profile=volume_profile,
                                 divergence=divergence, cvd=cvd, tsmom=tsmom, evidence_policy=evidence_policy)
    if not candidates:
        return _wait(base, f"No fresh {trend} entry zone is available at or beyond current price.")
    entry = candidates[0]
    base["setup_quality"] = {key: entry[key] for key in ("score", "maximum", "threshold", "groups", "eligible")}
    base["setup_quality"]["interpretation"] = "Evidence checklist, not a win probability."
    base["setup_quality"]["indicator_evidence"] = entry["indicator_evidence"]
    base["setup_quality"]["evidence_policy"] = evidence_policy
    base["chart_evidence"] = entry["annotations"]
    base["selected_htf_poi"] = entry["poi"]
    if not entry["eligible"]:
        missing = ", ".join(key.replace("_", " ") for key, value in entry["groups"].items() if not value)
        return _wait(base, f"Setup evidence {entry['score']}/{entry['maximum']}; missing {missing}. No entry approved.")

    entry_level = entry["top"] if trend == "bullish" else entry["bottom"]
    range_top, range_bottom = getattr(premium_discount, "top", None), getattr(premium_discount, "bottom", None)
    if range_top is None or range_bottom is None or not range_bottom < range_top:
        return _wait(base, "No valid dealing-range bounds for the proposed entry.")
    equilibrium = (range_top + range_bottom) / 2
    # A break naturally moves the current close out of discount/premium.
    # Evaluate the pending RETRACEMENT entry's location, not the breakout close.
    in_location = range_bottom <= entry_level < equilibrium if trend == "bullish" else equilibrium < entry_level <= range_top
    if not in_location:
        return _wait(base, f"The proposed retracement entry is not inside the confirmed {expected_zone} range.")
    atr = _atr(candles, current_price)
    buffer = max(atr * 0.25, current_price * 0.0005)
    stop = entry["bottom"] - buffer if trend == "bullish" else entry["top"] + buffer
    targets = staged_targets(trend, entry_level, stop, liquidity, candles)
    alternative_targets = [dict(t) for t in targets]
    if exit_policy == "single" and targets:
        targets = [{**targets[0], "fraction": 1.0}]
    target = targets[0]["price"] if targets else None
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
    if cost_policy == "minimum_stop_bps":
        stop_bps = risk / entry_level * 10000
        passed = stop_bps + 1e-12 >= minimum_stop_bps
        comparison = "≥" if passed else "<"
        cost_fact = {
            "kind": "Mandatory cost gate", "timestamp": pd.Timestamp(candles.timestamp.iloc[-1]).isoformat(),
            "price": current_price, "label": f"Stop distance {stop_bps:.2f} bps {comparison} {minimum_stop_bps:.2f} bps",
            "group": "minimum_stop_distance", "status": "passed" if passed else "failed",
            "mandatory": True, "plot": False,
        }
        base["chart_evidence"].append(cost_fact)
        base["cost_economics"] = {"policy": cost_policy, "stop_distance_bps": round(stop_bps, 4),
                                  "minimum_stop_bps": float(minimum_stop_bps), "passed": passed,
                                  "note": "Research friction gate; not a profitability estimate."}
        if not passed:
            base["primary_scenario"] = None
            return _wait(base, f"Stop distance {stop_bps:.2f} bps is below the {minimum_stop_bps:.2f} bps cost floor. No entry approved.")

    trigger = (
        "Bullish structure break is observed. Within 12 candles, require a zone touch and close above entry; "
        "enter only on a subsequent retest at entry. Cancel at stop or after 12 candles."
        if trend == "bullish"
        else "Bearish structure break is observed. Within 12 candles, require a zone touch and close below entry; "
        "enter only on a subsequent retest at entry. Cancel at stop or after 12 candles."
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
        "reason": f"Evidence {entry['score']}/{entry['maximum']}: " + ", ".join(
            key.replace("_", " ") for key, value in entry["groups"].items() if value) + ". Retest entry is still pending.",
        "targets": targets,
        "research_exit_alternative": {
            "targets": alternative_targets,
            "status": "not_promoted: staged exits did not improve the tested new entries",
        },
        "management": {
            "mode": exit_policy, "entry_expiry_bars": 12,
            "stop_after_t1": _round(entry_level) if exit_policy == "staged" and len(targets) > 1 else None,
            "stop_change_effective": "next_bar_after_t1",
            "note": ("50% at T1, runner at T2; price breakeven is a loss after costs." if exit_policy == "staged"
                     else "50% at T1, runner at T2; keep original stop.") if len(targets)>1
                    else "Exit 100% at T1. Staged management is not enabled or no second target is available.",
        },
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
    base["forecast_scenario"] = deepcopy(base["primary_scenario"])
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
        return "incomplete", None, "Higher-timeframe evidence is incomplete; no MTFA entry is approved. Any displayed local scenario is conditional and lacks higher-timeframe validation."
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
                     if s.confirmed and s.type == pivot_type and 0 <= s.index < len(candles)
                     and s.index + getattr(swings, "window", 0) < len(candles)),
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
            "basis": "local_structure" if context == "local" else "mtfa_context_and_local_structure",
            "target_source": "unswept_liquidity" if target is not None else None,
            "trigger_source": "confirmed_swing",
            "trigger_timestamp": pd.Timestamp(candles.timestamp.iloc[pivot.index]).isoformat(),
            "interpretation": "Conditional directional scenario after activation; not a confirmed entry or calibrated prediction.",
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
