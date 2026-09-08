"""One closed-snapshot entry, one nearby exit. No forecast of an entry approach.

Reuses the existing evidence policy unchanged. The entry event must already
have occurred on the latest closed candle. This research execution policy has
no calibrated probability and does not place an order.
"""
from copy import deepcopy
from math import isfinite

import pandas as pd

from core.use_cases.market_analysis.setup_evidence import rank_entry_zones
from core.use_cases.market_analysis.trade_plan import _atr, _current_price, _market_context, _round


def _number(value):
    try:
        value = float(value)
        return value if isfinite(value) and value > 0 else None
    except (ValueError, TypeError):
        return None


def decision_levels(candles, direction, *, liquidity, order_blocks, fvg, swings, mtfa):
    """Closest untouched local/active HTF obstacle, at its near edge.

    Ignore future/unconfirmed pivots and zones invalidated before the snapshot.
    A farther target may never replace a closer one to improve reward/risk.
    """
    current = float(candles.close.iloc[-1]); last = len(candles) - 1
    bullish = direction == "bullish"
    result = []

    def add(value, source, index=None, timeframe=None):
        value = _number(value)
        if value is None or not (value > current if bullish else value < current):
            return
        item = {"price": value, "source": source, "role": "exit", "observed": True}
        if index is not None:
            item["timestamp"] = pd.Timestamp(candles.timestamp.iloc[index]).isoformat()
        if timeframe is not None:
            item["timeframe"] = timeframe
        result.append(item)

    for pool in getattr(liquidity, "pools", []):
        index = getattr(pool, "last_index", -1)
        level = _number(getattr(pool, "level", None))
        if not 0 <= index <= last or level is None or pool.side != ("buy_side" if bullish else "sell_side"):
            continue
        after = candles.iloc[index+1:]
        if not ((after.high >= level).any() if bullish else (after.low <= level).any()):
            add(level, "local_liquidity", index)
    for pivot in getattr(swings, "swings", []):
        index = pivot.index
        confirmed_at = getattr(pivot, "confirmed_index", index + getattr(swings, "window", 5))
        if not pivot.confirmed or not 0 <= index <= confirmed_at <= last or pivot.type != ("high" if bullish else "low"):
            continue
        level = _number(pivot.price)
        if level is None:
            continue
        after = candles.iloc[index+1:]
        if not ((after.high >= level).any() if bullish else (after.low <= level).any()):
            add(level, "local_swing", index)
    for source, zones in (("local_order_block", getattr(order_blocks, "zones", [])),
                          ("local_fvg", getattr(fvg, "zones", []))):
        for zone in zones:
            formed = getattr(zone, "breakout_index", -1) if source == "local_order_block" else getattr(zone, "formed_index", -1)
            if not 0 <= formed <= last or zone.type != ("bearish" if bullish else "bullish"):
                continue
            bottom, top = _number(zone.bottom), _number(zone.top)
            if bottom is None or top is None or bottom >= top:
                continue
            # Use untouched near edges; partially consumed zones are not fresh obstacles.
            if getattr(zone, "mitigation_status", None) != "unmitigated":
                continue
            after = candles.iloc[formed+1:]
            if ((after.high >= bottom).any() if bullish else (after.low <= top).any()):
                continue
            add(bottom if bullish else top, source, formed)
    if mtfa.get("enabled") is True and mtfa.get("context") != "no_higher_timeframe":
        times = pd.to_datetime(candles.timestamp, utc=True)
        for zone in mtfa.get("htf_zones", []):
            if zone.get("direction") != ("bearish" if bullish else "bullish"):
                continue
            bottom, top = _number(zone.get("bottom")), _number(zone.get("top"))
            if bottom is None or top is None or bottom >= top:
                continue
            available = pd.to_datetime(zone.get("available_at"), utc=True, errors="coerce")
            if pd.isna(available) or available > times.iloc[-1]:
                continue
            after = candles.loc[times >= available]
            if ((after.high >= bottom).any() if bullish else (after.low <= top).any()):
                continue
            add(bottom if bullish else top, "htf_" + zone.get("source", "zone"), timeframe=zone.get("timeframe"))
    return sorted(result, key=lambda item: (abs(item["price"] - current), item["source"]))


def build_next_move_plan(candles, *, interval, structure, premium_discount, liquidity,
                         order_blocks, fvg, confluence, mtfa, swings=None, sweeps=None,
                         vwap=None, volume_profile=None, divergence=None, cvd=None, tsmom=None,
                         evidence_policy="smc_v2", cost_policy="none", minimum_stop_bps=0,
                         fee_bps_per_side=10, slippage_bps_per_side=2):
    if evidence_policy not in {"smc_v2", "indicators_v1"}:
        raise ValueError("Unsupported evidence policy")
    if cost_policy not in {"none", "minimum_stop_bps"}:
        raise ValueError("Unsupported cost policy")
    costs = [float(v) for v in (fee_bps_per_side, slippage_bps_per_side, minimum_stop_bps)]
    if any(not isfinite(v) or v < 0 for v in costs):
        raise ValueError("Cost assumptions must be finite and non-negative")
    fee, slip, floor = costs
    if fee + slip >= 10000:
        raise ValueError("Per-side friction must be less than 10000 bps")
    context_data = deepcopy(mtfa) if mtfa.get("enabled") is True else {"enabled": False, "context": "disabled"}
    current = _current_price(candles)
    trend = getattr(structure, "trend", None)
    plan = dict(interval=interval, trend_direction=trend or "undetermined", current_price=current,
                action="wait", entry_level=None, entry_zone=None, stop_loss=None, take_profit=None,
                risk_per_unit=None, risk_reward=None, wait_for_confirmation=True,
                confirmation_required="Wait for a closed-candle entry trigger; reanalyse when it occurs.",
                reason=None, evidence={"mtfa": context_data}, primary_scenario=None,
                policy_version="next-move-v1", execution_policy="next_move", evidence_policy=evidence_policy,
                validation_status="experimental_not_validated", targets=[], management=None,
                chart_evidence=[], decision_level=None, blockers=[])
    if current is None:
        plan["reason"] = "No valid closed price; no trade can be evaluated."
        return plan
    last = len(candles)-1; bar = candles.iloc[-1]
    plan["signal_bar_open"] = pd.Timestamp(bar.timestamp).isoformat()

    def fact(group, label, passed, mandatory=True, **extra):
        plan["chart_evidence"].append(dict(group=group, kind="Decision fact", label=label,
            status="passed" if passed else "failed", mandatory=mandatory, plot=False,
            timestamp=plan["signal_bar_open"], price=current, **extra))
        if mandatory and not passed:
            plan["blockers"].append(label)

    events = [e for e in getattr(structure, "events", []) if e.direction in {"bullish", "bearish"}
              and 0 <= e.index <= last and last-e.index <= 12 and _number(e.level) is not None]
    event = max(events, key=lambda e:e.index, default=None)
    direction = event.direction if event else trend
    if direction not in {"bullish", "bearish"}:
        plan["reason"] = "WAIT: no observed directional structure or supported next move."
        return plan
    bullish = direction == "bullish"
    sign = 1 if bullish else -1
    plan["move_direction"] = direction
    levels = decision_levels(candles, direction, liquidity=liquidity, order_blocks=order_blocks,
                             fvg=fvg, swings=swings, mtfa=context_data)
    plan["decision_level"] = levels[0] if levels else None
    context, bias, explanation = _market_context(context_data, direction)
    plan.update(market_context=context, context_summary=explanation)

    # Compute local facts even when higher context blocks execution, so an ON
    # chart still explains the same observed BOS/CHoCH as its OFF counterpart.
    candidates = rank_entry_zones(candles, direction, order_blocks=order_blocks, fvg=fvg,
        confluence=confluence, structure=structure, sweeps=sweeps, swings=swings, mtfa=context_data,
        vwap=vwap, volume_profile=volume_profile, divergence=divergence, cvd=cvd, tsmom=tsmom,
        evidence_policy=evidence_policy)
    candidate = candidates[0] if candidates else None
    if candidate:
        plan["chart_evidence"] = deepcopy(candidate["annotations"])
        plan["setup_quality"] = {k:candidate[k] for k in ("score", "maximum", "threshold", "groups", "eligible")}
        plan["setup_quality"]["indicator_evidence"] = candidate["indicator_evidence"]
        if candidate["poi"] is not None:
            plan["selected_htf_poi"] = candidate["poi"]
    elif event:
        plan["chart_evidence"].append(dict(kind=event.kind, label=f"Observed {event.kind} {direction}",
            group="observed_structure", status="passed", plot=True,
            timestamp=pd.Timestamp(candles.timestamp.iloc[event.index]).isoformat(), price=float(event.level)))
    fact("entry_evidence", "Linked entry evidence meets the existing policy" if candidate and candidate["eligible"]
         else "Entry evidence is incomplete: no qualified local reaction/structure setup", bool(candidate and candidate["eligible"]))
    linked = bool(event and candidate and any(
        item.get("group") == "recent_structure_break" and
        pd.Timestamp(item["timestamp"]) == pd.Timestamp(candles.timestamp.iloc[event.index])
        for item in candidate["annotations"]))
    fact("same_trigger_leg", "Entry evidence belongs to the current structure event" if linked
         else "Zone evidence is not linked to the current structure event", linked)
    if context_data.get("enabled") is True and context_data.get("context") != "no_higher_timeframe":
        fact("htf_execution_context", "Higher-timeframe context supports this move" if context in {"local", "aligned"}
             else explanation, context in {"local", "aligned"})
    # A latest-bar displacement break OR latest-bar rejection at the previously
    # broken level is an observed trigger. A drawn future retracement is not.
    held = bool(event and ((candles.close.iloc[event.index:] - event.level)*sign > 0).all())
    rejection = bool(event and event.index < last and bar.low <= event.level <= bar.high
                     and (bar.close-bar.open)*sign > 0 and (bar.close-event.level)*sign > 0)
    trigger = held and bool(event.index == last or rejection)
    fact("entry_trigger_now", "Latest closed candle confirms the entry break/retest" if trigger
         else "No fresh entry trigger on the latest closed candle; do not chase the level", trigger)
    if levels:
        fact("first_obstacle", f"Exit at nearest {levels[0]['source'].replace('_',' ')}: {levels[0]['price']:g}", True)
    else:
        fact("first_obstacle", "No untouched nearby exit level is available", False)

    if candidate and trigger and levels:
        target = levels[0]["price"]
        # Invalidation is outside both the originating POI and the signal bar.
        buffer = max(_atr(candles, current)*.25, current*.0005)
        stop = min(candidate["bottom"], float(bar.low))-buffer if bullish else max(candidate["top"],float(bar.high))+buffer
        risk, reward = abs(current-stop), (target-current)*sign
        friction = (fee+slip)/10000
        target_cost = (current+target)*friction
        stop_cost = (current+stop)*friction
        gross_rr = reward/risk if risk>0 else 0
        net_rr = (reward-target_cost)/(risk+stop_cost) if risk+stop_cost>0 else 0
        plan["cost_economics"] = dict(fee_bps_per_side=fee, slippage_bps_per_side=slip,
            stop_distance_bps=risk/current*10000, gross_reward_risk=gross_rr,
            net_reward_risk=net_rr, target_cost_per_unit=target_cost,
            note="Configured fee/slippage estimate, not expected return or a live quote.")
        bottom, top = _number(getattr(premium_discount,"bottom",None)), _number(getattr(premium_discount,"top",None))
        valid_range = bool(getattr(premium_discount,"range_available",False) and bottom and top and bottom<top)
        in_location = valid_range and (bottom<=current<(bottom+top)/2 if bullish else (bottom+top)/2<current<=top)
        fact("entry_location", "Current entry is in discount" if bullish and in_location else
             "Current entry is in premium" if in_location else "Current entry is outside the required dealing-range location", bool(in_location))
        fact("net_reward_risk", f"Reward/risk after assumed costs: {net_rr:.2f} (minimum 1.50)", net_rr>=1.5)
        if cost_policy == "minimum_stop_bps":
            fact("minimum_stop_distance", f"Stop distance {risk/current*10000:.2f} bps; minimum {floor:.2f}", risk/current*10000>=floor)
        if stop<=0:
            fact("valid_stop", "No positive structural stop is available", False)
        if not plan["blockers"]:
            action = "long" if bullish else "short"
            plan.update(action=action, entry_level=_round(current), stop_loss=_round(stop),
                take_profit=_round(target), risk_per_unit=_round(risk), risk_reward=round(gross_rr,2),
                wait_for_confirmation=False,
                confirmation_required="Closed-candle trigger observed. Entry reference is the last close; recheck live price before entry.",
                reason=f"{'BUY' if bullish else 'SELL'} toward {target:g}: confirmed {event.kind} / reaction, linked evidence, and {net_rr:.2f} reward/risk after assumed costs.",
                targets=[dict(price=_round(target),fraction=1.0,source=levels[0]["source"])],
                management=dict(mode="single",note="Exit 100% at the next level; reassess there. No automatic order execution."),
                primary_scenario=dict(kind="next_move", direction=direction, setup=True,
                    title=f"{'BUY' if bullish else 'SELL'} to the next level",trigger=_round(current),
                    target=_round(target),invalidation=_round(stop)))
            return plan
    plan["reason"] = "WAIT: " + "; ".join(plan["blockers"][:3]) + "."
    return plan
