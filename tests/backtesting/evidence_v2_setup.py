"""Deterministic evidence, not probabilities. All indices refer to closed snapshots.

Policy v1 is deliberately predeclared: recent break <=12 bars; directional
body >=0.8 previous ATR and >=60% candle range; sweep <=10 bars before break.
These are testable research rules, NOT empirically established optimal values.
"""
from math import isfinite

import pandas as pd

from common.utils.indicators import average_true_range


def higher_timeframe_zones(candles, interval, order_blocks, fvg):
    """Carry POIs, including touched-but-not-close-invalidated OBs.

    OB origin time is NOT its availability time: it is known only when the
    breakout candle closes. FVG availability likewise uses its third candle.
    """
    delta = (pd.DateOffset(months=int(interval[:-1])) if interval.endswith("M") else
             pd.Timedelta(int(interval[:-1]), unit={"m":"min","h":"h","d":"D","w":"W"}[interval[-1]]))
    result = []
    for source, zones in (("order_block", order_blocks.zones), ("fvg", fvg.zones)):
        for z in zones:
            j = z.breakout_index if source == "order_block" else z.formed_index
            if not 0 <= j < len(candles) or not 0 < z.bottom < z.top:
                continue
            after = candles.iloc[j + 1:]
            invalid = (after.close < z.bottom).any() if z.type == "bullish" else (after.close > z.top).any()
            if invalid or (source == "fvg" and z.mitigation_status == "fully_mitigated"):
                continue
            result.append({"timeframe": interval, "source": source, "direction": z.type,
                           "bottom": float(z.bottom), "top": float(z.top),
                           "available_at": (pd.Timestamp(candles.timestamp.iloc[j]) + delta).isoformat(),
                           "status": z.mitigation_status})
    return result


def rank_entry_zones(candles, direction, *, order_blocks, fvg, confluence, structure, sweeps, swings, mtfa):
    """Rank actual candidates by independent evidence groups, then distance.

    HTF alignment is not awarded once per timeframe; OB+FVG is one group,
    never another point for every overlapping detector output.
    """
    price = float(candles.close.iloc[-1])
    bullish = direction == "bullish"
    atr = average_true_range(candles, period=14)
    events = [e for e in getattr(structure, "events", []) if e.direction == direction
              and 0 <= e.index < len(candles) and len(candles) - 1 - e.index <= 12]
    active_htf = mtfa.get("enabled") and mtfa.get("context") != "no_higher_timeframe"
    ranked, seen = [], set()
    for source, zones in (("confluence", getattr(confluence, "zones", [])),
                          ("order_block", getattr(order_blocks, "zones", [])),
                          ("fvg", getattr(fvg, "zones", []))):
        for z in zones:
            bottom, top = float(z.bottom), float(z.top)
            if z.type != direction or not all(isfinite(v) and v > 0 for v in (bottom, top)) or bottom >= top:
                continue
            fresh = (z.fvg_mitigation_status in {"unmitigated", "partially_mitigated"}
                     and z.ob_mitigation_status == "unmitigated") if source == "confluence" else (
                         z.mitigation_status in ({"unmitigated", "partially_mitigated"} if source == "fvg" else {"unmitigated"}))
            if not fresh or not (bottom <= price if bullish else top >= price):
                continue
            formed = z.ob_breakout_index if source == "confluence" else z.breakout_index if source == "order_block" else z.formed_index
            origin = z.ob_candle_index if source == "confluence" else z.candle_index if source == "order_block" else z.start_index
            if not 0 <= origin <= formed < len(candles) or (bottom, top) in seen:
                continue
            seen.add((bottom, top))
            # Link the break to THIS originating leg, not an unrelated event.
            related = [e for e in events if origin <= e.index and abs(e.index - formed) <= 3]
            event = max(related, key=lambda e: e.index, default=None)
            displacement = False
            if event is not None and event.index > 0:
                bar = candles.iloc[event.index]
                body = (bar.close - bar.open) * (1 if bullish else -1)
                prior_atr = float(atr.iloc[event.index - 1])
                displacement = bool(isfinite(prior_atr) and prior_atr > 0 and body >= 0.8 * prior_atr
                                    and body >= 0.6 * (bar.high - bar.low))
            sweep = None
            if event is not None:
                matches = [s for s in getattr(sweeps, "events", [])
                           if s.pool_side == ("sell_side" if bullish else "buy_side")
                           and max(0, origin - 10) <= s.index < event.index
                           and event.index - s.index <= 10
                           and s.pool_last_index + getattr(swings, "window", 5) < s.index]
                sweep = max(matches, key=lambda s: s.index, default=None)
            pois = []
            if active_htf:
                for poi in mtfa.get("htf_zones", []):
                    if poi["direction"] != direction or max(bottom, poi["bottom"]) > min(top, poi["top"]):
                        continue
                    # A zone's HTF must support the governing bias, not oppose it.
                    if mtfa.get("htf_trends", {}).get(poi["timeframe"]) != direction:
                        continue
                    available = pd.Timestamp(poi["available_at"])
                    if available.tzinfo is None:
                        available = available.tz_localize("UTC")
                    times = pd.to_datetime(candles.timestamp, utc=True)
                    before_break = candles.loc[(times >= available) & (candles.index <= (event.index if event else len(candles)-1))]
                    touches = before_break[(before_break.low <= poi["top"]) & (before_break.high >= poi["bottom"])]
                    # HTF closed-candle invalidation can lag the LTF: check that too.
                    since = candles.loc[times >= available]
                    broken = (since.close < poi["bottom"]).any() if bullish else (since.close > poi["top"]).any()
                    if not touches.empty and not broken:
                        pois.append({**poi, "touch_timestamp": pd.Timestamp(touches.timestamp.iloc[-1]).isoformat(),
                                     "touch_price": float(touches.low.iloc[-1] if bullish else touches.high.iloc[-1])})
            poi = min(pois, key=lambda p: (p["top"]-p["bottom"], p["timeframe"], p["available_at"]), default=None)
            groups = {"recent_structure_break": event is not None, "displacement": displacement,
                      "preceding_sweep": sweep is not None, "ob_fvg_overlap": source == "confluence"}
            if active_htf:
                groups["htf_poi_reaction"] = poi is not None
            score = sum(groups.values())
            mandatory = event is not None and displacement and (not active_htf or poi is not None)
            threshold = 4 if active_htf else 3
            evidence = []
            if poi:
                evidence.append({"kind": "HTF POI", "timestamp": poi["touch_timestamp"],
                                 "price": poi["touch_price"],
                                 "label": f'{poi["timeframe"]} {poi["source"]} reaction', "zone": poi})
            if sweep:
                evidence.append({"kind": "Sweep", "timestamp": pd.Timestamp(sweep.timestamp).isoformat(),
                                 "price": float(sweep.wick_price), "label": f'Sweep of {sweep.pool_level:g}'})
            if event:
                evidence.append({"kind": event.kind, "timestamp": pd.Timestamp(event.timestamp).isoformat(),
                                 "reference_timestamp": pd.Timestamp(candles.timestamp.iloc[getattr(event,"reference_swing_index",event.index)]).isoformat(),
                                 "price": float(event.level), "label": f'{event.kind} close {direction}' + (" + displacement" if displacement else "")})
            ranked.append({"bottom": bottom, "top": top, "source": source,
                           "formed_index": formed, "score": score, "maximum": len(groups),
                           "groups": groups, "eligible": bool(mandatory and score >= threshold),
                           "threshold": threshold, "poi": poi, "annotations": evidence})
    return sorted(ranked, key=lambda z: (-z["eligible"], -z["score"],
                  abs(price-(z["top"] if bullish else z["bottom"])), -z["formed_index"], z["source"]))


def staged_targets(direction, entry, stop, liquidity, candles):
    """Nearest untouched opposing liquidity; no invented second target."""
    bullish = direction == "bullish"
    levels = set()
    for pool in getattr(liquidity, "pools", []):
        value = float(pool.level)
        if pool.side != ("buy_side" if bullish else "sell_side") or not isfinite(value):
            continue
        after = candles.iloc[pool.last_index + 1:]
        if (after.high >= value).any() if bullish else (after.low <= value).any():
            continue
        if value > max(entry, float(candles.close.iloc[-1])) if bullish else value < min(entry, float(candles.close.iloc[-1])):
            levels.add(value)
    ordered = sorted(levels, reverse=not bullish)[:2]
    return [{"label": f"T{i+1}", "price": p, "fraction": 0.5 if len(ordered)==2 else 1.0,
             "r": abs(p-entry)/abs(entry-stop), "source": "unswept_liquidity"} for i, p in enumerate(ordered)]
