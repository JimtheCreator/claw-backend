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


def indicator_evidence(candles, direction, *, vwap=None, volume_profile=None,
                       divergence=None, cvd=None, tsmom=None, break_index=None):
    """As-of-signal checks, NOT claims about an unobserved future entry candle.

    None means unavailable and never earns a point. Separate checks are
    inspectable features, not a claim of statistical independence.
    """
    sign = 1 if direction == "bullish" else -1
    price = float(candles.close.iloc[-1])
    result = {}
    def put(name, value, reason):
        result[name] = {"passed": None if value is None else bool(value),
                        "available": value is not None, "reason": reason,
                        "signal_bar_open": pd.Timestamp(candles.timestamp.iloc[-1]).isoformat(),
                        "timing": "evaluated_after_signal_bar_close"}
    points = getattr(vwap, "points", [])
    latest = points[-1] if points else None
    valid = latest is not None and latest.index == len(candles)-1 and isfinite(latest.vwap)
    put("vwap_side", sign*(price-latest.vwap)>0 if valid else None,
        "Last closed price versus session VWAP; not a reclaim trigger.")
    poc = getattr(volume_profile, "poc_price", None)
    valid = getattr(volume_profile, "profile_available", False) and poc is not None and isfinite(poc)
    put("volume_profile_side", sign*(price-poc)>0 if valid else None,
        "Price versus window POC; OHLCV-distributed profile, not trade-level volume at price.")
    available = getattr(tsmom, "signal_available", False)
    momentum = getattr(tsmom, "combined_signal", None)
    put("tsmom_alignment", sign*momentum>0 if available and momentum is not None and isfinite(momentum) else None,
        "Configured same-timeframe return horizons; insufficient history is unavailable.")
    points = {p.index:p for p in getattr(cvd, "points", [])}
    delta = points.get(break_index)
    real = delta is not None and delta.delta_source == "taker_buy_volume" and isfinite(delta.delta)
    put("cvd_break_confirmation", sign*delta.delta>0 if real else None,
        "Actual taker-buy minus taker-sell volume on the structure-break candle; price proxy excluded.")
    # The live divergence engine uses two right-hand bars for confirmation.
    # Do not count events whose second pivot cannot yet have been confirmed.
    last = len(candles)-1
    ready = len(candles)>=35 and all(v is not None and isfinite(v) for v in
            (getattr(divergence,"latest_rsi",None),getattr(divergence,"latest_macd_histogram",None)))
    opposing = any(e.direction != direction and last-12 <= e.second_swing_index+2 <= last
                   for e in getattr(divergence,"events",[]))
    put("no_opposing_divergence", not opposing if ready else None,
        "No opposing RSI/MACD divergence confirmed in the last 12 bars; absence is not positive pressure.")
    return result


def rank_entry_zones(candles, direction, *, order_blocks, fvg, confluence, structure, sweeps, swings, mtfa,
                     vwap=None, volume_profile=None, divergence=None, cvd=None, tsmom=None,
                     evidence_policy="smc_v2"):
    """Rank actual candidates by independent evidence groups, then distance.

    HTF alignment is not awarded once per timeframe; OB+FVG is one group,
    never another point for every overlapping detector output.
    """
    if evidence_policy not in {"smc_v2", "indicators_v1"}:
        raise ValueError(f"Unsupported evidence policy: {evidence_policy}")
    price = float(candles.close.iloc[-1])
    bullish = direction == "bullish"
    atr = average_true_range(candles, period=14)
    events = [e for e in getattr(structure, "events", []) if e.direction == direction
              and 0 <= e.index < len(candles) and len(candles) - 1 - e.index <= 12]
    active_htf = mtfa.get("enabled") is True and mtfa.get("context") != "no_higher_timeframe"
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
            indicators = indicator_evidence(candles, direction, vwap=vwap, volume_profile=volume_profile,
                                            divergence=divergence, cvd=cvd, tsmom=tsmom,
                                            break_index=event.index if event else None)
            smc_eligible = mandatory and score >= threshold
            if evidence_policy == "indicators_v1":
                groups.update({k:v["passed"] for k,v in indicators.items()})
                positive = sum(v["passed"] is True for k,v in indicators.items() if k != "no_opposing_divergence")
                # Preserve every old SMC gate; absent divergence cannot substitute
                # for a positive standalone confirmation, especially with MTFA off.
                mandatory = smc_eligible and positive >= (1 if active_htf else 2)
                mandatory = mandatory and indicators["no_opposing_divergence"]["passed"] is not False
                threshold += 1 if active_htf else 2
                score = sum(value is True for value in groups.values())
            evidence = []
            if poi:
                evidence.append({"kind": "HTF POI", "timestamp": poi["touch_timestamp"],
                                 "price": poi["touch_price"],
                                 "label": f'{poi["timeframe"]} {poi["source"]} reaction', "zone": poi,
                                 "group": "htf_poi_reaction", "status": "passed", "plot": True})
            if sweep:
                evidence.append({"kind": "Sweep", "timestamp": pd.Timestamp(sweep.timestamp).isoformat(),
                                 "price": float(sweep.wick_price), "label": f'Sweep of {sweep.pool_level:g}',
                                 "group": "preceding_sweep", "status": "passed", "plot": True})
            if event:
                evidence.append({"kind": event.kind, "timestamp": pd.Timestamp(event.timestamp).isoformat(),
                                 "reference_timestamp": pd.Timestamp(candles.timestamp.iloc[getattr(event,"reference_swing_index",event.index)]).isoformat(),
                                 "price": float(event.level), "label": f'{event.kind} close {direction}',
                                 "group": "recent_structure_break", "status": "passed", "plot": True})
            # One ledger item for every group that actually influenced this
            # policy. Price-located facts above may be plotted; the rest are
            # explicit observed summaries so the renderer never implies a
            # synthetic chart coordinate for momentum/profile facts.
            covered = {item["group"] for item in evidence}
            required = {"recent_structure_break", "displacement"}
            if active_htf:
                required.add("htf_poi_reaction")
            if evidence_policy == "indicators_v1" and indicators["no_opposing_divergence"]["passed"] is False:
                required.add("no_opposing_divergence")
            labels = {
                "recent_structure_break": "Recent structure break",
                "displacement": "Directional displacement",
                "preceding_sweep": "Preceding liquidity sweep",
                "ob_fvg_overlap": "OB/FVG overlap",
                "htf_poi_reaction": "HTF POI reaction",
                "vwap_side": "Price on directional VWAP side",
                "volume_profile_side": "Price on directional POC side",
                "tsmom_alignment": "TSMOM aligned",
                "cvd_break_confirmation": "Real CVD confirms break",
                "no_opposing_divergence": "No opposing RSI/MACD divergence",
            }
            for name, value in groups.items():
                if name in covered:
                    next(item for item in evidence if item["group"] == name)["mandatory"] = name in required
                    continue
                state = "unavailable" if value is None else "passed" if value else "failed"
                evidence.append({"kind": "Decision fact", "timestamp": pd.Timestamp(candles.timestamp.iloc[-1]).isoformat(),
                                 "price": price, "label": f'{labels.get(name,name.replace("_"," "))}: {state.upper()}',
                                 "group": name, "status": state, "mandatory": name in required, "plot": False})
            if evidence_policy == "indicators_v1":
                positive = sum(indicators[k]["passed"] is True for k in
                               ("vwap_side","volume_profile_side","tsmom_alignment","cvd_break_confirmation"))
                needed = 1 if active_htf else 2
                evidence.append({"kind": "Mandatory gate", "timestamp": pd.Timestamp(candles.timestamp.iloc[-1]).isoformat(),
                                 "price": price, "label": f'Standalone confirmations: {positive}/{needed}',
                                 "group": "standalone_confirmation_count",
                                 "status": "passed" if positive >= needed else "failed", "mandatory": True, "plot": False})
            ranked.append({"bottom": bottom, "top": top, "source": source,
                           "formed_index": formed, "score": score, "maximum": len(groups),
                           "groups": groups, "eligible": bool(mandatory and score >= threshold),
                           "threshold": threshold, "poi": poi, "annotations": evidence,
                           "indicator_evidence": indicators})
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
