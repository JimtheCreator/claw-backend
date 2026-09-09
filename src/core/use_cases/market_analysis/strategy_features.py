"""Shared, prefix-causal inputs for independent strategies, live and replay.

Detector end-of-frame mitigation labels are NEVER used historically. Formation
events are replayed forward, and only confirmed pivots become active.
"""
import numpy as np
import pandas as pd

from core.engines.fvg_engine import FVGEngine
from core.engines.order_block_engine import OrderBlockEngine
from core.engines.market_structure_engine import MarketStructureEngine
from core.engines.swing_structure_engine import SwingStructureEngine
from core.engines.regime_engine import RegimeEngine
from core.engines.tsmom_engine import TSMOMEngine


def interval_offset(interval):
    if interval == '1M':
        return pd.DateOffset(months=1)
    return pd.Timedelta(int(interval[:-1]), unit={'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W'}[interval[-1]])


def validate_frame(frame, interval):
    f = frame.copy().reset_index(drop=True)
    if f.empty:
        raise ValueError('Empty strategy history')
    f['timestamp'] = pd.to_datetime(f.timestamp, utc=True)
    numbers = f[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers[:, :4] <= 0).any() or (numbers[:, 4] < 0).any():
        raise ValueError('Invalid OHLCV')
    if (f.high < f[['open', 'low', 'close']].max(axis=1)).any() or (f.low > f[['open', 'high', 'close']].min(axis=1)).any():
        raise ValueError('Invalid OHLC ordering')
    expected = f.timestamp.iloc[:-1]+interval_offset(interval)
    if not np.array_equal(expected.to_numpy(), f.timestamp.iloc[1:].to_numpy()):
        raise ValueError('Strategy history must be sorted, unique and contiguous')
    return f


def momentum_features(frame, interval):
    horizons = TSMOMEngine._interval_lookback_defaults.get(interval)
    if horizons is None:
        raise ValueError(f'No declared TSMOM horizons for {interval}')
    f = pd.DataFrame(index=frame.index)
    returns = pd.concat([frame.close/frame.close.shift(h)-1 for h in horizons], axis=1)
    f['momentum_complete'] = returns.notna().all(axis=1)
    signs = np.sign(returns)
    f['momentum_sign'] = signs.mean(axis=1).where(f.momentum_complete, 0)
    f['momentum_sign'] = f.momentum_sign.where(f.momentum_sign.abs() == 1, 0).astype(int)
    upper, lower = frame.high.shift().rolling(20).max(), frame.low.shift().rolling(20).min()
    long = (frame.close > upper) & (frame.close.shift() <= upper.shift())
    short = (frame.close < lower) & (frame.close.shift() >= lower.shift())
    f['channel_trigger'] = np.select([long, short], [1, -1], default=0)
    typical, session = (frame.high+frame.low+frame.close)/3, frame.timestamp.dt.floor('D')
    volume = frame.volume.groupby(session).cumsum().replace(0, np.nan)
    f['vwap'] = (typical*frame.volume).groupby(session).cumsum()/volume
    f['session_bars'] = frame.groupby(session).cumcount()+1
    # Reject a truncated first UTC session instead of treating it as a full one.
    starts = frame.timestamp.groupby(session).transform('first')
    f['session_complete'] = starts == session
    taker = pd.to_numeric(frame.get('taker_buy_volume', pd.Series(np.nan, index=frame.index)), errors='coerce')
    genuine = np.isfinite(taker) & taker.between(0, frame.volume)
    f['delta'] = (2*taker-frame.volume).where(genuine)
    f['delta3'] = f.delta.rolling(3, min_periods=3).sum()
    f['real_flow'] = f.delta3.notna()
    return f


def zone_events(frame, interval, structure=None):
    if structure is None:
        swings = SwingStructureEngine(interval).detect_swings(frame)
        structure = MarketStructureEngine(interval).detect_structure(frame, swings)
    ob = OrderBlockEngine(interval).detect_order_blocks(frame, structure)
    fvg = FVGEngine(interval).detect_fvgs(frame)
    result = []
    for source, zones in (('order_block', ob.zones), ('fvg', fvg.zones)):
        for zone in zones:
            formed = zone.breakout_index if source == 'order_block' else zone.formed_index
            if zone.top <= zone.bottom:
                continue
            result.append(dict(source=source, sign=1 if zone.type == 'bullish' else -1,
                               bottom=float(zone.bottom), top=float(zone.top), formed=formed,
                               available_at=frame.timestamp.iloc[formed]+interval_offset(interval)))
    return result


def local_features(frame, interval):
    f = validate_frame(frame, interval)
    regime = RegimeEngine().features(f)
    momentum = momentum_features(f, interval)
    f = pd.concat([f, regime, momentum], axis=1)
    f['available_at'] = f.timestamp+interval_offset(interval)
    swings = SwingStructureEngine(interval).detect_swings(f)
    structure = MarketStructureEngine(interval).detect_structure(f, swings)
    confirmations, formations = {}, {}
    for s in swings.swings:
        if s.confirmed:
            confirmations.setdefault(s.index+swings.window, []).append(s)
    for z in zone_events(f, interval, structure):
        formations.setdefault(z['formed'], []).append(z)
    breaks = {e.index: e for e in structure.events}
    out, highs, lows, active, recent_sweeps, trend = [], [], [], [], [], 0
    for i, bar in enumerate(f.itertuples(index=False)):
        for pivot in confirmations.get(i, []):
            (highs if pivot.type == 'high' else lows).append(pivot)
        # Pivots admitted on this candle cannot retroactively explain its wick.
        high_sweep = any(p.index+swings.window < i and bar.high > p.price >= bar.close for p in highs)
        low_sweep = any(p.index+swings.window < i and bar.low < p.price <= bar.close for p in lows)
        highs = [p for p in highs if bar.close <= p.price]
        lows = [p for p in lows if bar.close >= p.price]
        event = breaks.get(i)
        sign = 1 if event and event.direction == 'bullish' else -1 if event else 0
        if sign:
            trend = sign
        prior_atr = f.atr.iloc[i-1] if i else np.nan
        body = sign*(bar.close-bar.open)
        displacement = bool(sign and np.isfinite(prior_atr) and prior_atr > 0 and
                            body >= .8*prior_atr and body >= .6*(bar.high-bar.low))
        active = [z for z in active if i-z['formed'] <= 64 and
                  not (bar.close < z['bottom'] if z['sign'] == 1 else bar.close > z['top']) and
                  not (z['source'] == 'fvg' and (bar.low <= z['bottom'] if z['sign'] == 1 else bar.high >= z['top']))]
        active.extend(formations.get(i, []))
        obs = [z for z in active if z['source'] == 'order_block' and z['sign'] == sign and abs(i-z['formed']) <= 3]
        gaps = [z for z in active if z['source'] == 'fvg' and z['sign'] == sign and abs(i-z['formed']) <= 3]
        overlap = any(max(a['bottom'], b['bottom']) <= min(a['top'], b['top']) for a in obs for b in gaps)
        sweep = any(s == sign and 0 < i-j <= 10 for j, s in recent_sweeps)
        recent_sweeps = [(j, s) for j, s in recent_sweeps if i-j <= 10]
        if high_sweep:
            recent_sweeps.append((i, -1))
        if low_sweep:
            recent_sweeps.append((i, 1))
        out.append(dict(structural_sign=trend, break_sign=sign, break_kind=event.kind if event else None,
                        break_level=event.level if event else None, displacement=displacement,
                        preceding_sweep=bool(sweep), ob_fvg_overlap=bool(overlap),
                        support=max((p.price for p in lows), default=np.nan),
                        resistance=min((p.price for p in highs), default=np.nan),
                        stop_support=lows[-1].price if lows else np.nan,
                        stop_resistance=highs[-1].price if highs else np.nan))
    return pd.concat([f, pd.DataFrame(out)], axis=1)


def top_down_context(local, middle, anchor, *, middle_interval, anchor_interval):
    """One anchor POI + a subsequent middle reaction, never TF-majority voting.

    Accepts complete historical frames, but returns only facts available by each
    local bar close. Zone terminal statuses are reconstructed, not backfilled.
    """
    middle = validate_frame(middle, middle_interval)
    anchor = validate_frame(anchor, anchor_interval)
    swings = SwingStructureEngine(anchor_interval).detect_swings(anchor)
    structure = MarketStructureEngine(anchor_interval).detect_structure(anchor, swings)
    zones = sorted(zone_events(anchor, anchor_interval, structure), key=lambda z: z['available_at'])
    aclose = anchor.timestamp+interval_offset(anchor_interval)
    events = sorted([(aclose.iloc[e.index], 1 if e.direction == 'bullish' else -1) for e in structure.events])
    active, records, ptr, eptr, bias = [], [], 0, 0, 0
    for j, bar in enumerate(middle.itertuples(index=False)):
        now = bar.timestamp+interval_offset(middle_interval)
        while ptr < len(zones) and zones[ptr]['available_at'] <= bar.timestamp:
            active.append(dict(zones[ptr]))
            ptr += 1
        while eptr < len(events) and events[eptr][0] <= now:
            bias = events[eptr][1]
            eptr += 1
        alive, reactions = [], []
        for z in active:
            age_end = z['available_at']+180*interval_offset(anchor_interval)
            invalid = bar.close < z['bottom'] if z['sign'] == 1 else bar.close > z['top']
            if now > age_end or invalid:
                continue
            alive.append(z)
            touch = bar.low <= z['top'] and bar.high >= z['bottom']
            reject = bar.close > z['top'] if z['sign'] == 1 else bar.close < z['bottom']
            if touch and reject and bias == z['sign']:
                reactions.append(dict(z, timeframe=anchor_interval, middle_timeframe=middle_interval,
                                      reaction_at=now.isoformat(), reaction_index=j))
        active = alive
        records.append(dict(available_at=now, index=j, reactions=reactions, bias=bias))
    # Retain only reactions within three *completed* middle bars and still-valid
    # zones; invalidate immediately on an intervening local close through it.
    result, recent, ptr, current_j, current_bias = [], [], 0, -1, 0
    for bar in local.itertuples(index=False):
        now = bar.available_at
        while ptr < len(records) and records[ptr]['available_at'] <= now:
            current_j = records[ptr]['index']
            current_bias = records[ptr]['bias']
            recent.extend(records[ptr]['reactions'])
            ptr += 1
        recent = [z for z in recent if z['sign'] == current_bias and current_j-z['reaction_index'] < 3 and
                  not (bar.close < z['bottom'] if z['sign'] == 1 else bar.close > z['top'])]
        ready = [z for z in recent if pd.Timestamp(z['reaction_at']) <= bar.timestamp]
        ready.sort(key=lambda z: (-pd.Timestamp(z['reaction_at']).value, z['top']-z['bottom'], z['source']))
        result.append([dict(z, available_at=z['available_at'].isoformat()) for z in ready])
    return result
