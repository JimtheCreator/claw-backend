"""Offline candidates. Not imported by live task/planner; no fitted parameters.

Reuse production structural primitives, with causal availability and prefix tests.
All input candles must be closed, sorted and contiguous UTC hourly observations.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from core.engines.swing_structure_engine import SwingStructureEngine
from core.engines.market_structure_engine import MarketStructureEngine
from tests.backtesting.run_trade_plans import resample_closed


@dataclass(frozen=True)
class Signal:
    index: int
    model: str
    sign: int
    stop: float
    target: float
    evidence: dict


def structural_features(frame, interval):
    swings = SwingStructureEngine(interval).detect_swings(frame)
    structure = MarketStructureEngine(interval).detect_structure(frame, swings)
    f = frame.copy().reset_index(drop=True)
    previous = f.close.shift()
    tr = pd.concat([f.high-f.low, (f.high-previous).abs(), (f.low-previous).abs()], axis=1).max(axis=1)
    f['atr'] = tr.rolling(14, min_periods=14).mean()
    for key in ('trend', 'break_sign'):
        f[key] = 0
    for key in ('support', 'resistance', 'broken_support', 'broken_resistance',
                'high_a', 'high_b', 'high_ai', 'high_bi', 'low_a', 'low_b', 'low_ai', 'low_bi'):
        f[key] = np.nan
    confirmed = {}
    for s in swings.swings:
        if s.confirmed:
            confirmed.setdefault(s.index+swings.window, []).append(s)
    events = {e.index: e for e in structure.events}
    highs, lows, state = [], [], {}
    for i in range(len(f)):
        for s in confirmed.get(i, []):
            (highs if s.type == 'high' else lows).append(s)
        for points, name, level in ((highs, 'high', 'resistance'), (lows, 'low', 'support')):
            if points:
                state[level] = points[-1].price
            if len(points) >= 2:
                for suffix, point in zip(('a', 'b'), points[-2:]):
                    state[f'{name}_{suffix}'] = point.price
                    state[f'{name}_{suffix}i'] = point.index
        if i in events:
            event = events[i]
            sign = 1 if event.direction == 'bullish' else -1
            state['trend'] = sign
            state['broken_support' if sign == 1 else 'broken_resistance'] = event.level
            f.at[i, 'break_sign'] = sign
        for key, value in state.items():
            f.at[i, key] = value
    return f


def prepare(frame, mtfa=True):
    f = structural_features(frame, '1h')
    f['recent_low'] = f.low.rolling(12).min()
    f['recent_high'] = f.high.rolling(12).max()
    f['prior_low'] = f.low.shift().rolling(24).min()
    f['prior_high'] = f.high.shift().rolling(24).max()
    if not mtfa:
        return f
    h = structural_features(resample_closed(frame, '4h', 4), '4h')
    keys = ['trend', 'support', 'resistance', 'broken_support', 'broken_resistance', 'atr']
    h['available_at'] = h.timestamp + pd.Timedelta(hours=4)
    # join on close-time: a 04:00 HTF open cannot appear until 08:00 UTC.
    left = f.assign(available_at=f.timestamp+pd.Timedelta(hours=1))
    right = h[['available_at']+keys].rename(columns={k: 'htf_'+k for k in keys})
    return pd.merge_asof(left, right, on='available_at', direction='backward')


def htf_evidence(row, sign):
    if row.get('htf_trend') != sign or not np.isfinite(row.get('htf_atr', np.nan)):
        return None
    side = 'support' if sign == 1 else 'resistance'
    tolerance = .25*row['htf_atr']
    for key in (f'htf_{side}', f'htf_broken_{side}'):
        level = row[key]
        if np.isfinite(level) and row['recent_low']-tolerance <= level <= row['recent_high']+tolerance:
            return {'htf_direction': int(sign), 'htf_poi': float(level), 'htf_poi_source': key}
    return None


def signals(frame, model, mtfa=True):
    if model not in {'structure_break', 'sweep_reclaim', 'trendline_break'}:
        raise ValueError(model)
    pending, used_lines, results = {}, set(), []
    rows = frame.to_dict('records')
    for i, r in enumerate(rows):
        if i < 30 or not np.isfinite(r['atr']):
            continue
        proposals = []
        if model == 'structure_break' and r['break_sign']:
            sign = int(r['break_sign'])
            anchor = r['support' if sign == 1 else 'resistance']
            proposals.append((sign, anchor, {'event': 'BOS/CHoCH', 'bar': i}))
        elif model == 'sweep_reclaim':
            for sign, s in list(pending.items()):
                invalid = r['low'] < s['low'] if sign == 1 else r['high'] > s['high']
                if i-s['index'] > 3 or invalid:
                    pending.pop(sign)
                elif sign*(r['close']-s['trigger']) > 0:
                    anchor = s['low'] if sign == 1 else s['high']
                    proposals.append((sign, anchor, {'event': 'sweep then reclaim break',
                                                     'sweep_bar': s['index'], 'swept_level': s['level']}))
                    pending.pop(sign)
            if r['low'] < r['prior_low'] < r['close']:
                pending[1] = dict(index=i, low=r['low'], high=r['high'], trigger=r['high'], level=r['prior_low'])
            if r['high'] > r['prior_high'] > r['close']:
                pending[-1] = dict(index=i, low=r['low'], high=r['high'], trigger=r['low'], level=r['prior_high'])
        elif model == 'trendline_break':
            for sign, side in ((1, 'high'), (-1, 'low')):
                a, b, ai, bi = (r[f'{side}_{k}'] for k in ('a', 'b', 'ai', 'bi'))
                if not all(np.isfinite(v) for v in (a,b,ai,bi)) or bi <= ai or i-bi > 24:
                    continue
                if sign*(b-a) >= 0:
                    continue
                key = (sign, int(ai), int(bi))
                if key in used_lines or i <= bi+3:
                    continue
                slope = (b-a)/(bi-ai)
                level = b+slope*(i-bi)
                prev = b+slope*(i-1-bi)
                # A line already broken before it was confirmable is not a fresh signal.
                if i == bi+4 and sign*(rows[i-1]['close']-prev) > 0:
                    used_lines.add(key)
                    continue
                if sign*(rows[i-1]['close']-prev) <= 0 < sign*(r['close']-level):
                    used_lines.add(key)
                    anchor = r['support' if sign == 1 else 'resistance']
                    proposals.append((sign, anchor, {'event': 'confirmed trendline break',
                        'anchors': [int(ai), int(bi)], 'line_level': float(level)}))
        # Opposite simultaneous events are ambiguous, not two independent trades.
        if len(proposals) != 1:
            continue
        sign, anchor, evidence = proposals[0]
        stop = anchor-sign*.25*r['atr']
        risk = sign*(r['close']-stop)
        if not np.isfinite(risk) or risk <= 0:
            continue
        if mtfa:
            context = htf_evidence(r, sign)
            if context is None:
                continue
            evidence.update(context)
        results.append(Signal(i, model, sign, float(stop), float(r['close']+sign*2*risk), evidence))
    return results


def execute(signal, future, fee=10, slip=2):
    """Next-open entry; all costs charged on traded notional, R on initial stop."""
    if len(future) < 48:
        return None
    sign, stop, target = signal.sign, signal.stop, signal.target
    entry = float(future.iloc[0].open)
    risk = sign*(entry-stop)
    reward = sign*(target-entry)
    if risk <= 0 or reward/risk < 1.5:
        return None
    for n, bar in enumerate(future.iloc[:48].itertuples(index=False)):
        if bar.low <= stop if sign == 1 else bar.high >= stop:
            exit_price = min(stop, bar.open) if sign == 1 else max(stop, bar.open)
            reason = 'stop'
        elif bar.high >= target if sign == 1 else bar.low <= target:
            exit_price, reason = target, 'target'
        elif n == 47:
            exit_price, reason = bar.close, 'time'
        else:
            continue
        gross = sign*(exit_price-entry)/risk
        cost = (entry+exit_price)*(fee+slip)/10000/risk
        stress = (entry+exit_price)*(fee+5)/10000/risk
        carry = entry*5/10000*((n+1)/24)/risk
        return dict(entry=entry, stop=stop, target=target, exit_price=float(exit_price),
                    gross_r=float(gross), net_r=float(gross-cost), stress_r=float(gross-stress-carry),
                    stop_bps=float(risk/entry*10000), bars=n+1, exit=reason,
                    entry_time=future.timestamp.iloc[0].isoformat(),
                    exit_time=(pd.Timestamp(bar.timestamp)+pd.Timedelta(hours=1)).isoformat())


def replay(frame, entries, start, end):
    busy, trades = -1, []
    for signal in entries:
        i = signal.index
        now = frame.timestamp.iloc[i]+pd.Timedelta(hours=1)
        if i <= busy or now < start or now+pd.Timedelta(hours=48) > end:
            continue
        future = frame.iloc[i+1:i+49]
        result = execute(signal, future)
        if result is None:
            continue
        busy = i+result['bars']
        trades.append(dict(signal_time=now.isoformat(), direction='long' if signal.sign == 1 else 'short',
                           evidence=asdict(signal)['evidence'], **result))
    return trades
