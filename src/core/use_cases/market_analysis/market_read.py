"""Observed local market context, independent of trade eligibility and HTF gates."""
from math import isfinite

import pandas as pd


def build_market_read(candles, interval, structure, swings):
    if candles.empty:
        return None
    current = float(candles.close.iloc[-1])
    if not isfinite(current) or current <= 0:
        return None
    trend = getattr(structure, 'trend', None) or 'undetermined'
    levels = {'support': None, 'resistance': None}
    for pivot in getattr(swings, 'swings', []):
        if (not pivot.confirmed or not 0 <= pivot.index < len(candles)
                or pivot.index+getattr(swings, 'window', 0) >= len(candles)):
            continue
        value = float(pivot.price)
        if not isfinite(value) or value <= 0:
            continue
        after = candles.close.iloc[pivot.index+1:]
        side = None
        if pivot.type == 'low' and value < current and not (after < value).any():
            side = 'support'
        elif pivot.type == 'high' and value > current and not (after > value).any():
            side = 'resistance'
        if side and (levels[side] is None or abs(value-current) < abs(levels[side]['price']-current)):
            levels[side] = {'price': value, 'index': int(pivot.index),
                            'timestamp': pd.Timestamp(candles.timestamp.iloc[pivot.index]).isoformat(),
                            'source': 'confirmed_local_swing'}
    events = [e for e in getattr(structure, 'events', []) if 0 <= e.index < len(candles)]
    last = max(events, key=lambda e: e.index, default=None)
    last_break = None if last is None else {
        'kind': last.kind, 'direction': last.direction, 'price': float(last.level),
        'timestamp': pd.Timestamp(candles.timestamp.iloc[last.index]).isoformat(),
        'index': int(last.index), 'bars_ago': len(candles)-1-int(last.index),
        'label': f'Observed {last.direction} {last.kind}', 'plot': True,
    }
    focus = levels['support' if trend == 'bearish' else 'resistance'] if trend in {'bullish', 'bearish'} else None
    crossing = 'below' if trend == 'bearish' else 'above'
    next_check = (f'A {interval} candle close {crossing} {focus["price"]:,.6g} would extend {trend} structure. '
                  'Reassess the entry setup then; a level is not an entry signal.' if focus else
                  'No unbroken confirmed level in the trend direction is available. Watch for a new confirmed swing; do not invent an entry.')
    return {'trend_direction': trend, 'as_of_bar_open': pd.Timestamp(candles.timestamp.iloc[-1]).isoformat(),
            **levels, 'last_break': last_break, 'next_check': next_check,
            'meaning': 'Observed requested-timeframe context, not a forecast or permission to trade.'}
