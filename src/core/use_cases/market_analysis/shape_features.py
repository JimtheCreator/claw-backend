"""Causal standalone VWAP/TSMOM research inputs; detectors remain unchanged."""
import numpy as np
import pandas as pd

from core.engines.tsmom_engine import TSMOMEngine
from core.use_cases.market_analysis.strategy_features import local_features


def shape_features(frame, interval):
    f = local_features(frame, interval)
    session = f.timestamp.dt.floor('D')
    typical = (f.high+f.low+f.close)/3
    volume = f.volume.groupby(session).cumsum().replace(0, np.nan)
    second = (typical.pow(2)*f.volume).groupby(session).cumsum()/volume
    f['vwap_sigma'] = (second-f.vwap.pow(2)).clip(lower=0).pow(.5)
    f['vwap_z'] = (f.close-f.vwap)/f.vwap_sigma.replace(0, np.nan)
    for h in TSMOMEngine._interval_lookback_defaults[interval]:
        f[f'own_return_{h}'] = f.close/f.close.shift(h)-1
    return f
