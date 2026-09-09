"""Causal descriptive regime features. Thresholds are hypotheses, not an edge."""
import numpy as np
import pandas as pd


def wilder(values, period=14):
    """Seed at the first complete window; thereafter Wilder's recursive mean."""
    values = pd.Series(values, dtype=float)
    out = np.full(len(values), np.nan)
    seed = values.rolling(period, min_periods=period).mean().first_valid_index()
    if seed is None:
        return pd.Series(out, index=values.index)
    start = values.index.get_loc(seed)
    out[start] = values.loc[:seed].iloc[-period:].mean()
    for i in range(start+1, len(values)):
        if np.isfinite(values.iloc[i]):
            out[i] = (out[i-1]*(period-1)+values.iloc[i])/period
    return pd.Series(out, index=values.index)


class RegimeEngine:
    def features(self, frame):
        previous = frame.close.shift()
        tr = pd.concat([frame.high-frame.low, (frame.high-previous).abs(),
                        (frame.low-previous).abs()], axis=1).max(axis=1)
        atr = wilder(tr)
        up, down = frame.high.diff(), -frame.low.diff()
        plus = wilder(up.where((up > down) & (up > 0), 0.0))/atr.replace(0, np.nan)
        minus = wilder(down.where((down > up) & (down > 0), 0.0))/atr.replace(0, np.nan)
        total = plus+minus
        dx = 100*(plus-minus).abs()/total.replace(0, np.nan)
        dx = dx.mask(total == 0, 0.0)
        adx = wilder(dx)
        relative = atr/frame.close
        percentile = relative.rolling(252, min_periods=252).rank(pct=True)*100
        return pd.DataFrame(dict(atr=atr, adx=adx, atr_percentile=percentile))

    @staticmethod
    def describe(row):
        adx, percentile = row.get('adx'), row.get('atr_percentile')
        valid = adx is not None and np.isfinite(adx)
        vol_valid = percentile is not None and np.isfinite(percentile)
        return dict(trend='unknown' if not valid else 'trending' if adx >= 25 else 'non_trending',
                    adx=float(adx) if valid else None,
                    volatility='unknown' if not vol_valid else 'high' if percentile >= 75 else
                    'low' if percentile <= 25 else 'normal',
                    atr_percentile=float(percentile) if vol_valid else None,
                    validation='unvalidated_descriptive_thresholds')
