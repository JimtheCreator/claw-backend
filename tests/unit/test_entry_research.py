import numpy as np
import pandas as pd
import pytest

from tests.backtesting.research_entry_models import Signal, prepare, signals, execute, replay
from tests.backtesting.run_entry_research import stats, eligible


def candles(n=250):
    rng = np.random.default_rng(51)
    close = 100+np.cumsum(rng.normal(0, .6, n))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        open=op, high=np.maximum(op, close)+.3, low=np.minimum(op, close)-.3,
        close=close, volume=np.ones(n)))


@pytest.mark.parametrize('length', [61, 99, 147, 199])
def test_full_history_features_equal_prefix_available_at_that_time(length):
    raw = candles()
    full = prepare(raw)
    prefix = prepare(raw.iloc[:length].copy())
    pd.testing.assert_series_equal(full.iloc[length-1], prefix.iloc[-1])
    for model in ('structure_break', 'sweep_reclaim', 'trendline_break'):
        for on in (False, True):
            assert [s for s in signals(full, model, on) if s.index < length] == signals(prefix, model, on)


def test_unclosed_four_hour_bar_cannot_enter_context():
    raw = candles()
    full = prepare(raw)
    at = 101  # closes 06:00; containing 04:00 HTF bar still open
    changed = raw.copy()
    changed.loc[at+1:at+2, ['high', 'close']] *= 2
    updated = prepare(changed)
    keys = [k for k in full.columns if k.startswith('htf_')]
    pd.testing.assert_series_equal(full.loc[at, keys], updated.loc[at, keys])


def test_off_is_identical_without_htf_or_with_poisoned_htf():
    raw = candles()
    full = prepare(raw)
    local = prepare(raw, mtfa=False)
    for key in full.columns:
        if key.startswith('htf_'):
            full[key] = -999
    for model in ('structure_break', 'sweep_reclaim', 'trendline_break'):
        actual = signals(full, model, False)
        assert actual == signals(local, model, False)
        assert all(not any(k.startswith('htf_') for k in s.evidence) for s in actual)


def future():
    f = candles(48)
    f[['open', 'close']] = 100.
    f['high'], f['low'] = 101., 99.
    return f


def test_both_hit_bar_uses_stop_and_charges_actual_notional_cost():
    f = future()
    f.loc[0, ['high', 'low']] = [106, 97]
    result = execute(Signal(0, 'test', 1, 98, 104, {}), f)
    assert result['exit'] == 'stop'
    assert result['gross_r'] == -1
    assert result['net_r'] == pytest.approx(-1-(100+98)*.0012/2)


def test_gap_through_stop_cancels_entry_but_held_position_gaps_at_open():
    s = Signal(0, 'test', 1, 98, 104, {})
    f = future()
    f.loc[0, 'open'] = 97
    assert execute(s, f) is None
    f.loc[0, 'open'] = 100
    f.loc[1, ['open', 'low']] = [96, 95]
    r = execute(s, f)
    assert r['exit_price'] == 96
    assert r['gross_r'] == -2


@pytest.mark.parametrize('sign', [1, -1])
def test_next_open_entry_not_signal_close_and_48_bar_time_exit(sign):
    f = future()
    f.loc[0, 'open'] = 100.2 if sign == 1 else 99.8
    r = execute(Signal(0, 'test', sign, 100-sign*3, 100+sign*6, {}), f)
    assert r['entry'] == f.loc[0, 'open']
    assert r['bars'] == 48
    assert r['exit'] == 'time'
    assert r['stress_r'] < r['net_r'] < r['gross_r']


def test_replay_has_no_overlaps_or_cross_window_exits():
    f = pd.concat([future(), future().assign(timestamp=lambda d: d.timestamp+pd.Timedelta(hours=48))], ignore_index=True)
    candidates = [Signal(i, 'test', 1, 97, 106, {}) for i in (0, 1, 20, 48, 49, 80)]
    rows = replay(f, candidates, f.timestamp.iloc[0], f.timestamp.iloc[-1]+pd.Timedelta(hours=1))
    assert len(rows) == 1
    assert rows[0]['bars'] == 48


def test_no_sample_cannot_be_selected():
    assert not eligible(stats([]))
