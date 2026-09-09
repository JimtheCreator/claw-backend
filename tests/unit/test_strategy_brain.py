import copy
import json

import numpy as np
import pandas as pd
import pytest

from core.engines.regime_engine import RegimeEngine, wilder
from core.use_cases.market_analysis.brain_shadow import brain_policy
from core.use_cases.market_analysis.strategy_brain import evaluate_brain, evaluate_strategies, arbitrate
from core.use_cases.market_analysis.strategy_features import local_features, top_down_context, momentum_features
from core.use_cases.market_analysis.strategy_risk import evaluate_risk, StrategyRiskPolicy


def row(**changes):
    result = dict(close=100., atr=1., support=98., resistance=105., stop_support=98., stop_resistance=105.,
                  momentum_complete=True, momentum_sign=1, channel_trigger=1, session_bars=4,
                  session_complete=True, vwap=99., delta=10., delta3=30., real_flow=True,
                  break_sign=0, displacement=False, preceding_sweep=False, ob_fvg_overlap=False,
                  adx=30., atr_percentile=80.)
    return {**result, **changes}


def candles(n=900):
    rng = np.random.default_rng(32)
    close = 200*np.exp(np.cumsum(rng.normal(0, .006, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        open=op, high=np.maximum(op, close)+.4, low=np.minimum(op, close)-.4, close=close,
        volume=np.full(n, 100.), taker_buy_volume=np.full(n, 65.)))


def resample(f, rule, size):
    g = f.set_index('timestamp').resample(rule)
    out = g.agg(dict(open='first', high='max', low='min', close='last', volume='sum', taker_buy_volume='sum'))
    return out[g.close.count() == size].reset_index()


def test_momentum_has_own_trigger_no_bos_or_htf_dependency():
    r = row()
    off = evaluate_brain(r, mtfa_enabled=False, htf_available=True,
                         htf_reactions=[{'sign': -1, 'SECRET_HTF_FIELD': 1}])
    on = evaluate_brain(r, mtfa_enabled=True, htf_available=False)
    assert off['arbitration']['action'] == 'long'
    assert off['candidates'][1] == on['candidates'][1]
    assert 'SECRET_HTF_FIELD' not in json.dumps(off)
    assert 'checks' not in off['candidates'][0]  # skipped, not a free point
    assert r == row()


def test_proxy_or_partial_cvd_cannot_confirm():
    for changes in (dict(real_flow=False), dict(delta3=None), dict(delta=-1), dict(session_complete=False)):
        result = evaluate_brain(row(**changes), mtfa_enabled=False)
        assert result['arbitration']['action'] == 'wait'
        assert result['candidates'][2]['status'] == 'unavailable'


def test_tier_three_all_four_checks_stricter_floor_and_no_available_disagreement_escape():
    valid = row(momentum_complete=False, break_sign=1, displacement=True, preceding_sweep=True, ob_fvg_overlap=True)
    assert evaluate_strategies(valid, mtfa_enabled=False)[2]['status'] == 'eligible'
    for group in ('displacement', 'preceding_sweep', 'ob_fvg_overlap'):
        assert evaluate_strategies({**valid, group: False}, mtfa_enabled=False)[2]['status'] == 'rejected'
    narrow = {**valid, 'atr': .1, 'stop_support': 99.6}
    assert not evaluate_strategies(narrow, mtfa_enabled=False)[2]['risk']['eligible']
    assert evaluate_strategies({**valid, 'momentum_complete': True, 'momentum_sign': 0}, mtfa_enabled=False)[2]['status'] == 'unavailable'
    assert evaluate_strategies(valid, mtfa_enabled=True, htf_available=False)[2]['status'] == 'unavailable'


def test_conflict_cannot_be_resolved_by_order_or_trend_regime():
    a = dict(strategy='smc_location_v1', status='eligible', direction='short', risk={})
    b = dict(strategy='momentum_flow_v1', status='eligible', direction='long', risk={})
    for state in ('trending', 'non_trending', 'unknown'):
        assert arbitrate([a, b], {'trend': state}) == arbitrate([b, a], {'trend': state})
        assert arbitrate([a, b], {'trend': state})['status'] == 'conflict'
    a['direction'] = 'long'
    assert arbitrate([a, b], {'trend': 'trending'})['selected_strategy'] == b['strategy']
    assert arbitrate([b, a], {'trend': 'non_trending'})['selected_strategy'] == a['strategy']


def test_missing_price_or_risk_is_json_safe_and_never_entry():
    result = evaluate_risk(100, np.nan, None, 1, StrategyRiskPolicy(), target_source='pivot')
    assert not result['eligible']
    json.dumps(result, allow_nan=False)
    for name in ('production_v1', '', 'indicators_v1'):
        with pytest.raises(ValueError):
            brain_policy(name)


def test_local_features_are_prefix_causal_and_do_not_mutate():
    f = candles()
    before = f.copy(deep=True)
    full = local_features(f, '1h')
    for n in (100, 381, 645):
        prefix = local_features(f.iloc[:n], '1h')
        pd.testing.assert_frame_equal(prefix, full.iloc[:n].reset_index(drop=True))
    pd.testing.assert_frame_equal(f, before)


def test_top_down_uses_closed_middle_and_anchor_poi_without_future_status():
    raw = candles(1800)
    local = local_features(raw, '1h')
    full = top_down_context(local, resample(raw, '4h', 4), resample(raw, '1D', 24),
                            middle_interval='4h', anchor_interval='1d')
    assert any(full), 'Fixture must exercise real reactions'
    for n in (503, 997, 1637):
        prefix = top_down_context(local.iloc[:n], resample(raw.iloc[:n], '4h', 4), resample(raw.iloc[:n], '1D', 24),
                                  middle_interval='4h', anchor_interval='1d')
        assert prefix == full[:n]
    for i, reactions in enumerate(full):
        for reaction in reactions:
            assert pd.Timestamp(reaction['available_at']) <= pd.Timestamp(reaction['reaction_at'])
            assert pd.Timestamp(reaction['reaction_at']) <= raw.timestamp.iloc[i]


def test_long_momentum_windows_are_not_shrunk_and_genuine_delta_matches():
    raw = candles(6050)
    f = momentum_features(raw, '1h')
    assert not f.momentum_complete.iloc[6047]
    assert bool(f.momentum_complete.iloc[6048])
    assert f.delta.iloc[-1] == 30
    assert f.delta3.iloc[-1] == 90
    with pytest.raises(ValueError):
        momentum_features(raw, 'bad')


def test_wilder_seed_flat_and_prefix():
    values = pd.Series(range(1, 31), dtype=float)
    result = wilder(values)
    assert result.iloc[13] == 7.5
    assert result.iloc[14] == pytest.approx((7.5*13+15)/14)
    pd.testing.assert_series_equal(result.iloc[:22], wilder(values.iloc[:22]))
    raw = candles(300)
    raw[['open', 'high', 'low', 'close']] = [100, 101, 99, 100]
    f = RegimeEngine().features(raw)
    assert f.adx.iloc[-1] == 0


def test_vector_features_match_existing_indicator_engines():
    from core.engines.tsmom_engine import TSMOMEngine
    from core.engines.vwap_engine import VWAPEngine
    from core.engines.cvd_engine import CVDEngine
    raw = candles(6060)
    features = momentum_features(raw, '1h')
    mom = TSMOMEngine('1h').calculate_signal(raw)
    vwap = VWAPEngine('1h').calculate_vwap(raw)
    cvd = CVDEngine('1h').calculate_cvd(raw)
    expected = int(mom.combined_signal) if abs(mom.combined_signal) == 1 else 0
    assert features.momentum_sign.iloc[-1] == expected
    assert features.vwap.iloc[-1] == pytest.approx(vwap.points[-1].vwap)
    assert features.delta3.iloc[-1] == pytest.approx(sum(p.delta for p in cvd.points[-3:]))


def test_shadow_adapter_ignores_poisoned_htf_and_keeps_full_history(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    from core.use_cases.market_analysis.momentum_history import MomentumCache
    from core.use_cases.market_analysis.brain_shadow import analyze_brain_shadow
    raw = candles(6050)
    cache = MomentumCache(tmp_path/'history.sqlite3')
    cache.put('BTCUSDT', '1h', raw)
    frame = raw.tail(400).reset_index(drop=True)
    before = frame.copy(deep=True)
    never = AsyncMock(side_effect=AssertionError('Cached history should suffice'))
    outputs = []
    for context in ({}, {'4h': 'POISON', '1d': 'POISON'}):
        outputs.append(asyncio.run(analyze_brain_shadow('BTCUSDT', '1h', frame, mtfa_enabled=False,
                          htf_frames=context, fetch_page=never, cache=cache)))
    assert outputs[0] == outputs[1]
    assert outputs[0]['momentum_history']['complete']
    assert outputs[0]['momentum_history']['available_bars'] == 6049
    json.dumps(outputs[0], allow_nan=False)
    pd.testing.assert_frame_equal(before, frame)


def test_risk_rechecked_at_fill_and_no_account_position_size():
    policy = StrategyRiskPolicy()
    planned = evaluate_risk(100, 99, 102, 1, policy, target_source='risk_multiple')
    assert planned['eligible'] and planned['position_size'] is None
    assert not evaluate_risk(101, 99, 102, 1, policy, target_source='risk_multiple')['eligible']
    assert not evaluate_risk(98, 99, 102, 1, policy, target_source='risk_multiple')['eligible']
    assert planned['net_target_r'] < planned['gross_target_r']
