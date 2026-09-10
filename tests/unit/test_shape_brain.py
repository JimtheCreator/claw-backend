import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from core.config.execution_ladder import get_execution_chain
from core.use_cases.market_analysis.shape_brain import evaluate_shapes, surface_agreement
from core.use_cases.market_analysis.shape_features import shape_features
from core.use_cases.market_analysis.execution_tier import analyze_shape_shadow, discover_execution, settled_frame, qa_mode
from core.use_cases.market_analysis.momentum_history import MomentumCache
from core.use_cases.market_analysis.strategy_features import interval_offset
from tests.unit.test_strategy_brain import candles, row, resample
from tests.backtesting.run_shape_research import execute_candidate

ROLES = dict(macro='1d', intermediate='4h', execution='1h')


def inputs(**changes):
    return row(available_at=pd.Timestamp('2024-01-01T08:00Z'), timestamp=pd.Timestamp('2024-01-01T07:00Z'),
               own_return_504=.1, own_return_1512=.2, own_return_6048=.3, **changes)


def shapes(r, **kwargs):
    return evaluate_shapes(r, roles=ROLES, **kwargs)


@pytest.mark.parametrize('change', [dict(channel_trigger=0), dict(vwap=500), dict(real_flow=False),
    dict(delta=-10, delta3=-30), dict(break_sign=-1, displacement=False), dict(adx=0), dict(session_complete=False)])
def test_tsmom_signal_and_risk_are_independent_of_other_toolkits(change):
    before = inputs()
    base = shapes(before)['candidates'][1]
    candidate = shapes({**before, **change})['candidates'][1]
    for key in ('status', 'signal_fired', 'checks', 'risk', 'direction', 'horizons'):
        assert candidate[key] == base[key]
    assert candidate['status'] == 'eligible'
    assert set(candidate['checks']) == {'own_return_horizons_agree'}
    assert before == inputs()


def test_vwap_reversion_fires_without_momentum_structure_or_flow():
    r = inputs(close=103., atr=2., vwap=100., vwap_sigma=1., vwap_z=3.,
               momentum_complete=False, channel_trigger=0, real_flow=False, break_sign=0)
    out = shapes(r)
    c = out['candidates'][2]
    assert c['signal_fired'] and c['status'] == 'eligible' and c['direction'] == 'short'
    assert c['risk']['target'] == 100 and c['risk']['stop'] == 103.5
    assert c['optional_cvd']['status'] == 'unavailable'
    assert out['arbitration']['selected_strategy'] == 'vwap_reversion_v1'
    mirrored = shapes({**r, 'close': 97., 'vwap_z': -3.})['candidates'][2]
    assert mirrored['direction'] == 'long' and mirrored['risk']['stop'] == 96.5
    assert mirrored['risk']['target'] == 100


def test_risk_rejection_does_not_erase_a_fired_standalone_signal():
    c = shapes(inputs(atr=.001))['candidates'][1]
    assert c['signal_fired'] is True and c['status'] == 'rejected'
    assert c['failed_checks'][0].startswith('risk:')
    assert c['checks'] == {'own_return_horizons_agree': True}


def test_cvd_is_optional_directional_diagnostic_never_a_gate():
    aligned = shapes(inputs())['candidates'][1]
    opposed = shapes(inputs(delta3=-50))['candidates'][1]
    missing = shapes(inputs(real_flow=False))['candidates'][1]
    assert [c['optional_cvd']['score_adjustment'] for c in (aligned, opposed, missing)] == [1, -1, 0]
    assert all(c['status'] == 'eligible' for c in (aligned, opposed, missing))


def test_unavailable_and_disagreeing_horizons_are_not_the_same():
    assert shapes(inputs(momentum_complete=False))['candidates'][1]['status'] == 'unavailable'
    c = shapes(inputs(momentum_sign=0))['candidates'][1]
    assert c['status'] == 'rejected' and not c['signal_fired']


def test_smc_reuses_original_predicate_but_on_execution_series():
    from core.use_cases.market_analysis.strategy_brain import evaluate_strategies
    poi = dict(sign=1, bottom=97., top=99., available_at='2024-01-01T00:00:00Z', reaction_at='2024-01-01T04:00:00Z')
    r = inputs(break_sign=1, displacement=True)
    old = evaluate_strategies(r, mtfa_enabled=True, htf_available=True, htf_reactions=[poi])[0]
    new = shapes(r, htf_available=True, reactions=[poi])['candidates'][0]
    assert all(new[k] == old[k] for k in ('status', 'direction', 'checks', 'risk', 'context'))
    assert new['execution_interval'] == '1h'
    facts = {f['group']: f for f in new['chart_evidence']}
    assert facts['anchor_poi']['timeframe'] == '1d'
    assert facts['middle_reaction']['timeframe'] == '4h'
    assert facts['local_displacement_break']['timeframe'] == '1h'


def test_missing_or_disabled_macro_is_isolated_from_standalone_shapes():
    r = inputs()
    dirty = shapes(r, htf_available=True, reactions=[dict(sign=-1, SECRET=123)], qa_htf_off=True)
    clean = shapes(r, qa_htf_off=True)
    assert dirty == clean and 'SECRET' not in json.dumps(dirty)
    assert dirty['roles']['macro'] is None and 'checks' not in dirty['candidates'][0]
    missing = shapes(r, htf_available=False)
    assert missing['candidates'][1:] == clean['candidates'][1:]


def test_finer_tier_required_and_chain_is_not_mutable_shared_state():
    with pytest.raises(ValueError):
        evaluate_shapes(inputs(), roles=dict(intermediate='1h', execution='1h'))
    assert get_execution_chain('1m') == get_execution_chain('unknown') == []
    chain = get_execution_chain('4h'); chain.clear()
    assert get_execution_chain('4h')[0] == '1h'
    now = pd.Timestamp('2024-06-01', tz='UTC')
    for intermediate in ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d', '3d', '1w', '1M'):
        for execution in get_execution_chain(intermediate):
            assert now+interval_offset(execution) < now+interval_offset(intermediate)


def test_arbitration_is_permutation_invariant_unblended_and_unranked():
    a = dict(strategy='tsmom_v1', status='eligible', direction='long', risk={'target': 102})
    b = dict(strategy='vwap_reversion_v1', status='eligible', direction='long', risk={'target': 103})
    assert surface_agreement([a, b]) == surface_agreement([b, a])
    assert surface_agreement([a, b])['status'] == 'agreement_unranked'
    assert surface_agreement([a, b])['selected_strategy'] is None
    b['direction'] = 'short'
    assert surface_agreement([a, b])['status'] == 'conflict'
    assert surface_agreement([a])['selected_strategy'] == 'tsmom_v1'


def test_vwap_bands_are_prefix_causal_and_volume_weighted():
    raw = candles(900)
    raw.volume = np.arange(1, 901)
    full = shape_features(raw, '1h')
    pd.testing.assert_frame_equal(full.iloc[:381].reset_index(drop=True), shape_features(raw.iloc[:381], '1h'))
    tail = full[full.timestamp.dt.floor('D') == full.timestamp.iloc[-1].floor('D')]
    typical = (tail.high+tail.low+tail.close)/3
    center = np.average(typical, weights=tail.volume)
    sigma = np.sqrt(np.average((typical-center)**2, weights=tail.volume))
    assert full.vwap.iloc[-1] == pytest.approx(center)
    assert full.vwap_sigma.iloc[-1] == pytest.approx(sigma)


def klines(frame):
    return [[int(r.timestamp.value//10**6), r.open, r.high, r.low, r.close, r.volume, 0, 0, 0, r.taker_buy_volume] for r in frame.itertuples()]


def test_execution_fetch_filters_open_candle_and_rejects_stale_or_gapped_data():
    raw = candles(100)
    as_of = raw.timestamp.iloc[-1]
    f = settled_frame(klines(raw), '1h', as_of)
    assert len(f) == 99 and f.timestamp.iloc[-1]+pd.Timedelta(hours=1) == as_of
    with pytest.raises(ValueError):
        settled_frame(klines(raw.drop(index=80)), '1h', as_of)
    with pytest.raises(ValueError):
        settled_frame(klines(raw), '1h', as_of+pd.Timedelta(hours=3))


def test_nearest_execution_failure_tries_declared_finer_series_only():
    raw = candles(100); raw.timestamp = pd.date_range('2024-01-01', periods=100, freq='15min', tz='UTC')
    as_of = raw.timestamp.iloc[-1]+pd.Timedelta(minutes=15)
    calls = []
    async def fetch(**kwargs):
        calls.append(kwargs['interval'])
        if kwargs['interval'] == '1h':
            raise RuntimeError('Exchange missing interval')
        return klines(raw)
    tf, frame, failures = asyncio.run(discover_execution('TEST', '4h', as_of, fetch))
    assert tf == '15m' and calls == ['1h', '15m'] and failures == {'1h': 'RuntimeError'}
    assert len(frame) == 100


def test_auto_shadow_fetches_finer_data_and_missing_macro_does_not_kill_tsmom(tmp_path):
    raw = candles(6050)
    cache = MomentumCache(tmp_path/'cache.sqlite3'); cache.put('TEST', '1h', raw)
    as_of = raw.timestamp.iloc[-1]+pd.Timedelta(hours=1)
    requested = resample(raw, '4h', 4).tail(200).reset_index(drop=True)
    before = requested.copy(deep=True)
    calls = []
    async def fetch(**kwargs):
        calls.append(kwargs['interval'])
        if kwargs['interval'] == '1d':
            raise RuntimeError('Missing macro')
        assert kwargs['interval'] == '1h'
        return klines(raw.tail(1000))
    auto = asyncio.run(analyze_shape_shadow('TEST', '4h', requested, as_of=as_of, fetch_page=fetch, cache=cache))
    assert calls == ['1h', '1d']
    assert auto['candidates'][0]['status'] == 'unavailable'
    assert auto['candidates'][1]['status'] != 'unavailable'
    assert auto['momentum_history']['complete'] and auto['roles']['execution'] == '1h'
    calls.clear()
    off = asyncio.run(analyze_shape_shadow('TEST', '4h', requested, as_of=as_of, fetch_page=fetch, cache=cache, htf_mode='off'))
    assert calls == ['1h'] and off['roles']['macro'] is None
    assert auto['candidates'][1:] == off['candidates'][1:]
    json.dumps(off, allow_nan=False)
    pd.testing.assert_frame_equal(requested, before)


def test_unsupported_finer_interval_is_explicit_not_same_timeframe_fallback():
    raw = candles(100)
    raw.timestamp = pd.date_range('2024-01-01', periods=100, freq='min', tz='UTC')
    never = AsyncMock(side_effect=AssertionError('Must not fetch same timeframe'))
    out = asyncio.run(analyze_shape_shadow('TEST', '1m', raw, as_of=raw.timestamp.iloc[-1]+pd.Timedelta(minutes=1), fetch_page=never))
    assert out['arbitration']['status'] == 'no_execution_data' and out['roles']['execution'] is None
    never.assert_not_called()
    with pytest.raises(ValueError): qa_mode('enabled')


def test_momentum_cache_failure_does_not_erase_vwap_or_leak_details(tmp_path, monkeypatch):
    raw = candles(900)
    as_of = raw.timestamp.iloc[-1]+pd.Timedelta(hours=1)
    requested = resample(raw, '4h', 4)
    async def fetch(**kwargs):
        return klines(raw)
    monkeypatch.setattr('core.use_cases.market_analysis.execution_tier.load_momentum_history',
                        AsyncMock(side_effect=RuntimeError('private provider detail')))
    out = asyncio.run(analyze_shape_shadow('TEST', '4h', requested, as_of=as_of,
                      fetch_page=fetch, htf_mode='off'))
    assert out['candidates'][1]['status'] == 'unavailable'
    assert out['candidates'][2]['status'] != 'unavailable'
    assert 'private provider detail' not in json.dumps(out, allow_nan=False)


def test_chart_facts_explain_each_independent_shape_and_failed_risk():
    c = shapes(inputs(atr=.001))['candidates'][1]
    facts = {f['group']: f for f in c['chart_evidence']}
    assert facts['own_return_horizons']['returns'] == c['horizons']
    assert not facts['risk_contract']['passed']
    assert not facts['optional_cvd']['mandatory']
    c = shapes(inputs(close=103., atr=2., vwap=100., vwap_sigma=1., vwap_z=3.))['candidates'][2]
    facts = {f['group']: f for f in c['chart_evidence']}
    assert facts['session_vwap_band']['upper_two_sigma'] == 102
    assert facts['session_vwap_band']['z_score'] == 3


def test_research_chart_preserves_requested_interval_and_never_draws_future_candles():
    from scripts.preview_shape_research import shape_figure, evidence_text
    raw = candles(900)
    f = shape_features(raw, '1h')
    at = 800
    r = f.iloc[at].to_dict()
    r.update(momentum_complete=True, momentum_sign=1, atr=2., own_return_504=.1)
    c = shapes(r)['candidates'][1]
    before = deepcopy(c)
    fig = shape_figure('TEST', resample(raw, '4h', 4), raw, c, ROLES)
    observed = [t for t in fig.data if t.type == 'candlestick']
    assert len(observed) == 2 and len(observed[0].x) == 60 and len(observed[1].x) == 24
    assert all(pd.to_datetime(x, utc=True) <= r['available_at'] for t in observed for x in t.x)
    assert all(not a.showarrow for a in fig.layout.annotations)
    assert any(a.text == 'Long ▲' for a in fig.layout.annotations)
    assert fig.layout.meta['production_promoted'] is False
    assert c == before
    assert all(evidence_text(f) for f in c['chart_evidence'])


def test_optional_shadow_mode_cannot_enable_production():
    from core.use_cases.market_analysis.brain_shadow import brain_policy
    assert brain_policy('shadow_v2') == 'shadow_v2'
    assert brain_policy('legacy') == 'legacy'
    with pytest.raises(ValueError):
        brain_policy('live_v2')


def test_simulator_costs_gaps_stop_first_and_actual_duration():
    c = dict(direction='long', risk=dict(stop=99., target=102., target_source='2R'))
    future = pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=2, freq='15min', tz='UTC'),
                               open=[100., 98.], high=[101., 103.], low=[99.5, 97.], close=[100., 102.]))
    result = execute_candidate(c, future, interval='15m', hold=2)
    assert result['exit'] == 'stop' and result['exit_price'] == 98
    assert result['gross_r'] == -2
    assert result['net_r'] == pytest.approx(-2-198*.0012)
    assert result['stress_r'] == pytest.approx(-2-198*.0015-100*.0005/48)
    future.loc[0, 'open'] = 101.5
    assert execute_candidate(c, future, hold=2) is None
