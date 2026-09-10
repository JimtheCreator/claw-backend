from types import SimpleNamespace as NS
from copy import deepcopy

import pytest

from tests.unit.test_mtfa_trade_plan import _candles
from core.use_cases.market_analysis.market_read import build_market_read
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from core.engines.analysis_chart_presentation import AnalysisChartPresentation


def inputs():
    df = _candles()
    swings = NS(window=3, swings=[NS(index=5, price=99., type='low', confirmed=True),
                                 NS(index=8, price=101., type='high', confirmed=True)])
    structure = NS(trend='bearish', events=[NS(index=10, kind='BOS', direction='bearish', level=100.5)])
    return df, swings, structure


def test_context_uses_confirmed_unbroken_local_levels_not_future_pivots():
    df, swings, structure = inputs()
    swings.swings.append(NS(index=19, price=100.1, type='high', confirmed=False))
    r = build_market_read(df, '1h', structure, swings)
    assert r['support']['price'] == 99
    assert r['resistance']['price'] == 101
    assert r['last_break']['bars_ago'] == 9
    assert 'close below 99' in r['next_check']
    df.loc[11, 'close'] = 98
    assert build_market_read(df, '1h', structure, swings)['support'] is None


@pytest.mark.parametrize('context', ['mixed', 'incomplete', 'local'])
def test_entry_rejection_does_not_erase_local_analysis(context):
    df, swings, structure = inputs()
    mtfa = {'enabled': context != 'local', 'htf_trends': {'4h':'bearish', '1d':'bullish'}}
    if context == 'incomplete':
        mtfa['htf_unavailable'] = {'1d':'source unavailable'}
    plan = build_trade_plan(df, interval='1h', structure=structure, swings=swings,
                           mtfa=mtfa, premium_discount=None, liquidity=None, fvg=None,
                           order_blocks=None, confluence=None)
    assert plan['action'] == 'wait'
    assert plan['entry_level'] is None
    assert plan['market_read']['support']['price'] == 99
    original = deepcopy(plan)
    fig = AnalysisChartPresentation(df, {'symbol':'TEST', 'trade_plan':plan}, {}).figure()
    caption = ' '.join(a.text.replace('<br>', ' ') for a in fig.layout.annotations)
    assert 'Support 99.00' in caption and 'Resistance 101.00' in caption
    assert 'BEARISH STRUCTURE · NO ENTRY CONFIRMED' in caption
    assert not any(t.type == 'scatter' and 'lines' in (t.mode or '') for t in fig.data)
    assert plan == original


def test_off_context_does_not_depend_on_stale_higher_timeframes():
    df, swings, structure = inputs()
    kwargs = dict(interval='1h', structure=structure, swings=swings, premium_discount=None,
                  liquidity=None, fvg=None, order_blocks=None, confluence=None)
    clean = build_trade_plan(df, mtfa={'enabled':False}, **kwargs)
    stale = build_trade_plan(df, mtfa={'enabled':False, 'htf_trends':{'4h':'bullish'}}, **kwargs)
    assert clean == stale


@pytest.mark.parametrize('context', ['mixed', 'incomplete'])
def test_context_gate_blocks_entry_but_keeps_explicit_local_forecast(context):
    df, swings, structure = inputs()
    mtfa = {'enabled': True, 'htf_trends': {'4h': 'bearish', '1d': 'bullish'}}
    if context == 'incomplete':
        mtfa['htf_unavailable'] = {'1d': 'unavailable'}
    liquidity = NS(pools=[NS(side='sell_side', level=97., last_index=0)])
    plan = build_trade_plan(df, interval='1h', structure=structure, swings=swings,
                           mtfa=mtfa, premium_discount=None, liquidity=liquidity,
                           fvg=None, order_blocks=None, confluence=None)
    assert plan['action'] == 'wait' and plan['primary_scenario'] is None
    assert all(plan[k] is None for k in ('entry_level', 'stop_loss', 'take_profit'))
    assert plan['forecast_scenario']['direction'] == 'bearish'
    assert plan['forecast_scenario']['basis'] == 'local_structure'
    fig = AnalysisChartPresentation(df, {'symbol': 'TEST', 'trade_plan': plan}, {}).figure()
    badge = next(a for a in fig.layout.annotations if a.name == 'Forecast direction label')
    assert badge.text == '<b>Short ▼</b>'
    assert not any(t.name == 'Conditional forecast' for t in fig.data)
    assert any('HTF context is not validated' in a.text for a in fig.layout.annotations)
