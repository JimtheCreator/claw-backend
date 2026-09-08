import asyncio
import ast
import hashlib
import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, MagicMock
import sys
import zipfile

import pandas as pd
import pytest

from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.engines.cvd_engine import CVDEngine
from core.use_cases.market_analysis.data_access import _format_ohlcv_response, _recover_taker_buy_volume
from core.use_cases.market_analysis.setup_evidence import indicator_evidence, rank_entry_zones
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from tests.unit.test_setup_evidence import fixture


def entity(flow=None):
    return MarketDataEntity(symbol='BTCUSDT', interval='1h',
        timestamp=datetime(2026, 9, 8, tzinfo=timezone.utc),
        open=100, high=102, low=99, close=101, volume=10, taker_buy_volume=flow)


def kline(flow='2'):
    return [int(entity().timestamp.timestamp()*1000), '100', '102', '99', '101', '10',
            int((entity().timestamp+timedelta(hours=1)).timestamp()*1000)-1, '1000', 8, flow, '200', '0']


@pytest.mark.parametrize('value', [None, 'bad', float('nan'), float('inf'), -1, 11])
def test_bad_flow_is_unknown_not_zero_or_a_bad_ohlcv_candle(value):
    e = entity(value)
    assert e.taker_buy_volume is None
    assert json.loads(e.model_dump_json())['taker_buy_volume'] is None
    assert e.volume == 10


def test_binance_field_nine_survives_cold_and_refresh_fetches(monkeypatch):
    save = Mock()
    monkeypatch.setitem(sys.modules, 'src.core.services.tasks', NS(save_market_data_task=NS(delay=save)))
    market = importlib.import_module('core.use_cases.market.market_data')
    monkeypatch.setattr(market, 'save_market_data_task', NS(delay=save))
    now = entity().timestamp+timedelta(hours=1)
    client = NS(ensure_connected=AsyncMock(), get_klines=AsyncMock(side_effect=[[kline()], [kline()], []]))
    token = market.analysis_binance_client.set(client)
    try:
        cold = asyncio.run(market._fetch_from_binance_chronological('BTCUSDT', '1h', entity().timestamp, now, 1000, True))
        assert cold[0].taker_buy_volume == 2
        refreshed = asyncio.run(market._fetch_and_save_missing_data('BTCUSDT', '1h', entity().timestamp, now, 1000))
        assert refreshed[0].taker_buy_volume == 2
        assert all(json.loads(call.args[0][0])['taker_buy_volume'] == 2 for call in save.call_args_list)
    finally:
        market.analysis_binance_client.reset(token)


def test_influx_parse_and_formatter_feed_real_delta_against_candle_direction():
    raw = entity(2).model_dump()
    record = NS(get_time=lambda: raw['timestamp'], values=raw)
    recovered = MarketDataEntity(**InfluxDBMarketDataRepository.parse_flux_record(record))
    df = pd.DataFrame(_format_ohlcv_response([recovered]))
    result = CVDEngine('1h').calculate_cvd(df)
    # Green price candle, but negative actual taker delta: NOT a close-price proxy.
    assert result.points[0].delta == -6
    assert result.points[0].delta_source == 'taker_buy_volume'
    assert result.genuine_point_count == 1
    assert indicator_evidence(df, 'bullish', cvd=result, break_index=0)['cvd_break_confirmation']['passed'] is False


def test_influx_writes_real_zero_but_omits_unknown_and_queries_the_field():
    repo = object.__new__(InfluxDBMarketDataRepository)
    repo.bucket = 'test'
    manager = MagicMock()
    writer = manager.__enter__.return_value
    query = Mock(return_value=[])
    repo.client = NS(write_api=lambda: manager, query_api=lambda: NS(query=query))
    asyncio.run(repo.save_market_data_bulk([entity(0), entity(None), entity(2)]))
    points = writer.write.call_args.kwargs['record']
    assert 'taker_buy_volume=0' in points[0].to_line_protocol()
    assert 'taker_buy_volume' not in points[1].to_line_protocol()
    assert 'taker_buy_volume=2' in points[2].to_line_protocol()
    start = entity().timestamp
    asyncio.run(repo.get_historical_data_reverse('BTCUSDT', '1h', start, start+timedelta(hours=2), allow_downsample=False))
    assert 'r._field == "taker_buy_volume"' in query.call_args.args[0]


def test_cumulative_provenance_does_not_hide_missing_history():
    rows = [entity(2), entity(), entity(8)]
    for i, row in enumerate(rows):
        row.timestamp += timedelta(hours=i)
    r = CVDEngine().calculate_cvd(pd.DataFrame(_format_ohlcv_response(rows)))
    assert r.points[-1].delta_source == 'taker_buy_volume'
    assert r.points[-1].cumulative_delta_source == 'mixed'
    assert r.genuine_point_count == 2 and r.approximate_point_count == 1


def test_old_cache_recovers_matching_rows_once_and_persists_closed_flow(monkeypatch):
    save = Mock()
    monkeypatch.setitem(sys.modules, 'core.use_cases.market.market_data', NS(save_market_data_task=NS(delay=save)))
    client = NS(get_klines=AsyncMock(return_value=[kline()]))
    e = entity()
    result = asyncio.run(_recover_taker_buy_volume([e], client, 'BTCUSDT', '1h',
                         e.timestamp+timedelta(hours=1), pd.Timedelta(hours=1)))
    assert result[0].taker_buy_volume == 2
    client.get_klines.assert_awaited_once()
    assert json.loads(save.call_args.args[0][0])['taker_buy_volume'] == 2


def test_mismatched_provider_or_failure_never_invents_flow():
    e = entity()
    row = kline(); row[5] = '11'
    for response in (AsyncMock(return_value=[row]), AsyncMock(side_effect=RuntimeError('offline'))):
        result = asyncio.run(_recover_taker_buy_volume([e], NS(get_klines=response), 'BTCUSDT', '1h',
                             e.timestamp+timedelta(hours=1), pd.Timedelta(hours=1)))
        assert result[0].taker_buy_volume is None


def test_archive_field_nine_opt_in_and_missing_resample_not_zero(tmp_path):
    from tests.backtesting.run_trade_plans import download_month, resample_closed
    path = tmp_path/'BTCUSDT-1h-2026-09.zip'
    rows = [kline(), kline('8')]; rows[1][0] += 3600000
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('candles.csv', '\n'.join(','.join(map(str,r)) for r in rows))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    old, _ = download_month('BTCUSDT', '1h', '2026-09', tmp_path)
    rich, _ = download_month('BTCUSDT', '1h', '2026-09', tmp_path, include_taker_buy=True)
    assert 'taker_buy_volume' not in old
    assert rich.taker_buy_volume.tolist() == [2, 8]
    assert resample_closed(rich, '2h', 2).taker_buy_volume.iloc[0] == 10
    rich.loc[1, 'taker_buy_volume'] = float('nan')
    grouped = resample_closed(rich, '2h', 2)
    assert len(grouped) == 1 and pd.isna(grouped.taker_buy_volume.iloc[0])


@pytest.mark.parametrize('active', [False, True])
def test_default_eligibility_does_not_depend_on_vwap_or_profile(active):
    df, facts = fixture(); facts['mtfa']['enabled'] = active
    outcomes = []
    for v in (0, 1e9, None):
        indicators = {} if v is None else dict(vwap=NS(points=[NS(index=len(df)-1, vwap=v)]),
            volume_profile=NS(profile_available=True, poc_price=v))
        ranked = rank_entry_zones(df, 'bullish', **facts, **indicators)
        assert ranked[0]['eligible']
        outcomes.append([(r['bottom'], r['top'], r['score'], r['groups'], r['eligible']) for r in ranked])
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_live_task_cannot_enable_rejected_indicator_policy_from_environment():
    tree = ast.parse(Path('src/core/services/tasks.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'build_trade_plan']
    assert calls
    for call in calls:
        policy = next(k.value for k in call.keywords if k.arg == 'evidence_policy')
        assert isinstance(policy, ast.Constant) and policy.value == 'smc_v2'


def test_all_live_binance_entity_constructors_preserve_taker_volume():
    for file in ('src/core/services/tasks.py', 'src/core/use_cases/market/market_data.py',
                 'src/core/services/workers/websocket_subscription_manager.py'):
        tree = ast.parse(Path(file).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'MarketDataEntity']
        assert calls
        assert all('taker_buy_volume' in {k.arg for k in c.keywords} for c in calls)


def test_closed_websocket_candle_persists_base_taker_volume(monkeypatch):
    save = Mock()
    monkeypatch.setitem(sys.modules, 'src.core.services.tasks', NS(save_market_data_task=NS(delay=save)))
    module = importlib.import_module('core.services.workers.websocket_subscription_manager')
    monkeypatch.setattr(module, 'save_market_data_task', NS(delay=save))
    monkeypatch.setattr(module, 'redis_cache', NS(get_cached_data=AsyncMock(return_value=None),
                                               set_cached_data=AsyncMock()))
    manager = object.__new__(module.WebsocketSubscriptionManager)
    k = dict(t=kline()[0], T=kline()[6], o='100', h='102', l='99', c='101', v='10', V='2', x=True)
    asyncio.run(manager._cache_candle_data('btcusdt@kline_1h', {'k': k}))
    save.assert_called_once()
    assert json.loads(save.call_args.args[0][0])['taker_buy_volume'] == 2
