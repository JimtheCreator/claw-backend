import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from core.scanner.engine import closed_window, LOOKBACK, normalize_detections, validate_manifest
from core.scanner.market_sessions import MarketSession
from scripts.discover_scanner_universe import manifest_from_exchange


def test_forex_weekend_is_not_a_data_gap_but_missing_open_session_bar_is():
    fx=MarketSession('forex')
    cutoff=int(datetime(2026,9,28,12,tzinfo=timezone.utc).timestamp())
    stamps=fx.expected_opens(cutoff,3600,LOOKBACK)
    rows=[dict(timestamp=datetime.fromtimestamp(s,timezone.utc),open=1,high=2,low=1,close=1.5,volume=20) for s in stamps]
    assert closed_window(rows,'1h',cutoff,session=fx)[0]=='ready'
    assert closed_window(rows,'1h',cutoff)[0]=='gapped'
    rows[100]['timestamp']=rows[99]['timestamp']
    assert closed_window(rows,'1h',cutoff,session=fx)[0]!='ready'


def test_full_discovery_filters_inactive_and_nonspot_without_symbol_sampling():
    entries=[dict(symbol=f'TEST{i}USDT',status='TRADING',isSpotTradingAllowed=True) for i in range(1500)]
    entries += [dict(symbol='HALTEDUSDT',status='BREAK',isSpotTradingAllowed=True),
                dict(symbol='FUTUREUSDT',status='TRADING',isSpotTradingAllowed=False)]
    template=dict(id='binance-spot-pilot',provider='binance',market='spot',detectors=['engulfing'])
    manifest=manifest_from_exchange({'symbols':entries},template)
    assert len(manifest['symbols'])==1500
    assert manifest['events_enabled'] is False
    validate_manifest(manifest)


def test_unicode_exchange_identity_is_preserved_and_query_syntax_rejected():
    import pytest
    from core.scanner.watches import WatchCreate
    from infrastructure.database.questdb.candles import checked_identity
    spec=dict(id='binance-spot-pilot',provider='binance',market='spot',detectors=['engulfing'])
    for symbol in ['币安人生USDT', '牛来USDC']:
        assert validate_manifest(dict(spec,symbols=[symbol]))['symbols']==[symbol]
        assert WatchCreate(pattern_id='engulfing_bullish',symbols=[symbol]).symbols==[symbol]
        checked_identity(symbol)
    for symbol in ["BTC';DROP", 'BTC\\nUSDT', 'BTC USDT', 'BTC,USDT', 'BTC=USDT', 'BTC/USDT']:
        with pytest.raises(ValueError): validate_manifest(dict(spec,symbols=[symbol]))
        with pytest.raises(ValueError): checked_identity(symbol)


def test_virtual_watch_scope_cannot_be_registered_as_a_real_universe():
    import pytest
    spec = dict(id='all-markets', provider='binance', market='spot',
                symbols=['BTCUSDT'], detectors=['engulfing'])
    with pytest.raises(ValueError, match='reserved'): validate_manifest(spec)
    with pytest.raises(ValueError, match='boolean'):
        validate_manifest(dict(spec, id='real-universe', events_enabled='false'))


def test_full_universe_capacity_is_admitted_only_when_gateway_can_serve_it(monkeypatch):
    import pytest
    from core.scanner.capacity import scanner_stream_budget
    monkeypatch.setenv('SCANNER_STREAM_BUDGET','8000')
    monkeypatch.setenv('BINANCE_WS_CONNECTIONS','3')
    monkeypatch.setenv('BINANCE_WS_STREAMS_PER_CONNECTION','200')
    with pytest.raises(ValueError,match='gateway capacity'): scanner_stream_budget()
    monkeypatch.setenv('BINANCE_WS_CONNECTIONS','12')
    monkeypatch.setenv('BINANCE_WS_STREAMS_PER_CONNECTION','800')
    assert scanner_stream_budget()==8000
    monkeypatch.setenv('BINANCE_WS_STREAMS_PER_CONNECTION','900')
    with pytest.raises(ValueError): scanner_stream_budget()
