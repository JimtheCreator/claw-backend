import asyncio
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from core.use_cases.market_analysis.momentum_history import MomentumCache
from infrastructure.database.momentum_rollout import MomentumRollout, momentum_store, same_frame, mirror_lock
from infrastructure.database.questdb.momentum import QuestMomentum


def frame(taker=3,close=101):
    return pd.DataFrame([[pd.Timestamp('2026-09-01',tz='UTC'),100.,102.,99.,float(close),10.,taker]],
                        columns=['timestamp','open','high','low','close','volume','taker_buy_volume'])


def test_mirror_uses_committed_taker_rules_and_propagates_storage_failure(tmp_path):
    legacy=MomentumCache(tmp_path/'cache.sqlite');target=Mock()
    store=MomentumRollout(legacy,target,'dual')
    store.put('BTCUSDT','1h',frame())
    store.put('BTCUSDT','1h',frame(np.nan))
    assert target.put.call_args.args[2].taker_buy_volume.iloc[0]==3
    store.put('BTCUSDT','1h',frame(np.nan,close=100))
    assert pd.isna(target.put.call_args.args[2].taker_buy_volume.iloc[0])
    store.put('BTCUSDT','1h',frame(0,close=100))
    assert target.put.call_args.args[2].taker_buy_volume.iloc[0]==0
    target.put.side_effect=ConnectionError()
    with pytest.raises(ConnectionError):store.put('BTCUSDT','1h',frame(2))
    assert legacy.get('BTCUSDT','1h',frame().timestamp.iloc[0],1).taker_buy_volume.iloc[0]==2


def test_shadow_keeps_primary_even_if_target_missing_or_down(tmp_path,caplog):
    legacy=MomentumCache(tmp_path/'cache.sqlite');legacy.put('BTCUSDT','1h',frame())
    target=Mock();target.get.return_value=frame(0)
    store=MomentumRollout(legacy,target,'shadow')
    assert same_frame(store.get('BTCUSDT','1h',frame().timestamp.iloc[0],1),frame())
    assert 'mismatch' in caplog.text
    target.get.side_effect=ConnectionError()
    assert same_frame(store.get('BTCUSDT','1h',frame().timestamp.iloc[0],1),frame())
    assert 'unavailable' in caplog.text
    with pytest.raises(ConnectionError):MomentumRollout(legacy,target,'quest').get('BTCUSDT','1h',frame().timestamp.iloc[0],1)


def test_default_has_no_extra_store_and_mirror_lock_is_bounded(monkeypatch,tmp_path):
    monkeypatch.delenv('MOMENTUM_CANDLE_STORE',raising=False)
    factory=Mock(return_value=object())
    assert momentum_store(factory) is factory.return_value
    monkeypatch.setenv('MOMENTUM_CANDLE_STORE','bad');factory.reset_mock()
    with pytest.raises(ValueError):momentum_store(factory)
    factory.assert_not_called()
    path=tmp_path/'cache.sqlite'
    with mirror_lock(path):
        with pytest.raises(TimeoutError):
            with mirror_lock(path,timeout=0):pass


def test_quest_rejects_unbounded_or_unqualified_reads_before_io():
    store=QuestMomentum()
    for count in (0,400001,True):
        with pytest.raises(ValueError):asyncio.run(store.load_frame('BTCUSDT','1h',frame().timestamp.iloc[0],count))
    with pytest.raises(ValueError):asyncio.run(store.load_frame("BTC'",'1h',frame().timestamp.iloc[0],1))


def test_migration_page_advances_only_after_parity(tmp_path):
    from scripts.migrate_momentum_history import copy_page
    import sqlite3
    legacy=MomentumCache(tmp_path/'source.sqlite');legacy.put('BTCUSDT','1h',frame())
    db=sqlite3.connect(legacy.path)
    target=Mock();target.get.return_value=frame(0)
    end=int(frame().timestamp.iloc[0].value)
    try:
        with pytest.raises(RuntimeError,match='parity'):
            copy_page(db,target,'BTCUSDT','1h',-1,end,timeout=0)
        target.get.return_value=frame()
        assert same_frame(copy_page(db,target,'BTCUSDT','1h',-1,end),frame())
    finally:db.close()
