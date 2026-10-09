"""Opt-in chart/analyzer storage migration; default behavior stays on Influx.

Separate from SCANNER_CANDLE_STORE: market_data contains provisional candles.
Display samples and exact analyzer reads use separate, parity-tested queries.
A full retirement must qualify live mirroring and deletion semantics first.
"""
import logging
import os
from .questdb.market_db import QuestMarketData
from core.domain.entities.MarketDataEntity import MarketDataEntity
from .market_mirror_journal import MarketMirrorJournal, journal_root

log = logging.getLogger(__name__)


class QuestOnlyMarketData:
    """Preserve synchronous caller cleanup without constructing an Influx client."""
    def __init__(self):
        self.quest = QuestMarketData('binance', 'spot')
        self.client = self

    def close(self):
        # QuestCandles owns and closes an HTTP client per operation.
        pass

    def __getattr__(self, name):
        return getattr(self.quest, name)


def require_legacy_deletion():
    # A one-sided delete during mirroring would resurrect data after cutover.
    if os.getenv('MARKET_CANDLE_STORE', 'influx') != 'influx':
        raise RuntimeError('Market history deletion is unavailable during storage migration')


def canonical(value):
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    return value


class MarketDataRollout:
    def __init__(self, legacy, quest, mode):
        if mode not in {'dual','shadow','quest'}:
            raise ValueError('Invalid MARKET_CANDLE_STORE mode')
        self.legacy, self.quest, self.mode = legacy, quest, mode
        self.client = legacy.client  # Preserve existing caller cleanup contract.

    async def save_market_data_bulk(self, rows):
        await MarketMirrorJournal(self.legacy, self.quest).save(rows)

    async def read(self, method, *args, **kwargs):
        if self.mode == 'quest':
            return await getattr(self.quest, method)(*args, **kwargs)
        value = await getattr(self.legacy, method)(*args, **kwargs)
        if self.mode == 'shadow':
            try:
                other = await getattr(self.quest, method)(*args, **kwargs)
                if canonical(value) != canonical(other):
                    log.warning('Market history shadow mismatch (%s)', method)
            except Exception:
                log.warning('Market history shadow unavailable (%s)', method)
        return value

    async def get_historical_data(self, *args, **kwargs):
        return await self.read('get_historical_data', *args, **kwargs)

    async def get_historical_data_reverse(self, *args, **kwargs):
        return await self.read('get_historical_data_reverse', *args, **kwargs)

    async def get_all_symbols_for_interval(self, *args, **kwargs):
        return await self.read('get_all_symbols_for_interval', *args, **kwargs)

    async def get_min_timestamp(self, *args, **kwargs):
        return await self.read('get_min_timestamp', *args, **kwargs)

    async def get_last_update_timestamp(self, *args, **kwargs):
        return await self.read('get_last_update_timestamp', *args, **kwargs)

    async def get_all_timestamps_for_symbol(self, *args, **kwargs):
        return await self.read('get_all_timestamps_for_symbol', *args, **kwargs)


def market_data_store(legacy_factory, **kwargs):
    mode = os.getenv('MARKET_CANDLE_STORE', 'influx')
    if mode == 'quest_only':
        return QuestOnlyMarketData()
    if mode not in {'influx','dual','shadow','quest'}:
        raise ValueError('Invalid MARKET_CANDLE_STORE mode')
    if mode != 'influx':
        journal_root()  # Fail before opening a client if ordering is unconfigured.
    legacy = legacy_factory(**kwargs)
    if mode == 'influx':
        return legacy
    return MarketDataRollout(legacy, QuestMarketData('binance','spot'), mode)


async def persist_market_batch(data_list_json, legacy_factory):
    """Validate before opening storage; release the client on every write outcome."""
    rows = [MarketDataEntity.model_validate_json(item) for item in data_list_json]
    if not rows:
        return 0
    repo = market_data_store(legacy_factory)
    try:
        await repo.save_market_data_bulk(rows)
    finally:
        repo.client.close()
    return len(rows)
