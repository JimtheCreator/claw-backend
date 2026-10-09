"""Reversible scanner storage rollout. Dual-write retries are idempotent.

Modes: influx, dual (Influx reads), shadow (Influx reads + comparison), quest
(Quest reads while continuing both writes for rollback). No destructive switch.
"""
import logging
import os
from .questdb.candles import QuestCandles, epoch_us
from core.scanner.catalog import INTERVAL_SECONDS

log = logging.getLogger(__name__)


def canonical(rows):
    return sorted((epoch_us(r["timestamp"]), *(float(r[k]) for k in
                  ("open", "high", "low", "close", "volume"))) for r in rows)


def legacy_window(rows, interval, cutoff, limit):
    """Match Influx's bounded finalized-window read, including sparse histories."""
    start = int((cutoff - INTERVAL_SECONDS[interval] * (limit + 10)) * 1000000)
    end = int(cutoff * 1000000)
    return [row for row in rows if start <= epoch_us(row['timestamp']) < end]


class CandleRollout:
    finalized_only = True
    def __init__(self, legacy, quest, mode):
        if mode not in {"influx", "dual", "shadow", "quest"}:
            raise ValueError("Unknown SCANNER_CANDLE_STORE mode")
        self.legacy, self.quest, self.mode = legacy, quest, mode

    async def save(self, symbol, interval, rows, cutoff):
        await self.legacy.save(symbol, interval, rows, cutoff)
        if self.mode != "influx":
            await self.quest.save(symbol, interval, rows, cutoff)

    async def load(self, symbol, interval, cutoff, limit):
        if self.mode == "quest":
            return legacy_window(await self.quest.load(symbol, interval, cutoff, limit), interval, cutoff, limit)
        rows = await self.legacy.load(symbol, interval, cutoff, limit)
        if self.mode == "shadow":
            try:
                other = legacy_window(await self.quest.load(symbol, interval, cutoff, limit), interval, cutoff, limit)
                if canonical(rows) != canonical(other):
                    log.warning("Candle shadow mismatch: %s %s cutoff=%s", symbol, interval, cutoff)
            except Exception:
                log.warning("Candle shadow read unavailable: %s %s", symbol, interval)
        return rows


def scanner_candles(repository, manifest=None):
    if manifest and manifest['provider'] == 'massive':
        from .questdb.aggregate_candles import MassiveScannerCandles
        return MassiveScannerCandles(manifest['market'])
    if os.getenv('SCANNER_CANDLE_STORE') == 'quest_only':
        return QuestCandles()
    from .influxdb.scanner_candles import FinalizedBinanceCandles
    return CandleRollout(FinalizedBinanceCandles(repository), QuestCandles(),
                         os.getenv("SCANNER_CANDLE_STORE", "influx"))


def scanner_repository(**kwargs):
    if os.getenv('SCANNER_CANDLE_STORE') == 'quest_only':
        return None
    from .influxdb.market_db import InfluxDBMarketDataRepository
    return InfluxDBMarketDataRepository(**kwargs)
