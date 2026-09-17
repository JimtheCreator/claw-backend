"""Strict read-only adapter for the existing Binance candle measurement.

Unlike the chart repository, query failures propagate and no downsampling,
backfill, provider fallback, or missing-bar synthesis is permitted here.
"""
import asyncio
import json
import math
from datetime import datetime, timezone

from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import SYMBOL, LOOKBACK, timestamp_seconds, utc_iso


class StoredBinanceCandles:
    def __init__(self, repository, *, finalized_only=False):
        self.repository = repository
        self.finalized_only = finalized_only

    async def load(self, symbol, interval, cutoff, limit):
        if not SYMBOL.fullmatch(symbol) or interval not in INTERVAL_SECONDS:
            raise ValueError("Invalid candle identity")
        if not 1 <= limit <= LOOKBACK + 1:
            raise ValueError("Invalid candle limit")
        step = INTERVAL_SECONDS[interval]
        start = datetime.fromtimestamp(cutoff - step * (limit + 10), timezone.utc)
        end = datetime.fromtimestamp(cutoff, timezone.utc)
        measurement = "scanner_candles_v1" if self.finalized_only else "market_data"
        identity_filter = '|> filter(fn: (r) => r.provider == "binance" and r.market == "spot")' if self.finalized_only else ""
        query = f'''
from(bucket: {json.dumps(self.repository.bucket)})
|> range(start: {start.isoformat()}, stop: {end.isoformat()})
|> filter(fn: (r) => r._measurement == "{measurement}")
{identity_filter}
|> filter(fn: (r) => r.symbol == {json.dumps(symbol)} and r.interval == {json.dumps(interval)})
|> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
|> group(columns: [])
|> sort(columns: ["_time"], desc: true)
|> limit(n: {limit})
'''
        tables = await asyncio.to_thread(self.repository.client.query_api().query, query)
        return [dict(timestamp=record.get_time(), **{
            key: record.values[key] for key in ("open", "high", "low", "close", "volume")
        }) for table in tables for record in table.records]


def finalized_row(open_ms, close_ms, prices, interval, cutoff):
    step = INTERVAL_SECONDS[interval]
    if type(open_ms) is not int or type(close_ms) is not int:
        raise ValueError("Invalid candle time")
    if open_ms % (step * 1000) or close_ms != open_ms + step * 1000 - 1 or close_ms >= cutoff * 1000:
        raise ValueError("Candle is not a finalized UTC interval")
    values = dict(zip(("open", "high", "low", "close", "volume"), map(float, prices), strict=True))
    o, h, l, c, v = values.values()
    if not all(math.isfinite(x) for x in values.values()) or min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c):
        raise ValueError("Invalid finalized OHLCV")
    return dict(timestamp=utc_iso(open_ms / 1000), **values)


class FinalizedBinanceCandles(StoredBinanceCandles):
    def __init__(self, repository):
        super().__init__(repository, finalized_only=True)

    async def save(self, symbol, interval, rows, cutoff):
        from influxdb_client import Point, WritePrecision
        from influxdb_client.client.write_api import SYNCHRONOUS
        if not SYMBOL.fullmatch(symbol):
            raise ValueError("Invalid symbol")
        points = []
        for row in rows:
            open_ms = int(timestamp_seconds(row["timestamp"]) * 1000)
            checked = finalized_row(open_ms, open_ms + INTERVAL_SECONDS[interval] * 1000 - 1,
                                    [row[k] for k in ("open", "high", "low", "close", "volume")], interval, cutoff)
            point = (Point("scanner_candles_v1").tag("provider", "binance")
                     .tag("market", "spot").tag("symbol", symbol).tag("interval", interval)
                     .time(open_ms, WritePrecision.MS))
            for field in ("open", "high", "low", "close", "volume"):
                point.field(field, checked[field])
            points.append(point)
        if not points:
            return
        def write():
            with self.repository.client.write_api(write_options=SYNCHRONOUS) as writer:
                writer.write(bucket=self.repository.bucket, org=self.repository.org, record=points)
        # Propagate write errors; success must mean the durable store accepted it.
        await asyncio.to_thread(write)
