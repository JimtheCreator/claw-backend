"""Replay-safe, provider-qualified OHLCV storage using existing HTTP transport.

HTTP ILP acknowledges ingestion; SQL reads may briefly lag WAL application.
Provider and market remain part of every key, including deduplication.
"""
import asyncio
import math
import os
import re
import ssl
from functools import lru_cache
from datetime import datetime, timezone
import httpx
import certifi


@lru_cache(maxsize=4)
def _trusted_context(cert_file, cert_dir):
    # Thousands of short async tasks use separate event loops. Share only the
    # immutable trust configuration, not loop-owned sockets or HTTP clients.
    if cert_file:
        return ssl.create_default_context(cafile=cert_file)
    if cert_dir:
        return ssl.create_default_context(capath=cert_dir)
    return ssl.create_default_context(cafile=certifi.where())

TABLE = "watchers_candles"
DDL = f"""CREATE TABLE IF NOT EXISTS {TABLE} (
 timestamp TIMESTAMP, provider SYMBOL, market SYMBOL, symbol SYMBOL INDEX,
 interval SYMBOL, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
 volume DOUBLE, taker_buy_volume DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(timestamp, provider, market, symbol, interval)"""
IDENTIFIER = re.compile(r"^[\w-]{1,40}$")
INTERVAL_SECONDS = {'1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                    '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
                    '8h': 28800, '12h': 43200, '1d': 86400}


def checked_identity(*parts):
    if not all(isinstance(part, str) and IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError("Invalid candle identity")


def epoch_us(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Candle timestamps must be timezone-aware")
    delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def encode_row(provider, market, symbol, interval, row):
    checked_identity(provider, market, symbol, interval)
    values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
    o, h, l, c, v = values.values()
    if not all(math.isfinite(n) for n in values.values()) or min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c):
        raise ValueError("Invalid OHLCV")
    if row.get("taker_buy_volume") is not None:
        taker = float(row["taker_buy_volume"])
        if not math.isfinite(taker) or not 0 <= taker <= v:
            raise ValueError("Invalid taker volume")
        values["taker_buy_volume"] = taker
    fields = ",".join(f"{k}={v}" for k, v in values.items())
    return f"{TABLE},provider={provider},market={market},symbol={symbol},interval={interval} {fields} {epoch_us(row['timestamp']) * 1000}"


class QuestCandles:
    finalized_only = True
    def __init__(self, provider="binance", market="spot", *, url=None, client=None):
        checked_identity(provider, market)
        self.provider, self.market = provider, market
        self.url = (url or os.environ.get("QUESTDB_HTTP_URL", "http://127.0.0.1:9000")).rstrip("/")
        self.client = client

    async def request(self, method, path, **kwargs):
        if self.client is not None:
            response = await self.client.request(method, self.url + path, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=30, verify=_trusted_context(
                    os.environ.get('SSL_CERT_FILE'), os.environ.get('SSL_CERT_DIR'))) as client:
                response = await client.request(method, self.url + path, **kwargs)
        response.raise_for_status()
        return response

    async def query(self, sql):
        response = await self.request("GET", "/exec", params={"query": sql})
        body = response.json()
        if "error" in body:
            raise RuntimeError("QuestDB rejected the query")
        names = [c["name"] for c in body.get("columns", [])]
        return [dict(zip(names, row)) for row in body.get("dataset", [])]

    async def initialize(self):
        await self.query(DDL)

    async def wait_applied(self, *tables, timeout=10):
        """Fence acknowledged writes before dependent reads or event dispatch.

        Each server-side waiter captures a fixed sequencer transaction, so
        unrelated incoming writes cannot move the goalpost. A suspended table,
        missing table or deadline is a failed write operation for our callers;
        they retain/replay the original idempotent work.
        """
        if (not tables or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', name)
                              for name in tables)
                or not math.isfinite(timeout) or not 0 < timeout <= 30):
            raise ValueError('Invalid QuestDB visibility fence')

        async def wait_all():
            for table in dict.fromkeys(tables):
                result = await self.query(f"SELECT wait_wal_table('{table}') applied")
                if result != [{'applied': True}]:
                    raise RuntimeError('QuestDB did not confirm write visibility')

        await asyncio.wait_for(wait_all(), timeout=timeout)

    async def save(self, symbol, interval, rows, cutoff=None):
        await self.save_many([(symbol, interval, row) for row in rows], cutoff)

    async def save_many(self, records, cutoff=None):
        """One acknowledged ILP batch for many symbols; replay uses the same keys."""
        import time
        cutoff = time.time() if cutoff is None else cutoff
        if not math.isfinite(cutoff):
            raise ValueError('Invalid candle cutoff')
        rows = records
        if not rows:
            return
        if len(rows) > 10000:
            raise ValueError("Candle writes must be chunked to at most 10000 rows")
        encoded = []
        for symbol, interval, row in rows:
            if interval not in INTERVAL_SECONDS:
                raise ValueError('Unsupported candle interval')
            opened = epoch_us(row['timestamp'])
            duration = INTERVAL_SECONDS[interval] * 1000000
            if opened % duration or opened + duration > cutoff * 1000000:
                raise ValueError('Cannot store an unaligned or unfinished candle')
            encoded.append(encode_row(self.provider, self.market, symbol, interval, row))
        payload = '\n'.join(encoded) + '\n'
        await self.request("POST", "/write", content=payload.encode(),
                           headers={"Content-Type": "text/plain; charset=utf-8"})
        await self.wait_applied(TABLE)

    async def load(self, symbol, interval, cutoff, limit):
        checked_identity(symbol, interval)
        if type(limit) is not int or not 1 <= limit <= 10000 or not math.isfinite(cutoff):
            raise ValueError("Invalid candle read bounds")
        return await self.query(f"SELECT timestamp, open, high, low, close, volume FROM {TABLE} "
            f"WHERE provider='{self.provider}' AND market='{self.market}' "
            f"AND symbol='{symbol}' AND interval='{interval}' "
            f"AND timestamp < {int(cutoff * 1000000)} ORDER BY timestamp DESC LIMIT {limit}")
