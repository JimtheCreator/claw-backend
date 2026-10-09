"""Bounded Massive minute-aggregate stream, with serialized authentication.

One elected consumer per asset cluster. The caller owns a Redis lease; losing
it cancels the socket, even if the provider is quiet. No REST fallback here.
"""
import asyncio
import json
import math
import re
from dataclasses import dataclass
from .policy import websocket_budget


class MassiveAuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForexQuote:
    base: str
    quote: str
    timestamp_ms: int
    bid: float
    ask: float

    @property
    def symbol(self):
        return self.base + self.quote

    def payload(self):
        return dict(provider='massive', market='forex', symbol=self.symbol,
                    base_currency=self.base, quote_currency=self.quote,
                    timestamp_ms=self.timestamp_ms, bid=self.bid, ask=self.ask,
                    price=self.bid / 2 + self.ask / 2, price_basis='mid_quote')


def parse_quote(event):
    if event.get('ev') != 'C':
        return None
    pair = re.fullmatch(r'([A-Z0-9]{1,15})[-/]([A-Z0-9]{1,15})', event.get('p', ''))
    stamp = event.get('t')
    if isinstance(event.get('b'), bool) or isinstance(event.get('a'), bool):
        raise ValueError('Invalid Massive quote')
    bid, ask = float(event['b']), float(event['a'])
    if (not pair or type(stamp) is not int or stamp <= 0
            or not all(math.isfinite(v) and v > 0 for v in (bid, ask)) or bid > ask):
        raise ValueError('Invalid Massive quote')
    return ForexQuote(*pair.groups(), stamp, bid, ask)


@dataclass(frozen=True)
class MinuteBar:
    market: str
    base: str
    quote: str
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def symbol(self):
        return self.base + self.quote

    @property
    def instrument_id(self):
        return f"massive:{self.market}:{self.symbol}"

    def row(self):
        from datetime import datetime, timezone
        return {"timestamp": datetime.fromtimestamp(self.timestamp_ms / 1000, timezone.utc).isoformat(),
                **{k: getattr(self, k) for k in ("open", "high", "low", "close", "volume")}}


def parse_bar(event, cluster):
    expected = "CA" if cluster == "forex" else "XA"
    if event.get("ev") != expected:
        return None
    pair = event.get("pair", "")
    match = re.fullmatch(r"([A-Z0-9]{1,15})[-/]([A-Z0-9]{1,15})", pair)
    if not match:
        raise ValueError("Invalid Massive instrument")
    stamp = event.get("s")
    if type(stamp) is not int or stamp < 0 or stamp % 60000:
        raise ValueError("Invalid minute timestamp")
    values = [float(event[k]) for k in ("o", "h", "l", "c", "v")]
    o, h, l, c, v = values
    if not all(math.isfinite(n) for n in values) or min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c):
        raise ValueError("Invalid Massive OHLCV")
    return MinuteBar(cluster, *match.groups(), stamp, *values)


class MassiveStream:
    def __init__(self, redis, api_key, cluster="forex", *, connect=None):
        if cluster not in {"forex", "crypto"}:
            raise ValueError("Unsupported Massive cluster")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY is required")
        if connect is None:
            from websockets.asyncio.client import connect
        self.connect, self.api_key, self.cluster = connect, api_key, cluster
        self.budget = websocket_budget(redis)

    @staticmethod
    def events(raw):
        events = json.loads(raw)
        if not isinstance(events, list) or len(events) > 20000:
            raise ValueError("Invalid Massive frame")
        return events

    async def wait_status(self, socket, expected):
        async with asyncio.timeout(15):
            async for raw in socket:
                for event in self.events(raw):
                    status = event.get("status")
                    if status in {"auth_failed", "not_authorized", "max_connections"}:
                        raise MassiveAuthenticationError("Massive stream authorization or connection entitlement refused")
                    if status == expected:
                        return
        raise ConnectionError("Massive disconnected during handshake")

    async def run_once(self, on_bar, on_connected=None, on_quote=None, on_quotes=None):
        await self.budget.acquire()
        async with self.connect(f"wss://socket.massive.com/{self.cluster}",
                                open_timeout=15, close_timeout=5,
                                max_size=4 * 1024 * 1024, max_queue=16,
                                ping_interval=20, ping_timeout=20) as socket:
            await self.wait_status(socket, "connected")
            await self.budget.acquire()
            await socket.send(json.dumps({"action": "auth", "params": self.api_key}))
            await self.wait_status(socket, "auth_success")
            await self.budget.acquire()
            channel = "CA.*" if self.cluster == "forex" else "XA.*"
            if self.cluster == 'forex' and (on_quote is not None or on_quotes is not None):
                channel += ',C.*'
            await socket.send(json.dumps({"action": "subscribe", "params": channel}))
            if on_connected:
                await on_connected()
            async for raw in socket:
                quote_batch = []
                for event in self.events(raw):
                    if event.get("ev") == "status":
                        if event.get("status") in {"auth_failed", "not_authorized", "max_connections", "error"}:
                            raise MassiveAuthenticationError("Massive stream subscription refused")
                        continue
                    if event.get('ev') == 'C' and self.cluster == 'forex' and (on_quote is not None or on_quotes is not None):
                        try:
                            quote = parse_quote(event)
                        except (KeyError, ValueError, TypeError, OverflowError):
                            continue  # A malformed display quote cannot stop durable candle ingestion.
                        if on_quotes is not None:
                            quote_batch.append(quote)
                            if len(quote_batch) == 500:
                                await on_quotes(quote_batch)
                                quote_batch = []
                        else:
                            await on_quote(quote)
                        continue
                    bar = parse_bar(event, self.cluster)
                    if bar is not None:
                        await on_bar(bar)
                if quote_batch:
                    await on_quotes(quote_batch)
            raise ConnectionError("Massive stream ended")
