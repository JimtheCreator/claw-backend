import httpx
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from infrastructure.database.redis.rate_limiter import RedisRateLimiter
from common.logger import logger


def _to_polygon_fx_ticker(symbol: str) -> str:
    """'EURUSD' or 'C:EURUSD' -> 'C:EURUSD'"""
    plain = symbol.upper().replace("C:", "")
    return f"C:{plain}"


def _from_polygon_fx_ticker(ticker: str) -> str:
    """'C:EURUSD' -> 'EURUSD'"""
    return ticker.upper().replace("C:", "")


class MassiveClient:
    def __init__(self):
        self.api_key = os.getenv("MASSIVE_API_KEY")
        self.base_url = os.getenv("MASSIVE_API_URL")

        self.rate_limiter = RedisRateLimiter(
            max_per_minute=5,
            max_per_second=2,
            key_prefix="massive_rl",
            max_wait_seconds=90.0,  # forex ticker + sparkline services share this
            # 5/min budget, so a caller may need to wait longer than the
            # default 30s for its turn instead of giving up and firing an
            # unbudgeted request that risks a 429.
        )

    async def get_forex_pairs(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Fetch the full fx ticker universe, following `next_url` pagination.
        Confirmed against the real API: a single unpaginated call only
        returns the first 1000 results (alphabetical), which silently
        drops anything sorting after that -- e.g. XAUUSD. Each additional
        page is its own rate-limited request."""
        url = f"{self.base_url}/v3/reference/tickers"
        params: Optional[Dict[str, Any]] = {
            "market": "fx", "active": "true", "limit": 1000, "apiKey": self.api_key
        }

        all_results: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                next_url = url
                for page in range(max_pages):
                    await self.rate_limiter.acquire(weight=1)
                    response = await client.get(next_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    all_results.extend(data.get("results", []))

                    next_url = data.get("next_url")
                    if not next_url:
                        break
                    # Polygon-style pagination strips the API key from
                    # next_url for security; it must be re-attached, and
                    # `params` must be cleared since next_url already
                    # encodes the cursor + original query params.
                    params = {"apiKey": self.api_key}
                else:
                    logger.warning(
                        f"Massive fx pagination hit max_pages={max_pages} "
                        f"with more pages remaining (next_url still present) -- "
                        f"results may be incomplete."
                    )
            return all_results
        except Exception as e:
            logger.warning(
                f"Massive API unavailable ({type(e).__name__}: {e}) after "
                f"{len(all_results)} page(s) fetched. Returning what was "
                f"retrieved so far rather than fabricating fallback data."
            )
            return all_results

    async def search_forex_pairs(self, query: str) -> List[Dict[str, Any]]:
        """Targeted lookup for a single query, used by the on-demand
        live-search fallback. Far cheaper than paging through the entire fx
        universe just to check one symbol, and finds tickers regardless of
        where they'd sort in the unfiltered list -- confirmed directly
        against the real API (search=XAUUSD returns it even though it's
        outside the first 1000 unfiltered results)."""
        await self.rate_limiter.acquire(weight=1)

        url = f"{self.base_url}/v3/reference/tickers"
        params = {"market": "fx", "active": "true", "search": query, "apiKey": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception as e:
            logger.warning(f"Massive search API unavailable ({type(e).__name__}: {e}) for query '{query}'.")
            return []

    # ------------------------------------------------------------------
    # NEW: live pricing + OHLC, mirroring BinanceMarketData's
    # get_all_tickers() and get_klines() so the ticker/sparkline services
    # can be near drop-in copies of the Binance ones.
    # ------------------------------------------------------------------

    async def get_forex_snapshot(self) -> List[Dict[str, Any]]:
        """
        NOT USED by forex_ticker_service - confirmed 403 ("NOT_AUTHORIZED
        ... upgrade your plan") on a Starter-tier key via
        scripts/test_massive_forex_access.py. Real-time snapshot data is a
        paid add-on on Polygon (which Massive wraps). Left here in case
        the plan is upgraded later - get_forex_grouped_daily() below is
        the current plan-accessible substitute.
        """
        await self.rate_limiter.acquire(weight=1)

        url = f"{self.base_url}/v2/snapshot/locale/global/markets/forex/tickers"
        params = {"apiKey": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get("tickers", []) or []
        except Exception as e:
            logger.warning(
                f"Massive forex snapshot unavailable ({type(e).__name__}: {e}). "
                f"Returning empty list rather than fabricating prices."
            )
            return []

    async def get_forex_grouped_daily(
        self,
        date: Optional[str] = None,
        max_lookback_days: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Bulk end-of-day OHLC for EVERY forex pair in ONE call - the
        plan-accessible substitute for get_forex_snapshot(), which 403s.
        Confirmed working directly against the account (200, 1203
        results) via scripts/test_massive_forex_access.py.

        Not truly real-time - Polygon publishes grouped daily bars once
        a trading day closes, not intraday - but it's still a single
        request covering every fx pair, giving a usable price + day
        change without a paid real-time add-on. Walks backward from
        `date` (default: today, UTC) up to `max_lookback_days` to land
        on the most recent day with published data (skips weekends and
        the current day before it's closed).
        """
        target = (
            datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if date else datetime.now(timezone.utc)
        )

        for _ in range(max_lookback_days):
            date_str = target.strftime("%Y-%m-%d")
            await self.rate_limiter.acquire(weight=1)

            url = f"{self.base_url}/v2/aggs/grouped/locale/global/market/fx/{date_str}"
            params = {"adjusted": "true", "apiKey": self.api_key}

            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(url, params=params)
                    if response.status_code == 404:
                        target -= timedelta(days=1)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", []) or []
                    if results:
                        return results
            except Exception as e:
                logger.warning(
                    f"Massive grouped daily fx unavailable for {date_str} "
                    f"({type(e).__name__}: {e})."
                )

            target -= timedelta(days=1)

        logger.warning(
            f"Massive grouped daily fx: no data found after walking back "
            f"{max_lookback_days} day(s) from {date or 'today'}."
        )
        return []

    async def get_forex_aggregates(
        self,
        symbol: str,
        multiplier: int = 15,
        timespan: str = "minute",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 96,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLC bars for ONE forex pair - the sparkline equivalent of
        BinanceMarketData.get_klines(). `symbol` is the plain pair as
        stored in the watchlist (e.g. "EURUSD"); this converts it to the
        "C:EURUSD" form Polygon's aggs endpoint expects.

        Defaults to 15-minute bars over the last 2 days, matching the
        crypto sparkline service's "15m interval, up to 96 points (~24h)"
        shape so the frontend sparkline component doesn't need to branch
        on asset type.
        """
        ticker = _to_polygon_fx_ticker(symbol)

        if not to_date:
            to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not from_date:
            from_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")

        await self.rate_limiter.acquire(weight=1)

        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get("results", []) or []
        except Exception as e:
            logger.warning(
                f"Massive forex aggregates unavailable for {symbol} "
                f"({type(e).__name__}: {e})."
            )
            return []