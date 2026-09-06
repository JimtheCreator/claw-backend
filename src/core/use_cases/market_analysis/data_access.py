# src/core/use_cases/market_analysis/data_access.py
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from fastapi import BackgroundTasks
from core.domain.entities.MarketDataEntity import MarketDataEntity
import re

async def get_ohlcv_from_db(
    symbol: str,
    interval: str,
    timeframe: str,
    background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, list]:
    """
    Gets OHLCV data using the existing paginated fetcher with:
    - Automatic stale data detection
    - Background updates
    - Abuse protections
    Returns formatted: {
        'open': [list],
        'high': [list],
        'low': [list],
        'close': [list],
        'volume': [list],
        'timestamp': [list]
    }
    """
    from core.use_cases.market.market_data import fetch_crypto_data_paginated, analysis_binance_client
    from infrastructure.data_sources.binance.client import BinanceMarketData
    client = BinanceMarketData(use_pool=False, strict_errors=True)
    token = analysis_binance_client.set(client)
    try:
        # Convert timeframe to start/end parameters
        start_time = _parse_timeframe(timeframe)
        end_time = datetime.now(timezone.utc)

        # Use the existing paginated fetcher
        data = await fetch_crypto_data_paginated(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=1000
        )

        # logger.info(f"Formatted OHLCV data: {_format_ohlcv_response(data)}")

        if isinstance(data, dict):
            raise DataUnavailableError(f"{symbol} {interval}: market-data recovery failed: {data.get('error', 'invalid provider response')}")
        if not data:
            raise DataUnavailableError(f"No candles available for {symbol} {interval} in {timeframe}, including exchange recovery. Check the symbol and listing history.")
        
        # Ensure the data is in the expected format
        if not isinstance(data[0], MarketDataEntity):
            logger.warning("Data is not of type MarketDataEntity, attempting to convert.")
            try:
                data = [MarketDataEntity(**d) if isinstance(d, dict) else d for d in data]
            except Exception as conversion_error:
                logger.error(f"Data conversion failed: {conversion_error}")
                raise DataUnavailableError("Data format is invalid and cannot be converted.")

        data = sorted({d.timestamp: d for d in data}.values(), key=lambda d: d.timestamp)[-1000:]
        # Do not turn a failed stale-data refresh into a current forecast.
        import pandas as pd
        last_open = pd.Timestamp(data[-1].timestamp)
        last_open = last_open.tz_localize("UTC") if last_open.tzinfo is None else last_open
        count, unit = int(interval[:-1]), interval[-1]
        step = pd.DateOffset(months=count) if unit == "M" else pd.Timedelta(count, unit={"m":"min","h":"h","d":"D","w":"W"}[unit])
        from math import isfinite
        for candle in data:
            values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
            if (not all(isfinite(v) for v in values) or min(values[:4]) <= 0 or candle.volume < 0
                    or candle.high < max(candle.open,candle.close,candle.low)
                    or candle.low > min(candle.open,candle.close,candle.high)):
                raise DataUnavailableError(f"{symbol} {interval}: invalid OHLCV candle; analysis was not generated.")
        times = pd.to_datetime([c.timestamp for c in data], utc=True)
        if any(a + step != b for a,b in zip(times[:-1],times[1:])):
            raise DataUnavailableError(f"{symbol} {interval}: gaps remain after exchange recovery. No candles were invented to bridge them.")
        if last_open + step + step <= pd.Timestamp(end_time):
            raise DataUnavailableError(f"{symbol} {interval}: exchange recovery did not provide current candles. Last candle: {last_open.isoformat()}. No stale forecast was generated.")
        return _format_ohlcv_response(data)

    except DataUnavailableError:
        raise
    except Exception as e:
        logger.error(f"Failed to get OHLCV data: {str(e)}")
        raise DataUnavailableError(f"Could not retrieve {symbol} {interval} market data ({type(e).__name__}).") from e
    finally:
        analysis_binance_client.reset(token)
        try:
            await client.disconnect()
        except Exception:
            logger.warning("Analysis exchange-client cleanup failed.", exc_info=True)

def _parse_timeframe(timeframe: str) -> datetime:
    """Parse custom timeframe strings like '30m', '2d', '1w'"""
    match = re.match(r"^(\d+)([mhdwM])$", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    num, unit = match.groups()
    num = int(num)
    
    unit_map = {
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
        'M': 'days'  # Approximate month as 30 days
    }
    
    delta = timedelta(**{unit_map[unit]: num * (30 if unit == 'M' else 1)})
    return datetime.now(timezone.utc) - delta

def _format_ohlcv_response(data: List[MarketDataEntity]) -> Dict[str, list]:
    """Convert entity list to OHLCV arrays"""
    return {
        'open': [d.open for d in data],
        'high': [d.high for d in data],
        'low': [d.low for d in data],
        'close': [d.close for d in data],
        'volume': [d.volume for d in data],
        'timestamp': [d.timestamp for d in data]
    }
