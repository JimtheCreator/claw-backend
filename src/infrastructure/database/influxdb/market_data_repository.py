# src/infrastructure/database/influxdb/market_data_repository.py
from influxdb_client import InfluxDBClient
from core.interfaces.market_data_repository import MarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from common.logger import logger
from influxdb_client import InfluxDBClient, Point
from typing import Optional, Dict, Any, Tuple
import asyncio
import time
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError


class InfluxDBMarketDataRepository(MarketDataRepository):
    def __init__(self):
        # Improved connection configuration with timeout settings
        self.client = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL"),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG"),
            timeout=120_000  # 2 minutes timeout
        )
        self.bucket = os.getenv("INFLUXDB_BUCKET")
        self._verify_connection()
        
        # Cache to store minimum timestamps per symbol+interval
        self._min_timestamps_cache: Dict[Tuple[str, str], datetime] = {}

    def _verify_connection(self):
        """Verify InfluxDB connection on initialization with retry logic"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                if self.client.ping():
                    logger.info("Successfully connected to InfluxDB")
                    return
                else:
                    retry_count += 1
                    logger.warning(f"InfluxDB connection verification failed (attempt {retry_count}/{max_retries})")
                    time.sleep(1)
            except InfluxDBError as e:
                retry_count += 1
                if (retry_count >= max_retries):
                    logger.critical(f"Critical InfluxDB connection failure: {str(e)}")
                    raise
                logger.warning(f"InfluxDB connection error (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(1)
        
        logger.critical("Critical InfluxDB connection failure after retries")
        raise InfluxDBError("Failed to connect to InfluxDB after multiple attempts")

    @staticmethod
    def parse_flux_record(record) -> Optional[dict]:
        try:
            return {
                'timestamp': record.get_time() or datetime.now(timezone.utc),
                'symbol': record.values.get('symbol', 'UNKNOWN'),
                'interval': record.values.get('interval', '1m'),
                'open': float(record.values.get('open', 0.0)) if record.values.get('open') is not None else 0.0,
                'high': float(record.values.get('high', 0.0)) if record.values.get('high') is not None else 0.0,
                'low': float(record.values.get('low', 0.0)) if record.values.get('low') is not None else 0.0,
                'close': float(record.values.get('close', 0.0)) if record.values.get('close') is not None else 0.0,
                'volume': float(record.values.get('volume', 0.0)) if record.values.get('volume') is not None else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to parse InfluxDB record: {str(e)}")
            return None

    async def get_min_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the earliest available timestamp for a symbol+interval combination"""
        # Check cache first
        cache_key = (symbol, interval)
        if cache_key in self._min_timestamps_cache:
            return self._min_timestamps_cache[cache_key]
        
        # If not in cache, query from database
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -30y)
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.interval == "{interval}")
        |> filter(fn: (r) => r._field == "close")
        |> sort(columns: ["_time"], desc: false)
        |> limit(n: 1)
        '''
        
        try:
            result = self.client.query_api().query(query)
            for table in result:
                for record in table.records:
                    timestamp = record.get_time()
                    if timestamp:
                        # Store in cache
                        self._min_timestamps_cache[cache_key] = timestamp
                        return timestamp
            return None
        except Exception as e:
            logger.error(f"Error getting min timestamp: {str(e)}")
            return None

    async def get_last_update_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the latest timestamp for a symbol and interval."""
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.interval == "{interval}")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: 1)
        '''
        try:
            result = self.client.query_api().query(query)
            if result and result[0].records:
                return result[0].records[0].get_time()
        except Exception as e:
            logger.error(f"Failed to get last update timestamp: {e}")
        return None

    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 500
    ) -> list[MarketDataEntity]:
        """
        Get historical market data with pagination support.
        
        Args:
            symbol: The trading pair symbol
            interval: Timeframe interval (e.g., "1m", "5m", "1h")
            start_time: Optional start time filter
            end_time: Optional end time filter
            page: Page number (default: 1)
            page_size: Number of records per page (default: 500)
            
        Returns:
            List of MarketDataEntity objects
        """
        flux_interval = self._get_flux_interval(interval)
        
        # Set default start_time if not provided
        if not start_time:
            start_time = datetime.now(timezone.utc) - timedelta(days=30)  # Default to 30 days
        
        # Set default end_time if not provided
        if not end_time:
            end_time = datetime.now(timezone.utc)
        
        # Calculate offset for pagination
        offset = (page - 1) * page_size
        
        # Check if downsampling is appropriate based on the date range
        date_range = end_time - start_time
        if self._should_downsample(interval, date_range):
            return await self._get_downsampled_data(symbol, interval, start_time, end_time, page, page_size)
        
        # Regular query with pagination
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.interval == "{interval}")
        |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"], desc: false)
        |> limit(n: {page_size}, offset: {offset})
        '''
        
        try:
            result = self.client.query_api().query(query)
            parsed_records = []
            for table in result:
                for record in table.records:
                    parsed = self.parse_flux_record(record)
                    if parsed:
                        try:
                            parsed_records.append(MarketDataEntity(**parsed))
                        except ValidationError as e:
                            logger.error(f"Invalid MarketDataEntity: {str(e)}")
            
            logger.info(f"Data retrieved from INFLUXDB - Page {page}, {len(parsed_records)} records")
            return parsed_records
        except Exception as e:
            logger.error(f"InfluxDB query error: {str(e)}")
            return []
    
    def _flux_field_filter(fields: list[str]) -> str:
        return " or ".join([f'r._field == "{field}"' for field in fields])

    async def _get_downsampled_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        page: int = 1,
        page_size: int = 500
    ) -> list[MarketDataEntity]:
        """Get downsampled data for chart rendering optimization"""
        target_points = 300
        date_range_seconds = (end_time - start_time).total_seconds()
        window_seconds = max(int(date_range_seconds / target_points), 60)
        window_duration = self._seconds_to_flux_duration(window_seconds)
        offset = (page - 1) * page_size

        # --- FIX ---
        # The previous query used a custom `fn` in `aggregateWindow` that was syntactically
        # incorrect for InfluxDB v2.7+. The correct approach is to group the data,
        # then use `reduce()` to create the OHLCV record for each window. This
        # ensures the output is a stream of records as expected by the subsequent `pivot` function.
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.interval == "{interval}")
        |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
        |> aggregateWindow(every: {window_duration}, fn: first, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> group(columns: ["_time", "symbol", "interval"])
        |> reduce(
            identity: {{
                open: 0.0,
                high: 0.0,
                low: 0.0,
                close: 0.0,
                volume: 0.0,
                time: time(v: 0),
                symbol: "",
                interval: ""
            }},
            fn: (r, accumulator) => ({{
                open: if r._field == "open" then r._value else accumulator.open,
                high: if r._field == "high" then r._value else accumulator.high,
                low: if r._field == "low" then r._value else accumulator.low,
                close: if r._field == "close" then r._value else accumulator.close,
                volume: if r._field == "volume" then r._value else accumulator.volume,
                time: r._time,
                symbol: r.symbol,
                interval: r.interval
            }})
        )
        |> sort(columns: ["time"], desc: false)
        |> limit(n: {page_size}, offset: {offset})
        '''

        try:
            result = self.client.query_api().query(query)
            parsed_records = []
            for table in result:
                for record in table.records:
                    # Adjust parsing for the new query structure
                    parsed = self.parse_flux_record(record)
                    if parsed:
                        try:
                            # The record now directly contains the fields
                            parsed_records.append(MarketDataEntity(**parsed))
                        except ValidationError as e:
                            logger.error(f"Invalid MarketDataEntity: {str(e)}")

            logger.info(f"Retrieved {len(parsed_records)} downsampled records from InfluxDB for {symbol} ({interval})")
            return parsed_records
        except Exception as e:
            logger.error(f"InfluxDB downsampling query error: {str(e)}")
            return []

    def _seconds_to_flux_duration(self, seconds: int) -> str:
        """Convert seconds to a Flux duration string"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h"
        else:
            return f"{seconds // 86400}d"

    def _should_downsample(self, interval: str, date_range: timedelta) -> bool:
        """Determine if downsampling is needed based on interval and date range"""
        # For small intervals with large date ranges, we should downsample
        if interval == "1m" and date_range.days > 7:
            return True
        elif interval == "5m" and date_range.days > 30:
            return True
        elif interval == "15m" and date_range.days > 60:
            return True
        elif interval == "30m" and date_range.days > 90:
            return True
        elif interval == "1h" and date_range.days > 180:
            return True
        elif interval == "4h" and date_range.days > 365:
            return True
        elif interval == "1d" and date_range.days > 730:  # 2 years
            return True
        
        return False

    async def save_market_data_bulk(self, data_list: list[MarketDataEntity]) -> None:
        try:
            with self.client.write_api() as write_api:
                points = [
                    Point("market_data")
                    .tag("symbol", entity.symbol)
                    .tag("interval", entity.interval)
                    .field("open", entity.open)
                    .field("high", entity.high)
                    .field("low", entity.low)
                    .field("close", entity.close)
                    .field("volume", entity.volume)
                    .time(entity.timestamp)
                    for entity in data_list
                ]
                write_api.write(bucket=self.bucket, record=points)
        except Exception as e:
            logger.error(f"InfluxDB bulk write error: {str(e)}")
            
    async def delete_market_data(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        timeout: int = 250
    ) -> Dict[str, Any]:
        """
        Delete market data with improved timeout handling
        
        Args:
            symbol: Optional symbol filter
            interval: Optional interval filter
            start_time: Optional start time for deletion range
            end_time: Optional end time for deletion range
            timeout: Timeout in seconds (default: 120)
            
        Returns:
            Dict[str, Any]: Result of the deletion operation
        """
        # Build predicate components for delete operation
        predicate_parts = ['_measurement="market_data"']
        
        if symbol:
            predicate_parts.append(f'symbol="{symbol}"')
        
        if interval:
            predicate_parts.append(f'interval="{interval}"')
        
        # Join with AND operator
        predicate = ' AND '.join(predicate_parts)
        
        # For large deletions, use chunked approach
        if interval in ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d", "3d", "1w"] and (not start_time or not end_time):
            return await self._chunked_delete(symbol, interval, start_time, end_time, timeout)
        
        # For smaller deletions, use standard approach
        try:
            # Create a client with the specified timeout for this operation
            client_with_timeout = InfluxDBClient(
                url=os.getenv("INFLUXDB_URL"),
                token=os.getenv("INFLUXDB_TOKEN"),
                org=os.getenv("INFLUXDB_ORG"),
                timeout=timeout * 1000  # Convert to milliseconds
            )
            
            logger.info(f"Deleting market data with predicate: {predicate}")
            result = client_with_timeout.delete_api().delete(
                start=start_time if start_time else datetime(1970, 1, 1, tzinfo=timezone.utc),
                stop=end_time if end_time else datetime.now(timezone.utc) + timedelta(days=1),
                predicate=predicate,
                bucket=self.bucket
            )
            
            message = f"Successfully deleted data"
            if symbol:
                message += f" for symbol {symbol}"
            if interval:
                message += f" with interval {interval}"
            if start_time or end_time:
                message += " in specified time range"
                
            logger.info(message)
            client_with_timeout.close()
            return {"status": "success", "message": message}
            
        except Exception as e:
            error_msg = f"Failed to delete market data: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    async def _chunked_delete(
        self, 
        symbol: Optional[str], 
        interval: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        timeout: int
    ) -> Dict[str, Any]:
        """Delete large datasets in chunks to avoid timeouts"""
        if not start_time:
            # Get minimum timestamp or default to 5 years ago
            if symbol and interval:
                min_timestamp = await self.get_min_timestamp(symbol, interval)
                start_time = min_timestamp if min_timestamp else datetime.now(timezone.utc) - timedelta(days=1825)
            else:
                start_time = datetime.now(timezone.utc) - timedelta(days=1825)
        
        if not end_time:
            end_time = datetime.now(timezone.utc)
            
        # For larger intervals, we can use longer chunk periods
        # Chunking is only necessary for smaller intervals with large datasets
        if interval in ["1m", "5m", "15m", "30m", "1h"]:
            chunk_period = timedelta(days=90)  # 3 months chunks for smaller intervals
        elif interval in ["2h", "4h", "6h", "1d", "3d", "6d", "1w"]:
            chunk_period = timedelta(days=365)  # 1 year chunks for larger intervals
        else:
            chunk_period = timedelta(days=730)  # Default to 2 years for unknown intervals
            
        # Create chunks
        current_start = start_time
        success_count = 0
        error_count = 0
        
        while current_start < end_time:
            chunk_end = min(current_start + chunk_period, end_time)
            
            # Build predicate components for this chunk
            predicate_parts = ['_measurement="market_data"']
            
            if symbol:
                predicate_parts.append(f'symbol="{symbol}"')
            
            if interval:
                predicate_parts.append(f'interval="{interval}"')
            
            predicate = ' AND '.join(predicate_parts)
            
            try:
                # Create a client with the specified timeout for this operation
                client_with_timeout = InfluxDBClient(
                    url=os.getenv("INFLUXDB_URL"),
                    token=os.getenv("INFLUXDB_TOKEN"),
                    org=os.getenv("INFLUXDB_ORG"),
                    timeout=timeout * 1000  # Convert to milliseconds
                )
                
                logger.info(f"Deleting chunk from {current_start.isoformat()} to {chunk_end.isoformat()} with predicate: {predicate}")
                
                client_with_timeout.delete_api().delete(
                    start=current_start,
                    stop=chunk_end,
                    predicate=predicate,
                    bucket=self.bucket
                )
                
                success_count += 1
                logger.info(f"Successfully deleted chunk from {current_start.isoformat()} to {chunk_end.isoformat()}")
                client_with_timeout.close()
                
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to delete chunk from {current_start.isoformat()} to {chunk_end.isoformat()}: {str(e)}")
            
            # Move to next chunk
            current_start = chunk_end
            
            # Small delay to avoid overwhelming the database
            await asyncio.sleep(1)
            
        if error_count == 0:
            message = f"Successfully deleted all data"
            if symbol:
                message += f" for symbol {symbol}"
            if interval:
                message += f" with interval {interval}"
                
            return {"status": "success", "message": message}
        elif success_count > 0:
            message = f"Partially deleted data with {error_count} failed chunks"
            if symbol:
                message += f" for symbol {symbol}"
            if interval:
                message += f" with interval {interval}"
                
            return {"status": "partial", "message": message, "success_count": success_count, "error_count": error_count}
        else:
            message = f"Failed to delete any data"
            if symbol:
                message += f" for symbol {symbol}"
            if interval:
                message += f" with interval {interval}"
                
            return {"status": "error", "message": message}

    def _get_flux_interval(self, interval: str) -> str:
        """Map user-friendly intervals to Flux durations"""
        interval_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1mo"
        }
        return interval_mapping.get(interval, "1h")

    def __del__(self):
        """Ensure proper cleanup"""
        try:
            self.client.close()
        except:
            pass
