# src/infrastructure/database/influxdb/market_data_repository.py
from influxdb_client import InfluxDBClient
from core.interfaces.market_data_repository import MarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from common.logger import logger
from influxdb_client import InfluxDBClient, Point
from typing import Optional
from datetime import datetime, timezone
from pydantic import ValidationError


class InfluxDBMarketDataRepository(MarketDataRepository):
    def __init__(self):
        self.client = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL"),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG")
        )
        self.bucket = os.getenv("INFLUXDB_BUCKET")
        self._verify_connection()

    def _verify_connection(self):
        """Verify InfluxDB connection on initialization"""
        try:
            if self.client.ping():
                logger.info("✅ Successfully connected to InfluxDB")
            else:
                logger.error("❌ InfluxDB connection verification failed")
        except InfluxDBError as e:
            logger.critical(f"🔥 Critical InfluxDB connection failure: {str(e)}")
            raise

    
    # In market_data_repository.py
    @staticmethod
    def parse_flux_record(record) -> dict:
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

    async def get_historical_data(self, symbol: str, interval: str, start_time: Optional[datetime] = None) -> list[MarketDataEntity]:
        flux_interval = self._get_flux_interval(interval)
        start_str = f"start: {start_time.isoformat()}" if start_time else "start: -5y"
        
        query = f'''
        from(bucket: "{self.bucket}")
        |> range({start_str})
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.interval == "{interval}")
        |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
        |> aggregateWindow(every: {flux_interval}, fn: last, createEmpty: false)
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
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

            
            logger.info("Data retrieved from INFLUXDB")
            return parsed_records
        except Exception as e:
            logger.error(f"InfluxDB query error: {str(e)}")
            return []

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
            "1M": "1mo"  # Add monthly interval mapping
        }
        return interval_mapping.get(interval, "1h")

    def __del__(self):
        """Ensure proper cleanup"""
        self.client.close()

    
    