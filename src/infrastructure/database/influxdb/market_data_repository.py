# src/infrastructure/database/influxdb/market_data_repository.py
from influxdb_client import InfluxDBClient
from core.interfaces.market_data_repository import MarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from common.logger import logger

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
        return {
            'timestamp': record.get_time(),
            'symbol': record.values.get('symbol'),
            'interval': record.values.get('interval'),
            'open': record.values.get('open') or 0.0,  # Fallback to 0.0 if None
            'high': record.values.get('high') or 0.0,
            'low': record.values.get('low') or 0.0,
            'close': record.values.get('close') or 0.0,
            'volume': record.values.get('volume') or 0.0
        }

    async def save_market_data(self, data: MarketDataEntity) -> None:
        """Add error handling and batch writing"""
        try:
            with self.client.write_api() as write_api:
                write_api.write(
                    bucket=self.bucket,
                    record={
                        "measurement": "market_data",
                        "tags": {
                            "symbol": data.symbol,
                            "interval": data.interval  # Must be a string
                        },
                        "fields": {
                            "open": data.open,
                            "high": data.high,
                            "low": data.low,
                            "close": data.close,
                            "volume": data.volume
                        },

            

                        "time": data.timestamp.isoformat()
                    }
                )
        except InfluxDBError as e:
            logger.error(f"InfluxDB write error: {str(e)}")
            raise
        

    async def get_historical_data(self, symbol: str, interval: str) -> list[MarketDataEntity]:
        flux_interval = self._get_flux_interval(interval)
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -5y)
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r._field == "open" or r._field == "high" or r._field == "low" or r._field == "close" or r._field == "volume")
        |> aggregateWindow(every: {flux_interval}, fn: last)
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            result = self.client.query_api().query(query)
            parsed_records = []
            for table in result:
                for record in table.records:
                    parsed = self.parse_flux_record(record)
                    logger.debug(f"Parsed record: {parsed}")  # Debug log
                    parsed_records.append(MarketDataEntity(**parsed))

            if not parsed_records:
                logger.warning(f"No data found for {symbol}/{interval}")
                return []
        
            return parsed_records
        except InfluxDBError as e:
            logger.error(f"InfluxDB query error: {str(e)}")
            return []
        

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