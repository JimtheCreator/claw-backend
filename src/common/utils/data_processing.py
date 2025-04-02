import pandas as pd

def normalize_binance_data(klines: list) -> pd.DataFrame:
    """
    Convert Binance klines response to standardized DataFrame
    
    Binance Response Format:
    [
        Open time,
        Open,
        High,
        Low,
        Close,
        Volume,
        Close time,
        Quote asset volume,
        Number of trades,
        Taker buy base asset volume,
        Taker buy quote asset volume,
        Ignore
    ]
    """
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]
    
    df = pd.DataFrame(klines, columns=columns, dtype=float)
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Keep essential columns
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def downsample_sparkline(data: list, points: int = 20) -> list:
    """Reduce data points for efficient sparkline rendering"""
    if len(data) <= points:
        return data
    step = len(data) // points
    return [data[i] for i in range(0, len(data), step)][:points]