"""Detection lanes keep one market/timeframe from monopolizing the worker."""
from core.scanner.catalog import INTERVAL_SECONDS


def detection_queue(candidate, interval):
    manifest = candidate['manifest']
    identity = (manifest['provider'], manifest['market'])
    if identity not in {('binance', 'spot'), ('massive', 'forex'), ('massive', 'crypto')}:
        raise ValueError('Unsupported scanner provider/market')
    if interval not in INTERVAL_SECONDS:
        raise ValueError('Unsupported scanner interval')
    return f'scanner_{identity[0]}_{identity[1]}_{interval}'
