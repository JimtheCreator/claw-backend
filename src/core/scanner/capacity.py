"""Shared application limits for scanner admission and its socket owner."""
import os


def bounded(name, default, maximum):
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}')
    return value


def gateway_limits():
    return (bounded('BINANCE_WS_CONNECTIONS', 3, 24),
            bounded('BINANCE_WS_STREAMS_PER_CONNECTION', 200, 800))


def scanner_stream_budget():
    budget = bounded('SCANNER_STREAM_BUDGET', 200, 19000)
    connections, streams = gateway_limits()
    # Keep capacity for the price-alert feed and interactive chart subscriptions.
    if budget > connections * streams - 32:
        raise ValueError('Scanner stream budget exceeds gateway capacity minus 32 reserved streams')
    return budget
