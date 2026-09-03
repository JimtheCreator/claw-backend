from datetime import datetime, timedelta

import pandas as pd

from core.engines.fvg_engine import FVGEngine


def _make_df(candles, start=None):
    """candles: list of (high, low, close) tuples."""
    start = start or datetime(2026, 1, 1)
    n = len(candles)
    return pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "open": [c[2] for c in candles],
        "high": [c[0] for c in candles],
        "low": [c[1] for c in candles],
        "close": [c[2] for c in candles],
    })


def test_bullish_fvg_detected_and_fully_mitigated():
    candles = [
        (10, 9, 9.5),     # idx0: c1 - high=10
        (20, 15, 18),     # idx1: impulse candle
        (14, 13, 13.5),   # idx2: c3 - low=13 > c1.high=10 -> bullish gap [10, 13]
        (13, 12, 12.5),   # idx3: low=12 -> partial fill (in [10,13], not through bottom)
        (10, 9.5, 9.8),   # idx4: low=9.5 <= bottom(10) -> fully mitigated
    ]
    df = _make_df(candles)

    result = FVGEngine(interval="1h").detect_fvgs(df)

    # The return move down (needed to test mitigation) can legitimately
    # form its own separate gap elsewhere in the series - that's real
    # behaviour, not a bug. Isolate the specific zone under test.
    bullish = [z for z in result.zones if z.type == "bullish" and z.start_index == 0]
    assert len(bullish) == 1
    z = bullish[0]
    assert z.top == 13.0
    assert z.bottom == 10.0
    assert z.formed_index == 2
    assert z.mitigation_status == "fully_mitigated"
    assert z.mitigated_index == 4


def test_bearish_fvg_detected_and_fully_mitigated():
    candles = [
        (21, 20, 20.5),   # idx0: c1 - low=20
        (10, 5, 7),       # idx1: impulse candle down
        (18, 17, 17.5),   # idx2: c3 - high=18 < c1.low=20 -> bearish gap [18, 20]
        (19, 16, 18.5),   # idx3: high=19 -> partial fill (in [18,20), not through top)
        (21, 18, 20.5),   # idx4: high=21 >= top(20) -> fully mitigated
    ]
    df = _make_df(candles)

    result = FVGEngine(interval="1h").detect_fvgs(df)

    bearish = [z for z in result.zones if z.type == "bearish" and z.start_index == 0]
    assert len(bearish) == 1
    z = bearish[0]
    assert z.top == 20.0
    assert z.bottom == 18.0
    assert z.mitigation_status == "fully_mitigated"
    assert z.mitigated_index == 4


def test_unmitigated_when_price_never_returns():
    candles = [
        (10, 9, 9.5),
        (20, 15, 18),
        (14, 13, 13.5),   # bullish gap [10, 13] formed here
        (16, 14, 15),     # stays above the gap entirely
        (18, 15, 17),
    ]
    df = _make_df(candles)

    result = FVGEngine(interval="1h").detect_fvgs(df)

    target = next(z for z in result.zones if z.start_index == 0)
    assert target.type == "bullish"
    assert target.mitigation_status == "unmitigated"
    assert target.mitigated_index is None


def test_formation_candle_never_self_mitigates():
    # candle3's low IS the gap's top by construction - must not register
    # as an instant partial/full fill on the candle that formed the zone.
    candles = [
        (10, 9, 9.5),
        (20, 15, 18),
        (14, 13, 13.5),   # forms gap [10, 13]; this candle's own low == top
    ]
    df = _make_df(candles)

    result = FVGEngine(interval="1h").detect_fvgs(df)

    assert len(result.zones) == 1
    assert result.zones[0].mitigation_status == "unmitigated"


def test_insufficient_candles_returns_empty_not_raise():
    df = _make_df([(10, 9, 9.5), (11, 10, 10.5)])
    result = FVGEngine(interval="1h").detect_fvgs(df)
    assert result.zones == []


def test_missing_columns_returns_empty_not_raise():
    df = pd.DataFrame({"timestamp": [datetime(2026, 1, 1)]})
    result = FVGEngine(interval="1h").detect_fvgs(df)
    assert result.zones == []


def test_no_gap_when_candles_overlap():
    # c1.high (15) is not < c3.low (12), and c1.low (9) is not > c3.high (14)
    # - candles overlap, no gap either direction.
    candles = [
        (15, 9, 12),
        (16, 10, 13),
        (14, 12, 13),
    ]
    df = _make_df(candles)

    result = FVGEngine(interval="1h").detect_fvgs(df)
    assert result.zones == []