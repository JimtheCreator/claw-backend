import pandas as pd

from core.use_cases.market_analysis.analysis_snapshot import closed_candles


def test_current_minute_cannot_supply_a_confirmed_close():
    frame = pd.DataFrame({"timestamp": ["2026-09-06T10:00:00Z", "2026-09-06T10:01:00Z"], "close": [100, 120]})
    result = closed_candles(frame, "1m", "2026-09-06T10:01:35Z")
    assert list(result.close) == [100]
    assert len(frame) == 2


def test_monthly_confirmation_uses_calendar_month_not_thirty_days():
    frame = pd.DataFrame({"timestamp": ["2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"]})
    assert len(closed_candles(frame, "1M", "2026-03-01T00:00:00Z")) == 1
    assert len(closed_candles(frame, "1M", "2026-03-31T00:00:00Z")) == 1


def test_snapshot_is_sorted_deduplicated_and_index_aligned():
    frame = pd.DataFrame({"timestamp": ["2026-09-06T10:02:00Z", "2026-09-06T10:00:00Z", "2026-09-06T10:00:00Z"], "close": [102, 99, 100]})
    result = closed_candles(frame, "1m", "2026-09-06T10:04:00Z")
    assert list(result.close) == [100, 102]
    assert list(result.index) == [0, 1]
