import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.use_cases.market_analysis.analyze_with_mtfa import analyze_with_mtfa


def _make_df(closes, start=None):
    start = start or datetime(2026, 1, 1)
    n = len(closes)
    return pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "open": closes, "high": closes, "low": closes, "close": closes,
    })


def _bullish_closes():
    rising = [100 + i for i in range(10)]
    pullback = [108, 107, 106]
    resume_up = [107, 108, 109, 110, 111, 112]  # breaks back above the 109 peak
    return rising + pullback + resume_up


def _bearish_closes():
    falling = [109 - i for i in range(10)]       # 109..100
    pullback = [101, 102, 103]
    resume_down = [102, 101, 100, 99, 98, 97]     # breaks back below the 100 trough
    return falling + pullback + resume_down


async def _fetcher_factory(closes_by_tf, calls=None):
    async def fetcher(symbol, interval):
        if calls is not None:
            calls.append((symbol, interval))
        return _make_df(closes_by_tf[interval])
    return fetcher


def test_mtfa_off_returns_standalone_context():
    async def run():
        closes_by_tf = {"15m": _bullish_closes()}
        fetcher = await _fetcher_factory(closes_by_tf)
        return await analyze_with_mtfa("BTCUSDT", "15m", mtfa_enabled=False, candle_fetcher=fetcher)

    result = asyncio.run(run())
    assert result["context"] == "standalone"
    assert result["htf"] == {}
    assert result["htf_trend_alignment"] == {}
    assert result["requested"]["interval"] == "15m"


def test_top_of_ladder_is_standalone_even_with_mtfa_on():
    async def run():
        closes_by_tf = {"1M": _bullish_closes()}
        fetcher = await _fetcher_factory(closes_by_tf)
        return await analyze_with_mtfa("BTCUSDT", "1M", mtfa_enabled=True, candle_fetcher=fetcher)

    result = asyncio.run(run())
    assert result["context"] == "standalone"


def test_mtfa_on_fetches_full_htf_chain_and_computes_alignment():
    async def run():
        bullish = _bullish_closes()
        closes_by_tf = {"1h": bullish, "4h": bullish, "1d": bullish, "1w": bullish}
        calls = []
        fetcher = await _fetcher_factory(closes_by_tf, calls)
        result = await analyze_with_mtfa("BTCUSDT", "1h", mtfa_enabled=True, candle_fetcher=fetcher)
        return result, calls

    result, calls = asyncio.run(run())

    assert result["context"] == "mtfa"
    assert set(result["htf"].keys()) == {"4h", "1d", "1w"}
    # every configured TF (requested + full chain) should have been fetched
    assert {tf for _, tf in calls} == {"1h", "4h", "1d", "1w"}
    # same bullish data everywhere -> every HTF should align with the requested TF
    assert all(v is True for v in result["htf_trend_alignment"].values())


def test_disagreeing_htf_trend_is_surfaced_not_suppressed():
    async def run():
        closes_by_tf = {"1h": _bullish_closes(), "4h": _bearish_closes(), "1d": _bearish_closes(), "1w": _bearish_closes()}
        fetcher = await _fetcher_factory(closes_by_tf)
        return await analyze_with_mtfa("BTCUSDT", "1h", mtfa_enabled=True, candle_fetcher=fetcher)

    result = asyncio.run(run())

    assert result["context"] == "mtfa"
    # requested TF's own result must be untouched/present despite disagreement
    assert result["requested"]["market_structure"].trend == "bullish"
    # disagreement is reported, not hidden
    assert result["htf_trend_alignment"]["4h"] is False


def test_one_failing_htf_is_excluded_others_unaffected():
    async def run():
        bullish = _bullish_closes()

        async def fetcher(symbol, interval):
            if interval == "1d":
                raise ConnectionError("simulated DB timeout")
            return _make_df(bullish)

        return await analyze_with_mtfa("BTCUSDT", "1h", mtfa_enabled=True, candle_fetcher=fetcher)

    result = asyncio.run(run())

    assert result["context"] == "mtfa"
    assert "1d" not in result["htf"]
    assert "1d" not in result["htf_trend_alignment"]
    assert set(result["htf"].keys()) == {"4h", "1w"}


def test_requested_tf_failure_propagates():
    async def failing_fetcher(symbol, interval):
        raise ConnectionError("simulated outage")

    async def run():
        await analyze_with_mtfa("BTCUSDT", "1h", mtfa_enabled=True, candle_fetcher=failing_fetcher)

    with pytest.raises(ConnectionError):
        asyncio.run(run())