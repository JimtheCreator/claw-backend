import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.core.domain.entities.FairValueGapEntity import FVGResult, FVGZone
from src.core.domain.entities.LiquidityEntity import LiquidityMapResult, LiquidityPool
from src.core.domain.entities.PremiumDiscountEntity import PremiumDiscountResult
from src.core.use_cases.market_analysis.analyze_with_mtfa import analyze_with_mtfa
from src.core.use_cases.market_analysis.trade_plan import build_trade_plan
from src.core.use_cases.market_analysis.trade_plan import _market_context, _structure_watch


def _candles() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h"),
        "open": [99.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
    })


def _long_plan(mtfa):
    return build_trade_plan(
        _candles(),
        interval="1h",
        structure=SimpleNamespace(trend="bullish"),
        premium_discount=PremiumDiscountResult(
            interval="1h", range_available=True, top=110, bottom=90,
            current_price=100, zone="discount",
        ),
        liquidity=LiquidityMapResult(interval="1h", pools=[
            LiquidityPool(side="buy_side", level=110, touches=2,
                          contributing_swing_indices=[4, 8], first_index=4, last_index=8),
        ]),
        fvg=FVGResult(interval="1h", zones=[
            FVGZone(type="bullish", bottom=96, top=98, start_index=8, formed_index=10,
                    formed_timestamp=pd.Timestamp("2026-01-01"), mitigation_status="unmitigated"),
        ]),
        order_blocks=SimpleNamespace(zones=[]),
        confluence=SimpleNamespace(zones=[]),
        mtfa=mtfa,
    )


def test_trade_plan_gives_conditional_levels_for_aligned_setup():
    plan = _long_plan({
        "enabled": True,
        "context": "mtfa",
        "htf_unavailable": {},
        "htf_trend_alignment": {"4h": True, "1d": True},
    })

    assert plan["action"] == "long"
    assert plan["trend_direction"] == "bullish"
    assert plan["entry_level"] == 98
    assert plan["stop_loss"] < 96
    assert plan["take_profit"] == 110
    assert plan["risk_reward"] >= 1.5
    assert plan["wait_for_confirmation"] is True
    assert "close back above" in plan["confirmation_required"]


def test_trade_plan_explains_split_higher_timeframes_without_predicting_reversal():
    plan = _long_plan({
        "enabled": True,
        "context": "mtfa",
        "htf_unavailable": {},
        "htf_trend_alignment": {"4h": False, "1d": True},
    })

    assert plan["action"] == "wait"
    assert plan["entry_level"] is None
    assert "split" in plan["reason"]
    assert plan["market_context"] == "mixed"
    assert plan["primary_scenario"] is None


def _pullback_plan(direction="bullish", confirmed=True, missing=False, target=True):
    bullish = direction == "bullish"
    return build_trade_plan(
        _candles(), interval="1m", structure=SimpleNamespace(trend="bearish" if bullish else "bullish"),
        premium_discount=None, fvg=None, order_blocks=None, confluence=None,
        liquidity=SimpleNamespace(pools=[SimpleNamespace(side="buy_side" if bullish else "sell_side",
                                                        level=110 if bullish else 90, last_index=3)] if target else []),
        swings=SimpleNamespace(swings=[SimpleNamespace(confirmed=confirmed, type="high" if bullish else "low",
                                                        price=102 if bullish else 98, index=5)]),
        mtfa={"enabled": True, "htf_trends": {"15m": direction, "1h": direction},
              "htf_requested": ["15m", "1h"], "htf_unavailable": {"1h": "error"} if missing else {}},
    )


@pytest.mark.parametrize("direction", ["bullish", "bearish"])
def test_countertrend_is_a_conditional_resumption_watch_not_a_trade(direction):
    plan = _pullback_plan(direction)
    assert plan["market_context"] == "pullback"
    assert "possible pullback" in plan["reason"]
    assert "not a confirmed reversal" in plan["reason"]
    assert plan["action"] == "wait"
    assert plan["entry_level"] is plan["stop_loss"] is None
    scenario = plan["primary_scenario"]
    assert scenario["direction"] == direction
    assert scenario["trigger"] == (102 if direction == "bullish" else 98)
    assert scenario["invalidation"] == (99 if direction == "bullish" else 101)
    assert scenario["target"] == (110 if direction == "bullish" else 90)
    assert "retest" in plan["confirmation_required"]


def test_provisional_pivot_cannot_confirm_a_reversal():
    assert _pullback_plan(confirmed=False)["primary_scenario"] is None


def test_missing_htf_does_not_draw_a_reversal_forecast():
    plan = _pullback_plan(missing=True)
    assert plan["primary_scenario"] is None
    assert plan["market_context"] == "incomplete"


def test_no_liquidity_target_is_not_invented_for_pullback():
    assert _pullback_plan(target=False)["primary_scenario"]["target"] is None


def test_nearest_htf_correction_is_a_nested_pullback_when_larger_frames_agree():
    context, bias, reason = _market_context({"enabled": True, "htf_trends": {
        "4h": "bullish", "15m": "bearish", "1h": "bullish"}}, "bearish")
    assert context == "pullback"
    assert bias == "bullish"
    assert "intermediate structure must confirm" in reason


def test_largest_frames_disagreement_does_not_become_a_majority_vote():
    context, bias, _ = _market_context({"enabled": True, "htf_trends": {
        "15m": "bullish", "1h": "bullish", "4h": "bearish"}}, "bullish")
    assert context == "mixed"
    assert bias is None


def test_already_broken_pivot_is_not_a_future_reversal_trigger():
    frame = _candles()
    frame.loc[7, "close"] = 103
    swings = SimpleNamespace(swings=[SimpleNamespace(confirmed=True, type="high", price=102, index=5)])
    assert _structure_watch(frame, swings, "bullish", None, "pullback") is None


def test_swept_liquidity_is_not_a_future_pullback_target():
    frame = _candles()
    frame.loc[2, "high"] = 111
    swings = SimpleNamespace(swings=[SimpleNamespace(confirmed=True, type="high", price=102, index=5)])
    liquidity = SimpleNamespace(pools=[SimpleNamespace(side="buy_side", level=110, last_index=1)])
    scenario = _structure_watch(frame, swings, "bullish", liquidity, "pullback")
    assert scenario["target"] is None


def test_mtfa_reuses_requested_snapshot_and_reports_missing_htf():
    requested = {"market_structure": SimpleNamespace(trend="bullish")}

    async def fetcher(symbol, interval):
        if interval == "1d":
            raise RuntimeError("data source unavailable")
        return pd.DataFrame({"interval": [interval]})

    async def fake_analyze(frame, interval):
        return {"market_structure": SimpleNamespace(trend="bullish")}

    async def run_test():
        with patch(
            "src.core.use_cases.market_analysis.analyze_with_mtfa.analyze_smc_structure",
            new=AsyncMock(side_effect=fake_analyze),
        ) as analysis:
            result = await analyze_with_mtfa(
                "BTCUSDT", "1h", True, fetcher, requested_result=requested
            )
        return result, analysis

    result, analysis = asyncio.run(run_test())

    assert analysis.await_count == 2
    assert result["requested"] is requested
    assert result["htf_requested"] == ["4h", "1d", "1w"]
    assert result["htf_unavailable"] == {"1d": "RuntimeError"}
    assert result["htf_trend_alignment"] == {"4h": True, "1w": True}
