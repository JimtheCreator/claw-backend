"""Exercise the planner-to-chart boundary with qualified and rejected setups."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from core.engines.analysis_chart_presentation import AnalysisChartPresentation
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from tests.unit.test_setup_evidence import fixture


def inputs(direction="bullish", mtfa=True):
    candles, facts = fixture()
    # A complete structural watch used to survive subsequent entry rejections.
    facts["swings"].swings = [NS(index=20, price=105., type="high", confirmed=True)]
    args = dict(interval="5m", premium_discount=NS(range_available=True, bottom=90., top=120.),
                liquidity=NS(pools=[NS(side="buy_side", level=108., last_index=2)]), **facts)
    if not mtfa:
        args["mtfa"] = {"enabled": False}
    if direction == "bearish":
        candles["open"], candles["close"] = 200-candles.open, 200-candles.close
        high, low = candles.high.copy(), candles.low.copy()
        candles["high"], candles["low"] = 200-low, 200-high
        args["structure"].trend = "bearish"
        event = args["structure"].events[0]
        event.direction, event.level = "bearish", 200-event.level
        pivot = args["swings"].swings[0]
        pivot.type, pivot.price = "low", 200-pivot.price
        for name in ("order_blocks", "fvg", "confluence"):
            for zone in args[name].zones:
                zone.type, zone.bottom, zone.top = "bearish", 200-zone.top, 200-zone.bottom
        args["premium_discount"] = NS(range_available=True, bottom=80., top=110.)
        args["liquidity"].pools[0].side, args["liquidity"].pools[0].level = "sell_side", 92.
        if mtfa:
            args["mtfa"]["htf_trends"] = {tf: "bearish" for tf in args["mtfa"]["htf_trends"]}
            for zone in args["mtfa"]["htf_zones"]:
                zone["direction"], zone["bottom"], zone["top"] = "bearish", 200-zone["top"], 200-zone["bottom"]
    return candles, args


@pytest.mark.parametrize("direction", ["bullish", "bearish"])
@pytest.mark.parametrize("mtfa", [False, True])
def test_qualified_forecast_preserves_prices_and_explicit_pending_status(direction, mtfa):
    candles, args = inputs(direction, mtfa)
    plan = build_trade_plan(candles, **args)
    assert plan["action"] == ("long" if direction == "bullish" else "short")
    assert plan["wait_for_confirmation"] is True
    assert plan["forecast_scenario"] == plan["primary_scenario"]
    assert plan["forecast_scenario"]["trigger"] == plan["entry_level"] == 100
    assert plan["forecast_scenario"]["target"] == plan["take_profit"]
    assert plan["forecast_scenario"]["invalidation"] == plan["stop_loss"]
    original = deepcopy(plan)
    fig = AnalysisChartPresentation(candles, {"trade_plan": plan}, {}).figure()
    badge = next(a for a in fig.layout.annotations if a.name == "Forecast direction label")
    assert badge.text == ("<b>Long pending ▲</b>" if direction == "bullish" else "<b>Short pending ▼</b>")
    assert plan == original


@pytest.mark.parametrize("direction", ["bullish", "bearish"])
@pytest.mark.parametrize("failure", ["missing_htf", "mixed_htf", "opposing_htf", "missing_poi",
                                    "weak_displacement", "missing_structure", "missing_overlap_and_sweep",
                                    "missing_range", "missing_target", "poor_rr", "cost_floor"])
def test_rejected_forecast_cannot_survive_any_gate(direction, failure):
    candles, args = inputs(direction)
    opposite = "bearish" if direction == "bullish" else "bullish"
    if failure == "missing_htf":
        args["mtfa"]["htf_unavailable"] = {"4h": "unavailable"}
    elif failure == "mixed_htf":
        args["mtfa"]["htf_trends"]["4h"] = opposite
    elif failure == "opposing_htf":
        args["mtfa"]["htf_trends"] = {tf: opposite for tf in args["mtfa"]["htf_trends"]}
    elif failure == "missing_poi":
        args["mtfa"]["htf_zones"] = []
    elif failure == "weak_displacement":
        candles.loc[25, "open"] = candles.close.iloc[25]
    elif failure == "missing_structure":
        args["structure"].events = []
    elif failure == "missing_overlap_and_sweep":
        args["confluence"].zones = []
    elif failure == "missing_range":
        args["premium_discount"].range_available = False
    elif failure == "missing_target":
        args["liquidity"].pools = []
    elif failure == "poor_rr":
        for name in ("order_blocks", "fvg", "confluence"):
            for zone in args[name].zones:
                if direction == "bullish":
                    zone.bottom = 95
                else:
                    zone.top = 105
    elif failure == "cost_floor":
        args.update(cost_policy="minimum_stop_bps", minimum_stop_bps=1000)
    plan = build_trade_plan(candles, **args)
    assert plan["action"] == "wait", failure
    assert plan["primary_scenario"] is plan["forecast_scenario"] is None
    assert plan["entry_level"] is plan["stop_loss"] is plan["take_profit"] is None
    assert plan["market_read"] and plan["reason"]
    fig = AnalysisChartPresentation(candles, {"trade_plan": plan}, {}).figure()
    assert not any(a.name == "Forecast direction label" for a in fig.layout.annotations)
    assert not any(s.name in {"Forecast risk", "Forecast reward"} for s in fig.layout.shapes)
    assert any(a.text == "MARKET WATCH · NO FORECAST APPROVED" for a in fig.layout.annotations)
    assert not any("CONDITIONAL FORECAST · TIMING" in a.text for a in fig.layout.annotations)


@pytest.mark.parametrize("failure", ["wait", "context", "missing_context", "ineligible", "missing_quality", "direction", "watch"])
def test_renderer_rejects_stale_or_inconsistent_forecast_payloads(failure):
    candles, args = inputs()
    plan = build_trade_plan(candles, **args)
    if failure == "wait":
        plan["action"] = "wait"
    elif failure == "context":
        plan["market_context"] = "mixed"
    elif failure == "missing_context":
        del plan["market_context"]
    elif failure == "ineligible":
        plan["setup_quality"]["eligible"] = False
    elif failure == "missing_quality":
        del plan["setup_quality"]
    elif failure == "direction":
        plan["primary_scenario"]["direction"] = "bearish"
    elif failure == "watch":
        plan["primary_scenario"]["setup"] = False
    original = deepcopy(plan)
    chart = AnalysisChartPresentation(candles, {"trade_plan": plan}, {})
    assert chart.scenarios == []
    assert not any(a.name == "Forecast direction label" for a in chart.figure().layout.annotations)
    assert plan == original
