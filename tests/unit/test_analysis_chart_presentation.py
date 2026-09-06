from copy import deepcopy
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd
import pytest

from core.engines.analysis_chart_presentation import AnalysisChartPresentation
from core.engines.chart_engine import ChartEngine


def example_data(action="wait"):
    # Deterministic illustrative candles, not live market data.
    close = 100 + np.sin(np.arange(180) * 0.19) * 2 + np.sin(np.arange(180) * 0.59) * 0.4
    opens = np.r_[close[0], close[:-1]]
    candles = pd.DataFrame(dict(timestamp=pd.date_range("2026-09-06 10:00", periods=180, freq="min", tz="UTC"),
                                open=opens, high=np.maximum(opens, close) + 0.4,
                                low=np.minimum(opens, close) - 0.4, close=close,
                                volume=100 + np.arange(180) % 40))
    plan = dict(action=action, interval="1m", trend_direction="bearish" if action != "long" else "bullish",
                current_price=float(close[-1]), entry_level=None, stop_loss=None, take_profit=None,
                reason="Bearish local structure inside a bullish higher-timeframe trend: possible pullback, not a confirmed reversal.",
                evidence={"mtfa": {"enabled": True, "htf_trends": {"15m": "bullish", "1h": "bullish"}}},
                primary_scenario=dict(kind="pullback_reversal_watch", direction="bullish", setup=False,
                                      title="Potential bullish resumption", trigger=102.8, target=105,
                                      invalidation=97.2))
    if action in {"long", "short"}:
        plan["evidence"]["mtfa"]["htf_trends"] = {tf: "bullish" if action == "long" else "bearish" for tf in ("15m", "1h")}
        plan["evidence"]["mtfa"]["htf_trend_alignment"] = {"15m": True, "1h": True}
        plan.update(entry_level=100, stop_loss=98 if action == "long" else 103,
                    take_profit=105 if action == "long" else 95,
                    entry_zone={"bottom": 99.8, "top": 100.2},
                    confirmation_required="Wait for a candle to reject the zone and confirm structure.",
                    reason="Directional structure, fresh zone and target meet the setup rules.")
        plan["primary_scenario"] = dict(direction="bullish" if action == "long" else "bearish", setup=True,
                                         title=f"{action.title()} setup", trigger=plan["entry_level"],
                                         target=plan["take_profit"], invalidation=plan["stop_loss"])
    zones = [dict(type="bullish", bottom=98.0, top=98.4, start_index=i,
                  mitigation_status="unmitigated") for i in range(100)]
    zones += [dict(type="bearish", bottom=102.4, top=102.8, start_index=130,
                   mitigation_status="unmitigated")]
    zones += [dict(type="bearish", bottom=101.0, top=101.2, start_index=178,
                   mitigation_status="fully_mitigated")]
    smc = {"fvg": {"zones": zones}, "liquidity": {"pools": []},
           "market_structure": {"events": [dict(kind="BOS", direction="bearish", index=i,
                                                reference_swing_index=i - 3, level=100.0) for i in range(50, 180, 15)]}}
    return candles, {"trade_plan": plan, "symbol": "ILLUSTRATION"}, smc


def test_wait_uses_only_the_plan_scenario_not_the_opposing_local_trend():
    candles, analysis, smc = example_data()
    original = deepcopy(analysis)
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert len(chart.scenarios) == 1
    assert chart.scenarios[0]["direction"] == "bullish"
    assert chart.scenarios[0]["target"] == 105
    assert not chart.scenarios[0]["setup"]
    assert any("Close above" in a.text for a in fig.layout.annotations)
    assert not any("Close below" in a.text for a in fig.layout.annotations)
    assert any("WAIT" in a.text for a in fig.layout.annotations)
    assert analysis == original
    assert (chart.end - chart.now) / (chart.end - chart.start) > 0.30


def test_chart_caps_historical_clutter_and_keeps_requested_timeframe():
    candles, analysis, smc = example_data()
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert len([s for s in fig.layout.shapes if s.type == "rect"]) == 1  # Forecast backdrop only.
    assert not any(a.text == "BOS" for a in fig.layout.annotations)
    trace = next(t for t in fig.data if t.type == "candlestick")
    assert len(trace.close) == 60
    assert pd.Timestamp(trace.x[-1]) - pd.Timestamp(trace.x[-2]) == pd.Timedelta(minutes=1)
    assert "1m chart" in fig.layout.title.text
    assert len(candles) == 180  # Selection must not discard analyzed history.


@pytest.mark.parametrize("action", ["long", "short"])
def test_setups_keep_exact_entry_stop_and_target_in_view(action):
    candles, analysis, smc = example_data(action)
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert len(chart.scenarios) == 1
    assert chart.scenarios[0]["trigger"] == analysis["trade_plan"]["entry_level"]
    assert chart.scenarios[0]["invalidation"] == analysis["trade_plan"]["stop_loss"]
    assert chart.scenarios[0]["target"] == analysis["trade_plan"]["take_profit"]
    low, high = fig.layout.yaxis.range
    assert low < chart.scenarios[0]["invalidation"] < high
    assert low < chart.scenarios[0]["target"] < high


def test_image_path_serializes_dates_and_uses_focused_chart():
    candles, analysis, smc = example_data()
    chart = ChartEngine(candles.to_dict("list"), analysis_data=analysis, smc_data=smc)
    with patch("core.engines.chart_engine.pio.to_image", return_value=b"png") as renderer:
        assert chart.create_chart("image") == b"png"
    payload = renderer.call_args.args[0]
    orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    assert renderer.call_args.kwargs["height"] == 900
    assert payload["layout"]["meta"]["presentation_version"] == "evidence-v4"
    candle_trace = next(t for t in payload["data"] if t["type"] == "candlestick")
    assert len(candle_trace["x"]) == 60
    assert "2026-09-06" in candle_trace["x"][0]


def test_missing_plan_scenario_is_not_replaced_with_an_invented_forecast():
    candles, analysis, smc = example_data()
    analysis["trade_plan"]["primary_scenario"] = None
    chart = AnalysisChartPresentation(candles, analysis, smc)
    assert chart.scenarios == []
    assert any("No supported path" in a.text for a in chart.figure().layout.annotations)


def test_unknown_target_stops_path_at_retest():
    candles, analysis, smc = example_data()
    analysis["trade_plan"]["primary_scenario"]["target"] = None
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    path = next(t for t in fig.data if t.name == "Conditional scenario")
    assert len(path.x) == 3
    assert path.y[-1] == analysis["trade_plan"]["primary_scenario"]["trigger"]
    assert any("No unswept target" in a.text for a in fig.layout.annotations)


def test_mtfa_off_does_not_require_higher_timeframe_confirmation_in_watch_labels():
    candles, analysis, smc = example_data()
    analysis["trade_plan"]["evidence"]["mtfa"] = {"enabled": False}
    analysis["trade_plan"]["reason"] = "Wait for a valid fresh zone."
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert "MTFA OFF" in fig.layout.title.text
    assert all("MTFA before entry" not in a.text for a in fig.layout.annotations)


def test_monthly_chart_keeps_month_label_distinct_from_minutes():
    candles, analysis, smc = example_data()
    analysis["trade_plan"]["interval"] = "1M"
    candles.timestamp = pd.date_range("2011-01-01", periods=len(candles), freq="MS", tz="UTC")
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert "1 month chart" in fig.layout.title.text
