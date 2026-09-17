from copy import deepcopy
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd
import pytest

from core.engines.analysis_chart_presentation import AnalysisChartPresentation
from core.engines.chart_engine import ChartEngine


def direction_label(fig):
    assert not any(t.name in {"Conditional forecast", "Next move to exit"} for t in fig.data)
    assert not any(t.type == "scatter" and "lines" in (t.mode or "") for t in fig.data)
    return next(a for a in fig.layout.annotations if a.name == "Forecast direction label")


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
                    market_context="aligned", setup_quality={"eligible": True}, wait_for_confirmation=True,
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


def test_wait_rejects_legacy_scenario_even_when_it_has_complete_geometry():
    candles, analysis, smc = example_data()
    original = deepcopy(analysis)
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert chart.scenarios == []
    assert any("WAIT · NO ENTRY CONFIRMED" in a.text for a in fig.layout.annotations)
    assert not any(a.name == "Forecast direction label" for a in fig.layout.annotations)
    assert not any(s.name in {"Forecast risk", "Forecast reward"} for s in fig.layout.shapes)
    assert not any("BUY NOW" in a.text for a in fig.layout.annotations)
    assert analysis == original
    assert (chart.end - chart.now) / (chart.end - chart.start) > 0.30


def test_chart_caps_historical_clutter_and_keeps_requested_timeframe():
    candles, analysis, smc = example_data("long")
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert len([s for s in fig.layout.shapes if s.type == "rect"]) == 3  # Backdrop + one TP/SL pair, not 100 zones.
    assert not any(a.text == "BOS" for a in fig.layout.annotations)
    trace = next(t for t in fig.data if t.type == "candlestick")
    assert len(trace.close) == 60
    assert pd.Timestamp(trace.x[-1]) - pd.Timestamp(trace.x[-2]) == pd.Timedelta(minutes=1)
    assert "1m chart" in fig.layout.title.text
    assert len(candles) == 180  # Selection must not discard analyzed history.


def test_disabled_mtfa_heading_does_not_echo_stale_trends():
    candles, analysis, smc = example_data("long")
    analysis["trade_plan"]["evidence"]["mtfa"]["enabled"] = False
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert "MTFA OFF" in fig.layout.title.text
    assert "15m bullish" not in fig.layout.title.text
    assert "1h bullish" not in fig.layout.title.text


def test_chart_summarizes_every_material_fact_and_discloses_anchor_cap():
    candles, analysis, smc = example_data("long")
    timestamp=candles.timestamp.iloc[-3].isoformat()
    analysis["trade_plan"]["chart_evidence"]=[
        dict(kind=f"K{i}",label=f"fact {i}",group=f"g{i}",status="passed" if i else "failed",
             mandatory=i==0,plot=i<7,timestamp=timestamp,price=100+i*.05)
        for i in range(9)
    ]
    chart=AnalysisChartPresentation(candles,analysis,smc);fig=chart.figure()
    rendered=" ".join(str(a.text).replace("<br>"," ") for a in fig.layout.annotations)
    assert all(f"fact {i}" in rendered for i in range(9))
    assert "[required]" in rendered
    assert "5 chart anchors shown; 2 additional" in rendered
    assert len([a for a in fig.layout.annotations if str(a.text).startswith("<b>") and ". K" in str(a.text)])==5


@pytest.mark.parametrize("action", ["long", "short"])
def test_forecast_preserves_plan_and_displays_direction_target_and_stop(action):
    candles, analysis, smc = example_data(action)
    original = deepcopy(analysis)
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    assert len(chart.scenarios) == 1
    assert chart.scenarios[0]["trigger"] == analysis["trade_plan"]["entry_level"]
    assert chart.scenarios[0]["invalidation"] == analysis["trade_plan"]["stop_loss"]
    assert chart.scenarios[0]["target"] == analysis["trade_plan"]["take_profit"]
    low, high = fig.layout.yaxis.range
    assert all(low < chart.scenarios[0][k] < high for k in ("trigger", "target", "invalidation"))
    badge = direction_label(fig)
    assert badge.text == ("<b>Long pending ▲</b>" if action == "long" else "<b>Short pending ▼</b>")
    assert badge.y < candles.low.iloc[-1] if action == "long" else badge.y > candles.high.iloc[-1]
    assert badge.showarrow is False
    assert any("FORECAST · ENTRY PENDING" in a.text for a in fig.layout.annotations)
    assert any("TP " in a.text for a in fig.layout.annotations)
    assert any("SL / invalid." in a.text for a in fig.layout.annotations)
    boxes = {s.name: s for s in fig.layout.shapes if s.name in {"Forecast risk", "Forecast reward"}}
    assert set(boxes) == {"Forecast risk", "Forecast reward"}
    assert all(pd.Timestamp(s.x0) == chart.now for s in boxes.values())
    assert {boxes["Forecast reward"].y0, boxes["Forecast reward"].y1} == {chart.current, chart.scenarios[0]["target"]}
    assert {boxes["Forecast risk"].y0, boxes["Forecast risk"].y1} == {chart.current, chart.scenarios[0]["invalidation"]}
    assert min(boxes["Forecast risk"].y1, boxes["Forecast reward"].y1) == max(boxes["Forecast risk"].y0, boxes["Forecast reward"].y0) == chart.current
    assert analysis == original


def test_image_path_serializes_dates_and_uses_focused_chart():
    candles, analysis, smc = example_data("long")
    chart = ChartEngine(candles.to_dict("list"), analysis_data=analysis, smc_data=smc)
    with patch("core.engines.chart_engine.pio.to_image", return_value=b"png") as renderer:
        assert chart.create_chart("image") == b"png"
    payload = renderer.call_args.args[0]
    orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    assert renderer.call_args.kwargs["height"] == 900
    assert payload["layout"]["meta"]["presentation_version"] == "conditional-forecast-v8"
    assert payload["layout"]["meta"]["forecast_reference"] == "last_closed_candle"
    candle_trace = next(t for t in payload["data"] if t["type"] == "candlestick")
    assert len(candle_trace["x"]) == 60
    assert "2026-09-06" in candle_trace["x"][0]


def test_missing_plan_scenario_is_not_replaced_with_an_invented_forecast():
    candles, analysis, smc = example_data()
    analysis["trade_plan"]["primary_scenario"] = None
    chart = AnalysisChartPresentation(candles, analysis, smc)
    assert chart.scenarios == []
    assert any("No supported forecast" in a.text for a in chart.figure().layout.annotations)


def test_unknown_later_target_does_not_hide_the_first_leg():
    candles, analysis, smc = example_data("long")
    analysis["trade_plan"]["primary_scenario"]["target"] = None
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert not any(t.name == "Conditional forecast" for t in fig.data)
    assert not any(a.name == "Forecast direction label" for a in fig.layout.annotations)
    assert any("Activation 100.00" in a.text for a in fig.layout.annotations)
    assert any("No complete forecast" in a.text for a in fig.layout.annotations)


@pytest.mark.parametrize("action, checkpoint", [("short", 102.8), ("long", 97.0)])
def test_current_reference_does_not_relabel_pending_entry_or_draw_a_retest(action, checkpoint):
    candles, analysis, smc = example_data(action)
    analysis["trade_plan"]["entry_level"] = checkpoint
    analysis["trade_plan"]["primary_scenario"]["trigger"] = checkpoint
    if action == "long":
        analysis["trade_plan"]["stop_loss"] = 95
        analysis["trade_plan"]["primary_scenario"]["invalidation"] = 95
    analysis["trade_plan"]["wait_for_confirmation"] = True
    analysis["trade_plan"]["reason"] += " Retest entry is still pending."
    original = deepcopy(analysis)
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    badge = direction_label(fig)
    assert badge.text == ("<b>Long pending ▲</b>" if action == "long" else "<b>Short pending ▼</b>")
    assert pd.Timestamp(badge.x) == candles.timestamp.iloc[-1]
    assert any("FORECAST · ENTRY PENDING" in a.text for a in fig.layout.annotations)
    assert any("Retest entry is still pending." in a.text for a in fig.layout.annotations)
    assert not any(term in a.text for a in fig.layout.annotations
                   for term in ("NEXT PROJECTED MOVE", "UP toward", "DOWN toward"))
    level_lines = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == checkpoint and s.y1 == checkpoint]
    assert len(level_lines) == 1
    assert level_lines[0].line.color == AnalysisChartPresentation.amber
    assert analysis == original


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


@pytest.mark.parametrize('changes', [dict(target=None), dict(target=99.), dict(invalidation=103.),
                                   dict(target=float('nan')), dict(trigger='bad')])
def test_invalid_or_missing_forecast_geometry_never_draws_tp_sl_boxes(changes):
    candles, analysis, smc = example_data("long")
    analysis['trade_plan']['primary_scenario'].update(changes)
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert not any(t.name == 'Conditional forecast' for t in fig.data)
    assert not any(a.name == "Forecast direction label" for a in fig.layout.annotations)
    assert not any(s.name in {'Forecast risk', 'Forecast reward'} for s in fig.layout.shapes)


def test_rejected_reference_rr_cannot_be_presented_as_a_forecast():
    candles, analysis, smc = example_data()
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert not any(s.name in {"Forecast risk", "Forecast reward"} for s in fig.layout.shapes)
    assert analysis['trade_plan']['action'] == 'wait'
    assert analysis['trade_plan']['entry_level'] is None


@pytest.mark.parametrize("interval,freq", [("1m", "min"), ("1h", "h"), ("4h", "4h"), ("1d", "D")])
@pytest.mark.parametrize("enabled", [False, True])
def test_forecast_and_risk_bands_meet_now_without_changing_activation(interval, freq, enabled):
    candles, analysis, smc = example_data("long")
    candles.timestamp = pd.date_range("2026-01-01", periods=len(candles), freq=freq, tz="UTC")
    analysis["trade_plan"]["interval"] = interval
    analysis["trade_plan"]["evidence"]["mtfa"]["enabled"] = enabled
    original = deepcopy(analysis)
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    badge = direction_label(fig)
    bands = [s for s in fig.layout.shapes if s.name in {"Forecast risk", "Forecast reward"}]
    assert pd.Timestamp(badge.x) == chart.now
    assert all(pd.Timestamp(s.x0) == chart.now for s in bands)
    assert badge.text == "<b>Long pending ▲</b>"
    assert bands[0].y0 == chart.current != 100
    assert bands[0].y1 == 105
    assert all("dashed" not in a.text.lower() for a in fig.layout.annotations)
    assert any("Pending badge requires the stated retest" in a.text for a in fig.layout.annotations)
    assert analysis == original


def test_tiny_activation_to_tp_distance_does_not_hide_current_price_reward_area():
    candles, analysis, smc = example_data("long")
    analysis["trade_plan"]["primary_scenario"].update(trigger=102.8, target=102.81)
    analysis["trade_plan"]["current_price"] = 500  # Renderer owns its candle snapshot.
    chart = AnalysisChartPresentation(candles, analysis, smc)
    fig = chart.figure()
    reward = next(s for s in fig.layout.shapes if s.name == "Forecast reward")
    assert reward.y0 == candles.close.iloc[-1]
    assert reward.y1 == 102.81
    assert reward.y1 - reward.y0 > 1
    assert chart.forecast_rr == pytest.approx(.01 / (102.8-98))
    assert any("Activation-based R:R" in a.text for a in fig.layout.annotations)
    assert any("Below the planner" in a.text.replace("<br>", " ") for a in fig.layout.annotations)
    tags = [s for s in fig.layout.shapes if s.type == "path" and s.name.endswith(" tag")]
    assert len(tags) == 4  # TP, SL, current reference and separate confirmation.
    assert all(s.xref == "x domain" and s.yref == "y domain" for s in tags)
    labels = [a for a in fig.layout.annotations if (a.name or "").startswith("Forecast ") and a.name != "Forecast direction label"]
    centers = sorted(a.y for a in labels)
    assert len(labels) == 4 and min(b-a for a, b in zip(centers, centers[1:])) >= 33/455
    assert all(0 < y < 1 for y in centers)
    assert analysis["trade_plan"]["action"] == "long"
    assert analysis["trade_plan"]["wait_for_confirmation"] is True


@pytest.mark.parametrize("direction,current,message", [
    ("bullish", 105, "reached or passed the target"),
    ("bullish", 98, "beyond scenario invalidation"),
    ("bearish", 95, "reached or passed the target"),
    ("bearish", 103, "beyond scenario invalidation"),
])
def test_current_price_outside_scenario_cannot_draw_reversed_reward_boxes(direction, current, message):
    candles, analysis, smc = example_data("long" if direction == "bullish" else "short")
    candles.loc[candles.index[-1], "close"] = current
    original = deepcopy(analysis)
    fig = AnalysisChartPresentation(candles, analysis, smc).figure()
    assert not any(t.name == "Conditional forecast" for t in fig.data)
    assert not any(a.name == "Forecast direction label" for a in fig.layout.annotations)
    assert not any(s.name in {"Forecast reward", "Forecast risk"} for s in fig.layout.shapes)
    assert any(message in a.text for a in fig.layout.annotations)
    assert analysis == original
