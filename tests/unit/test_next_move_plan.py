from copy import deepcopy
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from tests.unit.test_setup_evidence import fixture
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from core.engines.analysis_chart_presentation import AnalysisChartPresentation


def inputs():
    df, facts = fixture()
    df.loc[29,["open","high","low","close"]]=[101.2,102,100.8,101.8]
    facts["mtfa"]={"enabled":False}
    return df,dict(interval="5m",premium_discount=NS(range_available=True,top=120,bottom=90),
                   liquidity=NS(pools=[NS(side="buy_side",level=110,last_index=2)]),
                   execution_policy="next_move",**facts)


def test_buy_is_at_current_close_and_exits_at_first_obstacle():
    df,args=inputs()
    plan=build_trade_plan(df,**args)
    assert plan["action"]=="long"
    assert plan["entry_level"]==df.close.iloc[-1]
    assert plan["take_profit"]==110 and len(plan["targets"])==1
    assert plan["targets"][0]["fraction"]==1
    assert plan["stop_loss"]<99 and plan["wait_for_confirmation"] is False
    assert plan["cost_economics"]["net_reward_risk"]>=1.5
    fig=AnalysisChartPresentation(df,{"trade_plan":plan,"symbol":"TEST"},{}).figure()
    trace=next(t for t in fig.data if t.name=="Next move to exit")
    assert list(trace.y)==[df.close.iloc[-1],110]
    assert not any("Retest" in a.text or "T2" in a.text for a in fig.layout.annotations)


def test_short_is_mirror_and_uses_a_stop_above_entry():
    df,args=inputs()
    df["open"],df["close"] = 200-df.open,200-df.close
    high,low=df.high.copy(),df.low.copy()
    df["high"],df["low"]=200-low,200-high
    args["structure"].trend="bearish"
    event=args["structure"].events[0]; event.direction="bearish";event.level=99
    for name in ("order_blocks","fvg","confluence"):
        for zone in args[name].zones:
            zone.type="bearish";zone.bottom,zone.top=200-zone.top,200-zone.bottom
    args["premium_discount"]=NS(range_available=True,bottom=80,top=110)
    args["liquidity"]=NS(pools=[NS(side="sell_side",level=90,last_index=2)])
    plan=build_trade_plan(df,**args)
    assert plan["action"]=="short" and plan["entry_level"]==pytest.approx(98.2)
    assert plan["stop_loss"]>plan["entry_level"]>plan["take_profit"]==90


def test_distant_future_entry_is_not_relabelled_as_a_buy():
    df,args=inputs();df.loc[29,["open","high","low","close"]]=[103,104,102,103]
    plan=build_trade_plan(df,**args)
    assert plan["action"]=="wait" and plan["primary_scenario"] is None
    assert plan["entry_level"] is None and plan["decision_level"]["price"]==110
    assert "No fresh entry trigger" in plan["reason"]
    fig=AnalysisChartPresentation(df,{"trade_plan":plan,"symbol":"TEST"},{}).figure()
    assert not any(t.name in {"Next move to exit","Conditional scenario"} for t in fig.data)
    assert any("Watch 110" in a.text for a in fig.layout.annotations)


def test_nearer_obstacle_cannot_be_skipped_to_improve_reward():
    df,args=inputs()
    args["liquidity"].pools.append(NS(side="buy_side",level=104.5,last_index=2))
    plan=build_trade_plan(df,**args)
    assert plan["decision_level"]["price"]==104.5
    assert plan["action"]=="wait" and plan["primary_scenario"] is None
    assert plan["cost_economics"]["net_reward_risk"]<1.5


def test_actual_opposing_zone_is_an_exit_before_farther_liquidity():
    df,args=inputs()
    args["order_blocks"].zones.append(NS(type="bearish",bottom=108,top=109,breakout_index=20,
        candle_index=18,mitigation_status="unmitigated"))
    plan=build_trade_plan(df,**args)
    assert plan["action"]=="long" and plan["take_profit"]==108
    assert plan["decision_level"]["source"]=="local_order_block"


def test_mtfa_off_isolated_and_on_keeps_local_fact_when_blocked():
    df,args=inputs()
    _,f=fixture()
    args["mtfa"]=deepcopy(f["mtfa"])
    on=build_trade_plan(df,**args)
    assert on["action"]=="long"
    args["mtfa"]["htf_trends"]={"1h":"bearish","4h":"bearish"}
    blocked=build_trade_plan(df,**args)
    assert blocked["action"]=="wait" and blocked["primary_scenario"] is None
    assert any(a["kind"]=="CHoCH" for a in blocked["chart_evidence"])
    args["mtfa"]["enabled"]=False
    stale=deepcopy(args["mtfa"])
    off=build_trade_plan(df,**args)
    args["mtfa"]={"enabled":False}
    clean=build_trade_plan(df,**args)
    assert off==clean and off["action"]=="long"
    assert "selected_htf_poi" not in off
    assert all("htf" not in a["group"] for a in off["chart_evidence"])
    assert stale["htf_trends"]=={"1h":"bearish","4h":"bearish"}


def test_obsolete_or_future_event_cannot_authorize_current_entry():
    df,args=inputs();args["structure"].events[0].index=30
    assert build_trade_plan(df,**args)["action"]=="wait"
    df,args=inputs();df.loc[27,"close"]=100.5
    assert build_trade_plan(df,**args)["action"]=="wait"


def test_future_htf_obstacle_does_not_change_exit():
    df,args=inputs();_,f=fixture();args["mtfa"]=deepcopy(f["mtfa"])
    args["mtfa"]["htf_zones"].append(dict(direction="bearish",source="order_block",timeframe="1h",
        bottom=105,top=106,available_at="2026-01-01T00:00:00Z"))
    assert build_trade_plan(df,**args)["take_profit"]==110


@pytest.mark.parametrize("cost",[-1,float("nan"),float("inf"),10000])
def test_invalid_fee_configuration_fails_explicitly(cost):
    df,args=inputs()
    with pytest.raises(ValueError):build_trade_plan(df,**args,fee_bps_per_side=cost)


def test_optional_stop_floor_remains_separate_and_default_off():
    df,args=inputs();default=build_trade_plan(df,**args)
    assert all(a["group"]!="minimum_stop_distance" for a in default["chart_evidence"])
    plan=build_trade_plan(df,**args,cost_policy="minimum_stop_bps",minimum_stop_bps=1000)
    assert plan["action"]=="wait" and "Stop distance" in plan["reason"]
