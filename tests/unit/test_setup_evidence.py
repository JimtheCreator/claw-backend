from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core.use_cases.market_analysis.setup_evidence import rank_entry_zones, higher_timeframe_zones, staged_targets
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from tests.backtesting.run_trade_plans import simulate, resample_closed


def fixture():
    df = pd.DataFrame(dict(timestamp=pd.date_range("2025-01-02", periods=30, freq="5min", tz="UTC"),
                           open=[100.]*30,high=[101.]*30,low=[99.]*30,close=[100.]*30,volume=[1.]*30))
    df.loc[25,["open","high","low","close"]]=[98,102,98,102]
    df.loc[26:,["open","high","low","close"]]=[103,104,102,103]
    event=NS(direction="bullish",index=25,kind="CHoCH",level=101,timestamp=df.timestamp.iloc[25])
    ob=NS(type="bullish",bottom=98,top=100,candle_index=23,breakout_index=25,mitigation_status="unmitigated")
    fvg=NS(type="bullish",bottom=99,top=100,start_index=23,formed_index=25,mitigation_status="unmitigated")
    overlap=NS(type="bullish",bottom=99,top=100,ob_candle_index=23,ob_breakout_index=25,
               fvg_mitigation_status="unmitigated",ob_mitigation_status="unmitigated")
    mtfa=dict(enabled=True,context="mtfa",htf_trends={"1h":"bullish","4h":"bullish"},
              htf_zones=[dict(timeframe="1h",direction="bullish",source="order_block",bottom=97,top=101,
                              available_at="2025-01-01T12:00:00+00:00",status="mitigated")])
    facts=dict(order_blocks=NS(zones=[ob]),fvg=NS(zones=[fvg]),confluence=NS(zones=[overlap]),
               structure=NS(trend="bullish",events=[event]),sweeps=NS(events=[]),swings=NS(window=3,swings=[]),mtfa=mtfa)
    return df,facts


def test_scores_distinct_groups_and_ranks_confluence():
    df,facts=fixture();ranked=rank_entry_zones(df,"bullish",**facts)
    assert ranked[0]["source"]=="confluence"
    assert ranked[0]["score"]==4 and ranked[0]["eligible"]
    assert len(ranked[0]["annotations"])==2
    facts["confluence"].zones *= 10
    assert rank_entry_zones(df,"bullish",**facts)[0]["score"]==4


@pytest.mark.parametrize("change",["future_poi","wrong_direction","broken_poi","weak_break","no_structure"])
def test_missing_mandatory_evidence_cannot_be_offset_by_other_points(change):
    df,facts=fixture()
    if change=="future_poi":facts["mtfa"]["htf_zones"][0]["available_at"]="2025-01-03T00:00:00Z"
    if change=="wrong_direction":facts["mtfa"]["htf_zones"][0]["direction"]="bearish"
    if change=="broken_poi":df.loc[27,"close"]=96
    if change=="weak_break":df.loc[25,"open"]=101.9
    if change=="no_structure":facts["structure"].events=[]
    assert not any(z["eligible"] for z in rank_entry_zones(df,"bullish",**facts))


def test_htf_off_has_no_hidden_poi_gate():
    df,facts=fixture();facts["mtfa"]={"enabled":False}
    ranked=rank_entry_zones(df,"bullish",**facts)
    assert ranked[0]["eligible"] and "htf_poi_reaction" not in ranked[0]["groups"]


def test_order_block_available_at_break_close_not_origin():
    df,facts=fixture()
    zones=higher_timeframe_zones(df,"5m",facts["order_blocks"],facts["fvg"])
    assert pd.Timestamp(zones[0]["available_at"])==df.timestamp.iloc[25]+pd.Timedelta(minutes=5)


def test_staged_plan_has_real_levels_and_no_invented_runner():
    df,facts=fixture()
    liquidity=NS(pools=[NS(side="buy_side",level=p,last_index=2) for p in (101,108,112)])
    kwargs=dict(interval="5m",exit_policy="staged",premium_discount=NS(range_available=True,zone="premium",bottom=90,top=120),liquidity=liquidity,**facts)
    plan=build_trade_plan(df,**kwargs)
    assert plan["action"]=="long"
    assert [t["price"] for t in plan["targets"]]==[108,112]
    assert sum(t["fraction"] for t in plan["targets"])==1
    assert plan["management"]["stop_after_t1"]==plan["entry_level"]
    liquidity.pools=liquidity.pools[:2]
    plan=build_trade_plan(df,**kwargs)
    assert len(plan["targets"])==1 and plan["targets"][0]["fraction"]==1
    assert plan["management"]["stop_after_t1"] is None


def test_unproven_staged_management_is_not_the_default():
    df,facts=fixture()
    plan=build_trade_plan(df,interval="5m",premium_discount=NS(range_available=True,top=120,bottom=90),
                         liquidity=NS(pools=[NS(side="buy_side",level=p,last_index=2) for p in (108,112)]),**facts)
    assert plan["management"]["mode"]=="single"
    assert len(plan["targets"])==1 and plan["targets"][0]["fraction"]==1
    assert plan["management"]["stop_after_t1"] is None
    assert len(plan["research_exit_alternative"]["targets"])==2


def execution_plan():
    return dict(action="long",entry_level=100,stop_loss=98,take_profit=104,
                entry_zone={"bottom":99,"top":100},targets=[{"price":104,"fraction":0.5},{"price":108,"fraction":0.5}])


def bars(rows):
    return pd.DataFrame(rows,columns=["open","high","low","close"])


def test_no_same_bar_rejection_and_fill_or_favorable_target_credit():
    # Rejection at index0, fill at1. Its earlier high cannot earn target credit.
    future=bars([[100,102,99,101],[101,110,99,101],[101,102,97,98]])
    result=simulate(execution_plan(),future,fee_bps=0,slippage_bps=0)
    assert result["r"]==-1


def test_stop_wins_when_both_levels_touched_and_gap_is_not_filled_at_stop():
    future=bars([[100,102,99,101],[101,102,99,101],[97,110,96,101]])
    assert simulate(execution_plan(),future,fee_bps=0,slippage_bps=0)["r"]==-1.5


def test_partial_then_next_bar_breakeven_includes_costs():
    future=bars([[100,102,99,101],[101,102,99,101],[101,105,99,104],[104,105,99,100]])
    result=simulate(execution_plan(),future,staged=True)
    assert result["gross_r"]==1
    assert result["r"]<1 and result["exit"]=="stop"


def test_partial_bar_does_not_retroactively_move_stop():
    future=bars([[100,102,99,101],[101,102,99,101],[101,105,99,104]])
    result=simulate(execution_plan(),future,staged=True,fee_bps=0,slippage_bps=0)
    assert result["r"]==2 and result["exit"]=="time"


def test_resampling_drops_partial_or_gapped_higher_timeframes():
    df,_=fixture()
    assert len(resample_closed(df.iloc[:11],"1h",12))==0
    assert len(resample_closed(df.iloc[:12],"1h",12))==1
    assert len(resample_closed(df.drop(index=4).iloc[:11],"1h",12))==0
