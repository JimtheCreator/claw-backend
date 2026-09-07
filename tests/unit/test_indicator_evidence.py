from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from tests.unit.test_setup_evidence import fixture
from core.use_cases.market_analysis.setup_evidence import rank_entry_zones, indicator_evidence
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from tests.backtesting.evidence_v2_trade_plan import build_trade_plan as frozen


def indicators():
    return dict(vwap=NS(points=[NS(index=29,vwap=100)]),
                volume_profile=NS(profile_available=True,poc_price=100),
                tsmom=NS(signal_available=False),
                cvd=NS(points=[NS(index=25,delta=50,delta_source="candle_direction_approximation")]))


def test_missing_and_proxy_evidence_never_award_points():
    df,facts=fixture()
    evidence=indicator_evidence(df,"bullish",break_index=25,**indicators())
    assert evidence["vwap_side"]["passed"] is True
    assert evidence["volume_profile_side"]["passed"] is True
    assert evidence["cvd_break_confirmation"]["passed"] is None
    assert evidence["tsmom_alignment"]["passed"] is None
    assert evidence["no_opposing_divergence"]["passed"] is None


def test_frozen_baseline_hashes_are_unchanged():
    from pathlib import Path
    import hashlib
    root=Path(__file__).resolve().parents[1]/"backtesting"
    for name,digest in {
        "evidence_v2_setup.py":"21e74b8f6fc9755f41f3188233c41d3246bef9892c25460e8e5027d94e998b49",
        "evidence_v2_trade_plan.py":"2db437a0b9b5b5dce27660f7cdbaac4a9d034194b5fbc529c1e6ab30b18c861e",
    }.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest


def test_real_delta_is_directional_and_break_specific():
    df,_=fixture();data=indicators()
    data["cvd"].points[0].delta_source="taker_buy_volume"
    assert indicator_evidence(df,"bullish",break_index=25,**data)["cvd_break_confirmation"]["passed"] is True
    assert indicator_evidence(df,"bearish",break_index=25,**data)["cvd_break_confirmation"]["passed"] is False
    assert indicator_evidence(df,"bullish",break_index=24,**data)["cvd_break_confirmation"]["passed"] is None


def test_experimental_off_requires_two_positive_checks():
    df,facts=fixture();facts["mtfa"]["enabled"]=False
    data=indicators()
    assert rank_entry_zones(df,"bullish",**facts,**data,evidence_policy="indicators_v1")[0]["eligible"]
    data["volume_profile"]=None
    failed=rank_entry_zones(df,"bullish",**facts,**data,evidence_policy="indicators_v1")[0]
    assert not failed["eligible"]
    assert {a["group"] for a in failed["annotations"]}==set(failed["groups"])|{"standalone_confirmation_count"}
    gate=next(a for a in failed["annotations"] if a["group"]=="standalone_confirmation_count")
    assert gate["status"]=="failed" and gate["mandatory"]
    assert rank_entry_zones(df,"bullish",**facts,**data)[0]["eligible"]


@pytest.mark.parametrize("policy",["smc_v2","indicators_v1"])
def test_on_then_off_strips_stale_context_without_mutating_request(policy):
    df,facts=fixture()
    args=dict(interval="5m",premium_discount=NS(range_available=True,top=120,bottom=90),
              liquidity=NS(pools=[NS(side="buy_side",level=108,last_index=2)]))
    on=build_trade_plan(df,**facts,**args,**indicators(),evidence_policy=policy)
    assert on["selected_htf_poi"] is not None
    facts["mtfa"]["enabled"]=False
    stale=deepcopy(facts["mtfa"])
    off=build_trade_plan(df,**facts,**args,**indicators(),evidence_policy=policy)
    clean=build_trade_plan(df,**{**facts,"mtfa":{"enabled":False}},**args,**indicators(),evidence_policy=policy)
    assert off==clean
    assert facts["mtfa"]==stale
    assert off["selected_htf_poi"] is None
    assert off["evidence"]["mtfa"]=={"enabled":False,"context":"disabled"}
    assert "htf_poi_reaction" not in off["setup_quality"]["groups"]
    assert all(e["kind"]!="HTF POI" for e in off["chart_evidence"])
    assert on["selected_htf_poi"] is not None


def test_default_execution_matches_frozen_baseline():
    df,facts=fixture()
    args=dict(interval="5m",premium_discount=NS(range_available=True,top=120,bottom=90),
              liquidity=NS(pools=[NS(side="buy_side",level=108,last_index=2)]))
    old=frozen(df,**facts,**args)
    new=build_trade_plan(df,**facts,**args,**indicators())
    for key in ("action","entry_zone","entry_level","stop_loss","take_profit","targets","management"):
        assert old[key]==new[key]


def test_minimum_stop_policy_is_separate_visible_and_opt_in():
    df,facts=fixture()
    args=dict(interval="5m",premium_discount=NS(range_available=True,top=120,bottom=90),
              liquidity=NS(pools=[NS(side="buy_side",level=108,last_index=2)]),**facts)
    default=build_trade_plan(df,**args)
    assert default["action"]=="long"
    assert not any(e.get("group")=="minimum_stop_distance" for e in default["chart_evidence"])
    passed=build_trade_plan(df,**args,cost_policy="minimum_stop_bps",minimum_stop_bps=100)
    fact=next(e for e in passed["chart_evidence"] if e["group"]=="minimum_stop_distance")
    assert passed["action"]=="long" and fact["status"]=="passed" and fact["mandatory"]
    failed=build_trade_plan(df,**args,cost_policy="minimum_stop_bps",minimum_stop_bps=200)
    fact=next(e for e in failed["chart_evidence"] if e["group"]=="minimum_stop_distance")
    assert failed["action"]=="wait" and fact["status"]=="failed"
    assert "< 200.00 bps" in fact["label"]
    assert "cost floor" in failed["reason"] and failed["primary_scenario"] is None


def test_mixed_real_and_missing_taker_volume_is_not_fake_selling():
    from core.engines.cvd_engine import CVDEngine
    df,_=fixture()
    df["taker_buy_volume"]=0.75
    df.loc[25,"taker_buy_volume"]=float("nan")
    df.loc[26,"taker_buy_volume"]=2.0
    result=CVDEngine("5m").calculate_cvd(df)
    assert result.points[24].delta_source=="taker_buy_volume"
    assert result.points[24].delta==0.5
    assert result.points[25].delta_source=="candle_direction_approximation"
    assert result.points[26].delta_source=="candle_direction_approximation"


def test_divergence_confirmation_is_causal_and_separate():
    import pandas as pd
    df,_=fixture()
    df=pd.concat([df,df.iloc[-5:]],ignore_index=True)
    result=NS(latest_rsi=55,latest_macd_histogram=1,events=[])
    kwargs=dict(divergence=result)
    assert indicator_evidence(df,"bullish",**kwargs)["no_opposing_divergence"]["passed"] is True
    result.events=[NS(direction="bearish",second_swing_index=33)]
    assert indicator_evidence(df,"bullish",**kwargs)["no_opposing_divergence"]["passed"] is True
    result.events[0].second_swing_index=32
    assert indicator_evidence(df,"bullish",**kwargs)["no_opposing_divergence"]["passed"] is False
