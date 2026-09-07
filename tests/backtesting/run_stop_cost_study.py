"""Predeclared minimum-stop study with frozen old and evidence-v2 planners.

Run development first, freeze one shared threshold from old-planner results,
then pass that JSON to a separate temporal test run. No indicator policy is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from core.config.mtfa_ladder import get_htf_chain
from tests.backtesting.run_trade_plans import download_month,resample_closed,detect,simulate
from tests.backtesting.legacy_trade_plan import build_trade_plan as old_plan
from tests.backtesting.evidence_v2_trade_plan import build_trade_plan as new_plan
from tests.backtesting.evidence_v2_setup import higher_timeframe_zones

DEV_THRESHOLDS=(0,12,18,24,30,40,50)


def frames_for(symbol,start,end,cache):
    records=[];hourly=[];lower=[]
    for month in pd.period_range((start-pd.Timedelta(days=300)).strftime("%Y-%m"),end.strftime("%Y-%m"),freq="M"):
        df,item=download_month(symbol,"1h",str(month),cache);hourly.append(df);records.append(item)
    for month in pd.period_range((start-pd.Timedelta(days=7)).strftime("%Y-%m"),end.strftime("%Y-%m"),freq="M"):
        df,item=download_month(symbol,"5m",str(month),cache);lower.append(df);records.append(item)
    hourly=pd.concat(hourly).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    lower=pd.concat(lower).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return {"5m":lower,"15m":resample_closed(lower,"15min",3),"1h":hourly,
        "4h":resample_closed(hourly,"4h",4),"1d":resample_closed(hourly,"1D",24)},records


def worker(job):
    symbol,cache_path,out_path,start,end,stride,thresholds,window=job
    logging.disable(logging.CRITICAL);cache=Path(cache_path);out=Path(out_path)
    start,end=pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC")
    frames,manifests=frames_for(symbol,start,end,cache)
    durations={"5m":pd.Timedelta(minutes=5),"15m":pd.Timedelta(minutes=15),"1h":pd.Timedelta(hours=1),
               "4h":pd.Timedelta(hours=4),"1d":pd.Timedelta(days=1)}
    trades=[];counts=[]
    for tf in ("5m","15m"):
        frame=frames[tf];busy={};stats={};htf_cache={}
        selected=frame.index[(frame.timestamp+durations[tf]>=start)&
                             (frame.timestamp+109*durations[tf]<=end)][::stride]
        variants=[(planner,float(floor)) for planner in ("old","new") for floor in thresholds]
        for v in variants:
            busy[v]=-1;stats[v]={"decisions":0,"plans":0,"cost_rejected":0,"unfilled":0,"filled":0}
        for number,i in enumerate(selected):
            now=frame.timestamp.iloc[i]+durations[tf]
            df=frame.iloc[max(0,i-999):i+1].reset_index(drop=True)
            future=frame.iloc[i+1:i+109]
            if len(df)<100 or len(future)<108 or (df.timestamp.diff().dropna()!=durations[tf]).any() or (future.timestamp.diff().dropna()!=durations[tf]).any():continue
            facts=detect(df,tf)
            mtfa={"enabled":True,"context":"mtfa","htf_requested":get_htf_chain(tf),
                  "htf_trends":{},"htf_unavailable":{},"htf_zones":[]}
            for higher in mtfa["htf_requested"]:
                history=frames[higher];limit=history.timestamp.searchsorted(now-durations[higher],side="right");key=(higher,int(limit))
                if key not in htf_cache:
                    h=history.iloc[max(0,limit-1000):limit].reset_index(drop=True);hf=detect(h,higher,full=False)
                    htf_cache[key]=(hf["structure"].trend,higher_timeframe_zones(h,higher,hf["order_blocks"],hf["fvg"]))
                trend,zones=htf_cache[key];mtfa["htf_trends"][higher]=trend;mtfa["htf_zones"].extend(zones)
            plans={"old":old_plan(df,interval=tf,mtfa=mtfa,**{k:v for k,v in facts.items() if k!="sweeps"}),
                   "new":new_plan(df,interval=tf,mtfa=mtfa,exit_policy="single",**facts)}
            for variant in variants:
                planner,floor=variant
                if i<=busy[variant]:continue
                state=stats[variant];state["decisions"]+=1;plan=plans[planner]
                if plan["action"]=="wait":continue
                state["plans"]+=1
                stop_bps=abs(plan["entry_level"]-plan["stop_loss"])/plan["entry_level"]*10000
                if stop_bps+1e-12<floor:
                    state["cost_rejected"]+=1;continue
                outcome=simulate(plan,future)
                if outcome is None:
                    state["unfilled"]+=1;busy[variant]=i+12;continue
                state["filled"]+=1;busy[variant]=i+outcome["bars"]
                target_bps=abs(plan["take_profit"]-plan["entry_level"])/plan["entry_level"]*10000
                trades.append(dict(window=window,symbol=symbol,tf=tf,planner=planner,min_stop_bps=floor,
                    timestamp=now.isoformat(),direction=plan["action"],entry=plan["entry_level"],stop=plan["stop_loss"],
                    target=plan["take_profit"],stop_bps=stop_bps,target_bps=target_bps,
                    modeled_round_trip_cost_r=(plan["entry_level"]+plan["take_profit"])*12/10000/abs(plan["entry_level"]-plan["stop_loss"]),**outcome))
            if number%1000==0:print(f"{window} {symbol} {tf}: {number}/{len(selected)}",flush=True)
        counts.extend(dict(window=window,symbol=symbol,tf=tf,planner=p,min_stop_bps=f,**stats[(p,f)]) for p,f in variants)
    checkpoint={"trades":trades,"counts":counts,"manifests":manifests}
    (out/f"checkpoint-{window}-{symbol}.json").write_text(json.dumps(checkpoint))
    return checkpoint


def summarize(results,out,window):
    trades=pd.DataFrame([row for result in results for row in result["trades"]])
    counts=pd.DataFrame([row for result in results for row in result["counts"]])
    trades.to_csv(out/f"{window}-trades.csv",index=False);counts.to_csv(out/f"{window}-counts.csv",index=False)
    (out/f"{window}-manifests.json").write_text(json.dumps([m for r in results for m in r["manifests"]],indent=2))
    if trades.empty:return pd.DataFrame()
    trades["win"]=trades.r>0
    summary=trades.groupby(["planner","min_stop_bps"]).agg(n=("r","size"),win_rate=("win","mean"),
        avg_net_r=("r","mean"),avg_gross_r=("gross_r","mean"),median_stop_bps=("stop_bps","median"))
    summary.to_csv(out/f"{window}-summary.csv")
    relationships=[];quartiles=[]
    for planner,group in trades[trades.min_stop_bps==0].groupby("planner"):
        relationships.append(dict(planner=planner,n=len(group),minimum_stop_bps=group.stop_bps.min(),
            median_stop_bps=group.stop_bps.median(),maximum_stop_bps=group.stop_bps.max(),
            pearson_stop_bps_net_r=group.stop_bps.corr(group.r) if len(group)>2 else None,
            spearman_stop_bps_net_r=group.stop_bps.rank().corr(group.r.rank()) if len(group)>2 else None))
        if len(group)>3:
            ranked=group.assign(quartile=pd.qcut(group.stop_bps,4,duplicates="drop"))
            for label,part in ranked.groupby("quartile",observed=True):
                quartiles.append(dict(planner=planner,bin=str(label),n=len(part),stop_bps_min=part.stop_bps.min(),
                    stop_bps_max=part.stop_bps.max(),mean_net_r=part.r.mean(),mean_gross_r=part.gross_r.mean()))
    pd.DataFrame(relationships).to_csv(out/f"{window}-stop-relationship.csv",index=False)
    pd.DataFrame(quartiles).to_csv(out/f"{window}-stop-quartiles.csv",index=False)
    return summary.reset_index()


def freeze(dev,out):
    old=dev[dev.planner=="old"].sort_values("min_stop_bps")
    base=old[old.min_stop_bps==0].iloc[0]
    eligible=old[(old.min_stop_bps>0)&(old.n>=30)&(old.n>=.25*base.n)&(old.avg_net_r>=base.avg_net_r+.25)]
    chosen=eligible.sort_values(["avg_net_r","n"],ascending=[False,False]).iloc[0] if len(eligible) else base
    policy={"policy":"minimum_stop_bps","threshold_bps":float(chosen.min_stop_bps),
            "selection_source":"old planner development only","candidates_bps":list(DEV_THRESHOLDS),
            "criterion":"n>=30, retention>=25%, mean net R improves baseline by >=0.25; maximize mean R then N",
            "baseline_n":int(base.n),"selected_n":int(chosen.n),"baseline_avg_net_r":float(base.avg_net_r),
            "selected_avg_net_r":float(chosen.avg_net_r),
            "development_candidate":bool(chosen.min_stop_bps>0),
            "production_default":False}
    (out/"frozen-stop-policy.json").write_text(json.dumps(policy,indent=2));return policy


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["dev","test"],required=True)
    p.add_argument("--start",required=True);p.add_argument("--end",required=True);p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    p.add_argument("--symbols",nargs="+",default=["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","LTCUSDT"])
    p.add_argument("--stride",type=int,default=2);p.add_argument("--workers",type=int,default=3);p.add_argument("--summarize-only",action="store_true")
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.phase=="dev":thresholds=DEV_THRESHOLDS
    else:
        policy=json.loads((out/"frozen-stop-policy.json").read_text());thresholds=tuple(sorted({0,float(policy["threshold_bps"])}))
    jobs=[(s,args.cache,args.output,args.start,args.end,args.stride,thresholds,args.phase) for s in args.symbols]
    if args.summarize_only:results=[json.loads((out/f"checkpoint-{args.phase}-{s}.json").read_text()) for s in args.symbols]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(worker,jobs))
    summary=summarize(results,out,args.phase)
    root=Path(__file__).resolve().parents[2]
    config={**vars(args),"thresholds":thresholds,"fee_bps_per_side":10,"slippage_bps_per_side":2,
        "expiry_bars":12,"hold_bars":96,"history_bars":1000,
        "source_sha256":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [*root.glob("src/core/engines/*.py"),*root.glob("src/core/config/*.py"),Path(__file__),
             Path(__file__).with_name("run_trade_plans.py"),Path(__file__).with_name("legacy_trade_plan.py"),
             Path(__file__).with_name("evidence_v2_trade_plan.py"),Path(__file__).with_name("evidence_v2_setup.py")]}}
    (out/f"{args.phase}-config.json").write_text(json.dumps(config,indent=2))
    if args.phase=="dev":print(json.dumps(freeze(summary,out),indent=2))
    else:print(summary.to_string(index=False))


if __name__=="__main__":main()
