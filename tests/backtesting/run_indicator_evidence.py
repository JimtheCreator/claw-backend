"""Frozen v2 versus opt-in indicator policy, MTFA on AND off.

No tuning: report separate development/test windows and each feature on the
frozen baseline's actual fills. Missing features are not imputed as failures.
Uses cached checksum-verified Binance history and the existing cost simulator.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from tests.backtesting.run_trade_plans import download_month, resample_closed, detect, simulate
from tests.backtesting.evidence_v2_trade_plan import build_trade_plan as baseline
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from core.use_cases.market_analysis.setup_evidence import higher_timeframe_zones, rank_entry_zones
from core.engines.vwap_engine import VWAPEngine
from core.engines.volume_profile_engine import VolumeProfileEngine
from core.engines.rsi_macd_divergence_engine import RSIMACDDivergenceEngine
from core.engines.cvd_engine import CVDEngine
from core.engines.tsmom_engine import TSMOMEngine


def run(job):
    symbol, cache, output, start, end, stride = job
    logging.disable(logging.CRITICAL)
    cache=Path(cache); manifests=[]
    start,end=pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC")
    def load(tf,days):
        chunks=[]
        for month in pd.period_range((start-pd.Timedelta(days=days)).strftime("%Y-%m"),end.strftime("%Y-%m"),freq="M"):
            data,manifest=download_month(symbol,tf,str(month),cache)
            chunks.append(data);manifests.append(manifest)
        return pd.concat(chunks).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    hourly,lower=load("1h",300),load("5m",7)
    frames={"5m":lower,"15m":resample_closed(lower,"15min",3),"1h":hourly,
            "4h":resample_closed(hourly,"4h",4),"1d":resample_closed(hourly,"1D",24)}
    durations={k:pd.Timedelta(v) for k,v in {"5m":"5min","15m":"15min","1h":"1h","4h":"4h","1d":"1d"}.items()}
    rows=[];counts=[]
    for tf in ("5m","15m"):
        frame=frames[tf];cache_htf={};busy={};counter={}
        # Independent windows: all possible order/holding bars must fit before
        # the window boundary. No development outcome consumes test prices.
        indices=frame.index[(frame.timestamp+durations[tf]>=start)&
                            (frame.timestamp+109*durations[tf]<=end)][::stride]
        for number,i in enumerate(indices):
            now=frame.timestamp.iloc[i]+durations[tf]
            df=frame.iloc[max(0,i-999):i+1].reset_index(drop=True)
            future=frame.iloc[i+1:i+109]
            if len(df)<100 or len(future)<108 or (df.timestamp.diff().dropna()!=durations[tf]).any() or (future.timestamp.diff().dropna()!=durations[tf]).any():continue
            facts=detect(df,tf)
            mtfa={"enabled":True,"context":"mtfa","htf_requested":["1h","4h","1d"],"htf_trends":{},"htf_zones":[]}
            for ht in mtfa["htf_requested"]:
                h=frames[ht];limit=h.timestamp.searchsorted(now-durations[ht],side="right");key=(ht,int(limit))
                if key not in cache_htf:
                    prefix=h.iloc[max(0,limit-1000):limit].reset_index(drop=True)
                    hf=detect(prefix,ht,full=False)
                    cache_htf[key]=(hf["structure"].trend,higher_timeframe_zones(prefix,ht,hf["order_blocks"],hf["fvg"]))
                trend,zones=cache_htf[key];mtfa["htf_trends"][ht]=trend;mtfa["htf_zones"].extend(zones)
            indicators=None
            for enabled in (True,False):
                context=mtfa if enabled else {"enabled":False,"context":"disabled"}
                old=baseline(df,interval=tf,mtfa=context,**facts)
                # With no SMC-eligible candidate no added filter can rescue it.
                # Still evaluate SMC-eligible WAITs: ranking can select another zone.
                if not old.get("setup_quality",{}).get("eligible"):continue
                if indicators is None:
                    indicators=dict(vwap=VWAPEngine(tf).calculate_vwap(df),
                        volume_profile=VolumeProfileEngine(tf).calculate_profile(df),
                        divergence=RSIMACDDivergenceEngine(tf).detect_divergence(df),
                        cvd=CVDEngine(tf).calculate_cvd(df),tsmom=TSMOMEngine(tf).calculate_signal(df))
                new=build_trade_plan(df,interval=tf,mtfa=context,**facts,**indicators,evidence_policy="indicators_v1")
                ranked=rank_entry_zones(df,old["trend_direction"],mtfa=context,**indicators,
                    **{k:v for k,v in facts.items() if k not in {"liquidity","premium_discount"}})
                zone=old.get("entry_zone")
                match=next((z for z in ranked if zone and z["bottom"]==zone["bottom"] and z["top"]==zone["top"]),None)
                for version,plan in (("frozen_v2",old),("indicators_v1",new)):
                    key=(enabled,version)
                    if i<=busy.get(key,-1) or plan["action"]=="wait":continue
                    counter[key]=counter.get(key,0)+1
                    outcome=simulate(plan,future)
                    busy[key]=i+(outcome["bars"] if outcome else 12)
                    if outcome is None:continue
                    quality=match["indicator_evidence"] if version=="frozen_v2" else plan["setup_quality"]["indicator_evidence"]
                    rows.append(dict(symbol=symbol,tf=tf,mtfa=enabled,version=version,timestamp=now.isoformat(),
                        **outcome,**{k:v["passed"] for k,v in quality.items()}))
            if number%500==0:print(f"{symbol} {tf} {number}/{len(indices)}",flush=True)
        counts.extend(dict(symbol=symbol,tf=tf,mtfa=k[0],version=k[1],plans=v) for k,v in counter.items())
    result=(rows,manifests,counts)
    checkpoint=Path(output)/f"checkpoint-{symbol}-{start.date()}-{end.date()}.json"
    checkpoint.write_text(json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    p.add_argument("--start",required=True);p.add_argument("--end",required=True)
    p.add_argument("--split",required=True);p.add_argument("--stride",type=int,default=3)
    p.add_argument("--symbols",nargs="+",default=["BTCUSDT","ETHUSDT","SOLUSDT"])
    p.add_argument("--summarize-only",action="store_true",help="Rebuild reports from completed per-window checkpoints")
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    config=vars(args).copy()
    config.update(fee_bps=10,slippage_bps=2,bars=1000,hold_bars=96,expiry_bars=12,
                  limitation="Existing historical windows reused; test is a temporal separation, not untouched confirmation history.")
    config["source_sha256"]={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
        for pattern in ("src/core/engines/*.py","src/core/use_cases/market_analysis/*.py","tests/backtesting/*.py") for f in root.glob(pattern)}
    jobs=[(s,args.cache,args.output,start,end,args.stride) for start,end in
          ((args.start,args.split),(args.split,args.end)) for s in args.symbols]
    if args.summarize_only:
        results=[json.loads((out/f"checkpoint-{s}-{start}-{end}.json").read_text()) for s,_,_,start,end,_ in jobs]
    else:
        (out/"config.json").write_text(json.dumps(config,indent=2))
        with ProcessPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(run,jobs))
    trades=pd.DataFrame([r for rows,_,_ in results for r in rows])
    trades.to_csv(out/"trades.csv",index=False)
    (out/"manifests.json").write_text(json.dumps([m for _,ms,_ in results for m in ms],indent=2))
    (out/"counts.json").write_text(json.dumps([c for _,_,cs in results for c in cs],indent=2))
    if trades.empty:return
    trades["window"]=trades.timestamp.apply(lambda t:"dev" if t<args.split else "test")
    trades["win"]=trades.r>0
    summary=trades.groupby(["window","mtfa","version"]).agg(n=("r","size"),win_rate=("win","mean"),avg_r=("r","mean"))
    grid=pd.MultiIndex.from_product([["dev","test"],[False,True],["frozen_v2","indicators_v1"]],names=["window","mtfa","version"])
    summary=summary.reindex(grid)
    summary["n"]=summary.n.fillna(0).astype(int)
    summary.to_csv(out/"summary.csv")
    features=["vwap_side","volume_profile_side","tsmom_alignment","cvd_break_confirmation","no_opposing_divergence"]
    report=[]
    for (window,enabled),cohort in trades[trades.version=="frozen_v2"].groupby(["window","mtfa"]):
        for feature in features:
            valid=cohort.dropna(subset=[feature])
            corr=valid[feature].astype(float).corr(valid.r) if len(valid)>2 and valid[feature].nunique()>1 and valid.r.nunique()>1 else None
            for state in (True,False,None):
                group=cohort[cohort[feature].isna()] if state is None else cohort[cohort[feature]==state]
                report.append(dict(window=window,mtfa=enabled,feature=feature,state="unavailable" if state is None else state,
                    n=len(group),avg_r=group.r.mean(),win_rate=group.win.mean(),point_biserial_r=corr))
    pd.DataFrame(report).to_csv(out/"per-feature.csv",index=False)
    print((out/"summary.csv").read_text(),flush=True)


if __name__=="__main__":main()
