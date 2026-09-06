"""Verify recorded entries, score THEIR zones, and compare exits on matched entries.

Run after run_trade_plans; counters/portfolio simulations remain distinct from
matched-entry counterfactuals. Writes reports, never mutates cached source data.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tests.backtesting.run_trade_plans import download_month, resample_closed, detect, simulate, old_plan, build_trade_plan
from core.use_cases.market_analysis.setup_evidence import higher_timeframe_zones, rank_entry_zones, staged_targets


def frames_for(symbol, path, cache):
    grouped={"1h":[],"5m":[]}
    for record in json.loads((path/f"{symbol}-manifest.json").read_text()):
        filename=record["url"].rsplit("/",1)[-1]
        tf,month=filename[len(symbol)+1:-4].split("-",1)
        grouped[tf].append(download_month(symbol,tf,month,cache)[0])
    data={tf:pd.concat(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
          for tf,rows in grouped.items()}
    data["15m"]=resample_closed(data["5m"],"15min",3)
    data["4h"]=resample_closed(data["1h"],"4h",4)
    data["1d"]=resample_closed(data["1h"],"1D",24)
    return data


def rebuild(frames, tf, now, window):
    delta={"5m":pd.Timedelta(minutes=5),"15m":pd.Timedelta(minutes=15),"1h":pd.Timedelta(hours=1),
           "4h":pd.Timedelta(hours=4),"1d":pd.Timedelta(days=1)}
    frame=frames[tf]
    index=frame.timestamp.searchsorted(now-delta[tf],side="right")
    df=frame.iloc[max(0,index-window):index].reset_index(drop=True)
    mtfa=dict(enabled=True,context="mtfa",htf_requested=["1h","4h","1d"],htf_trends={},htf_unavailable={},htf_zones=[])
    for higher in mtfa["htf_requested"]:
        h=frames[higher]
        h=h[h.timestamp+delta[higher]<=now].tail(window).reset_index(drop=True)
        facts=detect(h,higher,full=False)
        mtfa["htf_trends"][higher]=facts["structure"].trend
        mtfa["htf_zones"].extend(higher_timeframe_zones(h,higher,facts["order_blocks"],facts["fvg"]))
    return df,mtfa,detect(df,tf),frame.iloc[index:index+108]


def metrics(rows):
    n=len(rows)
    if not n:
        return {"trades":0,"win_rate":None,"avg_r":None,"mean_r_day_bootstrap_95":None}
    z=1.96;rate=float((rows.r>0).mean())
    center=(rate+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    daily=rows.assign(day=pd.to_datetime(rows.timestamp,utc=True).dt.date).groupby("day").r.agg(["sum","count"])
    rng=np.random.default_rng(20250906)
    idx=rng.integers(0,len(daily),size=(2000,len(daily)))
    means=daily["sum"].to_numpy()[idx].sum(axis=1)/daily["count"].to_numpy()[idx].sum(axis=1)
    return {"trades":n,"win_rate":rate,"win_rate_wilson_95":[float(center-half),float(center+half)],
            "avg_r":float(rows.r.mean()),"avg_gross_r":float(rows.gross_r.mean()),
            "losing_trade_rate":float((rows.r<0).mean()),
            "mean_r_day_bootstrap_95":np.quantile(means,[.025,.975]).tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("output");parser.add_argument("cache");args=parser.parse_args()
    logging.disable(logging.CRITICAL)
    path,cache=Path(args.output),Path(args.cache)
    config=json.loads((path/"config.json").read_text())
    recorded=pd.read_csv(path/"trades.csv")
    pairs,costs,corrected=[],[],[]
    for symbol,subset in recorded.groupby("symbol"):
        frames=frames_for(symbol,path,cache)
        cached={}
        for row in subset.to_dict("records"):
            key=(row["tf"],row["timestamp"])
            if key not in cached:
                cached[key]=rebuild(frames,row["tf"],pd.Timestamp(row["timestamp"]),config["bars"])
            df,mtfa,facts,future=cached[key]
            baseline=row["version"].startswith("old")
            plan=(old_plan(df,interval=row["tf"],mtfa=mtfa,**{k:v for k,v in facts.items() if k!="sweeps"}) if baseline else
                  build_trade_plan(df,interval=row["tf"],mtfa=mtfa,exit_policy="staged",**facts))
            if plan["action"]=="wait" or abs(plan["entry_level"]-row["entry"])>1e-7:
                raise AssertionError(f"Non-reproducible historical plan: {key}")
            ranked=rank_entry_zones(df,plan["trend_direction"],mtfa=mtfa,
                                    **{k:v for k,v in facts.items() if k not in {"liquidity","premium_discount"}})
            zone=plan["entry_zone"]
            match=next((z for z in ranked if abs(z["bottom"]-zone["bottom"])<1e-7
                        and abs(z["top"]-zone["top"])<1e-7),None)
            row["score"]=match["score"] if match else None
            corrected.append(row)
            if row["version"] not in {"old","new_single"}:
                continue
            staged=staged_targets(plan["trend_direction"],plan["entry_level"],plan["stop_loss"],facts["liquidity"],df)
            if not staged or staged[0]["price"]!=plan["take_profit"]:
                staged=[{"price":plan["take_profit"],"fraction":1.0}]
            plan["targets"]=staged
            for mode in ("single","staged_be","staged_no_be"):
                result=simulate(plan,future,staged=mode!="single",breakeven=mode=="staged_be")
                assert result is not None
                pairs.append({"symbol":symbol,"tf":row["tf"],"timestamp":row["timestamp"],"selection":row["version"],
                              "mode":mode,"target_count":len(staged),"score":row["score"],**result})
            for fee,slip in ((0,0),(4,2),(10,2),(10,5)):
                result=simulate(plan,future,fee_bps=fee,slippage_bps=slip)
                costs.append({"selection":row["version"],"fee_bps":fee,"slippage_bps":slip,"r":result["r"]})
    verified=pd.DataFrame(corrected)
    verified.to_csv(path/"verified-trades.csv",index=False)
    verified.assign(win=verified.r>0).groupby(["version","score"]).agg(
        trades=("r","size"),win_rate=("win","mean"),avg_r=("r","mean")).to_csv(path/"by-score.csv")
    root=Path(__file__).resolve().parents[2]
    sources=[*root.glob("src/core/engines/*.py"),*root.glob("src/core/use_cases/market_analysis/*.py"),Path(__file__)]
    (path/"verification-source-hashes.json").write_text(json.dumps(
        {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},indent=2))
    pd.DataFrame(pairs).to_csv(path/"matched-exits.csv",index=False)
    pd.DataFrame(costs).groupby(["selection","fee_bps","slippage_bps"]).r.agg(["size","mean"]).to_csv(path/"cost-sensitivity.csv")
    versions=sorted({c["version"] for c in json.loads((path/"counts.json").read_text())})
    report={v:metrics(verified[verified.version==v]) for v in versions}
    quality={}
    for version in ("old","new_single"):
        rows=verified[verified.version==version].dropna(subset="score")
        correlation=rows.score.rank().corr(rows.r.rank()) if len(rows)>=3 and rows.score.nunique()>1 else None
        quality[version]={"spearman_score_net_r":float(correlation) if correlation is not None and np.isfinite(correlation) else None,
                          "bins":{str(s):metrics(g) for s,g in rows.groupby("score")}}
    (path/"verified-summary.json").write_text(json.dumps({"portfolio_variants":report,"quality":quality},indent=2))
    paired=pd.DataFrame(pairs)
    paired.groupby(["selection","mode"]).agg(trades=("r","size"),avg_r=("r","mean")).to_csv(path/"matched-summary.csv")
    print(json.dumps({"portfolio_variants":report,"quality":quality},indent=2))


if __name__=="__main__":main()
