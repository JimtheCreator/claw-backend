"""Shadow-only CVD/TSMOM measurements with richer historical inputs.

Does NOT change the deployable-input strategy comparison. Rebuilds every
frozen baseline fill and verifies its R before attributing research features.
"""
import argparse
import hashlib
import io
import json
import logging
from pathlib import Path
import zipfile

import pandas as pd

from tests.backtesting.run_trade_plans import detect, simulate, resample_closed
from tests.backtesting.evidence_v2_trade_plan import build_trade_plan
from core.use_cases.market_analysis.setup_evidence import higher_timeframe_zones, indicator_evidence
from core.engines.cvd_engine import CVDEngine
from core.engines.tsmom_engine import TSMOMEngine


def load(cache,symbol,tf):
    chunks=[]
    for path in sorted(cache.glob(f"{symbol}-{tf}-*.zip")):
        payload=path.read_bytes()
        assert hashlib.sha256(payload).hexdigest()==path.with_suffix(".sha256").read_text().strip()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            df=pd.read_csv(z.open(z.namelist()[0]),header=None,usecols=[0,1,2,3,4,5,9])
        df.columns=["timestamp","open","high","low","close","volume","taker_buy_volume"]
        df.timestamp=pd.to_datetime(df.timestamp,unit="us" if df.timestamp.iloc[0]>10**14 else "ms",utc=True)
        chunks.append(df)
    return pd.concat(chunks).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();cache=Path(args.cache);out=Path(args.output)
    logging.disable(logging.CRITICAL)
    config=json.loads((out/"config.json").read_text())
    trades=pd.read_csv(out/"trades.csv");records=[]
    for symbol,rows in trades[trades.version=="frozen_v2"].groupby("symbol"):
        lower=load(cache,symbol,"5m");hourly=load(cache,symbol,"1h")
        fifteen=resample_closed(lower,"15min",3)
        buys=lower.set_index("timestamp").taker_buy_volume.resample("15min").sum(min_count=3)
        fifteen["taker_buy_volume"]=fifteen.timestamp.map(buys)
        frames={"5m":lower,"15m":fifteen,"1h":hourly,"4h":resample_closed(hourly,"4h",4),"1d":resample_closed(hourly,"1d",24)}
        durations={k:pd.Timedelta(v) for k,v in {"5m":"5min","15m":"15min","1h":"1h","4h":"4h","1d":"1d"}.items()}
        for row in rows.itertuples(index=False):
            now=pd.Timestamp(row.timestamp);frame=frames[row.tf]
            limit=frame.timestamp.searchsorted(now-durations[row.tf],side="right")
            df=frame.iloc[max(0,limit-1000):limit].reset_index(drop=True)
            facts=detect(df,row.tf)
            mtfa={"enabled":bool(row.mtfa),"context":"mtfa" if row.mtfa else "disabled","htf_trends":{},"htf_zones":[]}
            if row.mtfa:
                for tf in ("1h","4h","1d"):
                    h=frames[tf];end=h.timestamp.searchsorted(now-durations[tf],side="right")
                    h=h.iloc[max(0,end-1000):end].reset_index(drop=True);hf=detect(h,tf,full=False)
                    mtfa["htf_trends"][tf]=hf["structure"].trend
                    mtfa["htf_zones"].extend(higher_timeframe_zones(h,tf,hf["order_blocks"],hf["fvg"]))
            plan=build_trade_plan(df,interval=row.tf,mtfa=mtfa,**facts)
            assert plan["action"]!="wait",(symbol,row.timestamp)
            outcome=simulate(plan,frame.iloc[limit:limit+108])
            assert outcome and abs(outcome["r"]-row.r)<1e-7,(symbol,row.timestamp,outcome,row.r)
            annotation=plan["chart_evidence"][-1]
            index=int(df.index[pd.to_datetime(df.timestamp,utc=True)==pd.Timestamp(annotation["timestamp"])][0])
            long_history=frame.iloc[:limit].reset_index(drop=True)
            momentum=TSMOMEngine(row.tf).calculate_signal(long_history)
            evidence=indicator_evidence(df,plan["trend_direction"],break_index=index,
                cvd=CVDEngine(row.tf).calculate_cvd(df),tsmom=momentum)
            records.append(dict(symbol=symbol,tf=row.tf,mtfa=row.mtfa,timestamp=row.timestamp,
                window="dev" if row.timestamp<config["split"] else "test",r=row.r,
                cvd_break_confirmation=evidence["cvd_break_confirmation"]["passed"],
                tsmom_alignment=evidence["tsmom_alignment"]["passed"],history_bars=len(long_history),
                horizons="/".join(str(h.lookback_bars) for h in momentum.horizons)))
    result=pd.DataFrame(records);result.to_csv(out/"richer-input-shadow-trades.csv",index=False)
    report=[]
    for (window,mtfa),cohort in result.groupby(["window","mtfa"]):
        for feature in ("cvd_break_confirmation","tsmom_alignment"):
            valid=cohort.dropna(subset=[feature])
            corr=valid[feature].astype(float).corr(valid.r) if len(valid)>2 and valid[feature].nunique()>1 else None
            for state in (True,False,None):
                group=cohort[cohort[feature].isna()] if state is None else cohort[cohort[feature]==state]
                report.append(dict(window=window,mtfa=mtfa,feature=feature,state="unavailable" if state is None else state,
                    n=len(group),avg_r=group.r.mean(),point_biserial_r=corr))
    pd.DataFrame(report).to_csv(out/"richer-input-shadow-features.csv",index=False)
    (out/"shadow-source.sha256").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(pd.DataFrame(report).to_string(index=False))


if __name__=="__main__":main()
