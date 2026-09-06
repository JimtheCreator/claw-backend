"""Reproducible close-by-close comparison. No DB, Celery, LLM or future pivots.

Run from repo: PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_trade_plans --help
Public Binance spot history; shorts are directional paper trades (not spot execution).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import logging
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

from core.engines.swing_structure_engine import SwingStructureEngine
from core.engines.market_structure_engine import MarketStructureEngine
from core.engines.liquidity_engine import LiquidityEngine
from core.engines.liquidity_sweep_engine import LiquiditySweepEngine
from core.engines.fvg_engine import FVGEngine
from core.engines.order_block_engine import OrderBlockEngine
from core.engines.imbalance_order_block_engine import ImbalanceOrderBlockEngine
from core.engines.premium_discount_engine import PremiumDiscountEngine
from core.config.mtfa_ladder import get_htf_chain
from core.use_cases.market_analysis.setup_evidence import higher_timeframe_zones, staged_targets, rank_entry_zones
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from tests.backtesting.legacy_trade_plan import build_trade_plan as old_plan


def download_month(symbol, interval, month, cache):
    name = f"{symbol}-{interval}-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{name}"
    target = cache / name
    if not target.exists():
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read()
        with urllib.request.urlopen(url + ".CHECKSUM", timeout=45) as response:
            expected = response.read().decode().split()[0]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {url}")
        target.write_bytes(payload)
        target.with_suffix(".sha256").write_text(expected)
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != target.with_suffix(".sha256").read_text().strip():
        raise ValueError(f"Corrupt cached archive: {target}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        df = pd.read_csv(archive.open(archive.namelist()[0]), header=None, usecols=range(6))
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    unit = "us" if df.timestamp.iloc[0] > 10**14 else "ms"
    df.timestamp = pd.to_datetime(df.timestamp, unit=unit, utc=True)
    return df, {"url": url, "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(df)}


def resample_closed(frame, rule, expected):
    indexed = frame.set_index("timestamp")
    group = indexed.resample(rule, origin="epoch", label="left", closed="left")
    result = group.agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"})
    return result[group.close.count() == expected].dropna().reset_index()


def detect(df, tf, full=True):
    swings = SwingStructureEngine(tf).detect_swings(df)
    structure = MarketStructureEngine(tf).detect_structure(df, swings)
    liq = LiquidityEngine(tf).map_liquidity(df, swings)
    fvg = FVGEngine(tf).detect_fvgs(df)
    ob = OrderBlockEngine(tf).detect_order_blocks(df, structure)
    result = dict(swings=swings, structure=structure, liquidity=liq, fvg=fvg, order_blocks=ob)
    if full:
        result["sweeps"] = LiquiditySweepEngine(tf).detect_sweeps(df, liq)
        result["confluence"] = ImbalanceOrderBlockEngine(tf).find_confluence(fvg, ob)
        result["premium_discount"] = PremiumDiscountEngine(tf).calculate_zone(df, swings, structure)
    return result


def simulate(plan, future, fee_bps=10, slippage_bps=2, staged=False, breakeven=True):
    """Common operationalization of legacy's prose trigger and new retest rule.

    First CLOSED rejection bar arms entry. A LATER bar must touch the limit.
    Stop before entry cancels. Stop wins if stop and target share a bar.
    Never infer the favorable intrabar path. Price BE effective next bar.
    Expires after 12 bars; max holding 96 bars; time exit included in realized R.
    """
    sign = 1 if plan["action"] == "long" else -1
    entry, stop = plan["entry_level"], plan["stop_loss"]
    risk = abs(entry-stop)
    zone = plan["entry_zone"]
    armed, filled, remaining, gross, costs, initial_stop = False, None, 1.0, 0.0, 0.0, stop
    targets = plan.get("targets") if staged else None
    targets = targets or [{"price": plan["take_profit"], "fraction": 1.0}]
    targets = [dict(t) for t in targets]
    be_pending = False
    for i, bar in enumerate(future.itertuples(index=False)):
        if filled is None:
            if i >= 12:
                return None
            if (bar.low <= stop if sign == 1 else bar.high >= stop):
                return None
            if armed and bar.low <= entry <= bar.high:
                filled = i
                costs = entry * (fee_bps + slippage_bps) / 10000 / risk
            else:
                armed = armed or (bar.low <= zone["top"] and bar.high >= zone["bottom"]
                                 and sign*(bar.close-entry)>0 and sign*(bar.close-bar.open)>0)
                continue
        if be_pending:
            stop = entry
            be_pending = False
        hit_stop = bar.low <= stop if sign == 1 else bar.high >= stop
        if hit_stop:
            exit_price = min(stop, bar.open) if sign == 1 else max(stop, bar.open)
            gross += remaining * sign*(exit_price-entry)/risk
            costs += remaining * exit_price * (fee_bps+slippage_bps)/10000/risk
            return {"r": gross-costs, "gross_r": gross, "exit": "stop", "bars": i+1,
                    "initial_stop": initial_stop}
        # Entry-bar favorable excursions could have happened BEFORE entry.
        if i > filled:
            for t in targets:
                if t.get("filled"):
                    continue
                if bar.high >= t["price"] if sign==1 else bar.low <= t["price"]:
                    fraction = min(remaining, t["fraction"])
                    gross += fraction * sign*(t["price"]-entry)/risk
                    costs += fraction * t["price"]*(fee_bps+slippage_bps)/10000/risk
                    remaining -= fraction
                    t["filled"] = True
                    if staged and breakeven and remaining > 0:
                        be_pending = True
            if remaining <= 1e-8:
                return {"r": gross-costs, "gross_r": gross, "exit": "target", "bars": i+1,
                        "initial_stop": initial_stop}
        if i-filled >= 95 or i == len(future)-1:
            gross += remaining * sign*(bar.close-entry)/risk
            costs += remaining * bar.close*(fee_bps+slippage_bps)/10000/risk
            return {"r": gross-costs, "gross_r": gross, "exit": "time", "bars": i+1,
                    "initial_stop": initial_stop}
    return None


def run_symbol(args):
    symbol, cache_path, output_path, start, end, stride, bars = args
    logging.disable(logging.CRITICAL)
    cache, output = Path(cache_path), Path(output_path)
    cache.mkdir(parents=True, exist_ok=True)
    manifests, hourly, lower = [], [], []
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    for month in pd.period_range((start-pd.Timedelta(days=300)).strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"):
        df, manifest = download_month(symbol, "1h", str(month), cache)
        hourly.append(df); manifests.append(manifest)
    for month in pd.period_range((start-pd.Timedelta(days=7)).strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"):
        df, manifest = download_month(symbol, "5m", str(month), cache)
        lower.append(df); manifests.append(manifest)
    hourly = pd.concat(hourly).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    lower = pd.concat(lower).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frames = {"5m":lower, "15m":resample_closed(lower,"15min",3), "1h":hourly,
              "4h":resample_closed(hourly,"4h",4), "1d":resample_closed(hourly,"1D",24)}
    durations = {"5m":pd.Timedelta(minutes=5),"15m":pd.Timedelta(minutes=15),
                 "1h":pd.Timedelta(hours=1),"4h":pd.Timedelta(hours=4),"1d":pd.Timedelta(days=1)}
    trades, counts, evidence_scores = [], [], []
    for tf in ("5m", "15m"):
        frame = frames[tf]
        versions = ("old", "old_staged", "old_staged_no_be", "new_single", "new_staged", "new_staged_no_be")
        busy = {v: -1 for v in versions}
        counter = {v: {"decisions":0,"plans":0,"waits":0,"unfilled":0} for v in versions}
        htf_cache = {}
        selected = frame.index[(frame.timestamp+durations[tf]>=start)&(frame.timestamp+durations[tf]<end)][::stride]
        for number, i in enumerate(selected):
            now = frame.timestamp.iloc[i]+durations[tf]
            df = frame.iloc[max(0,i+1-bars):i+1].reset_index(drop=True)
            if len(df)<100 or (df.timestamp.diff().dropna()!=durations[tf]).any():
                continue
            facts = detect(df, tf)
            mtfa = {"enabled":True,"context":"mtfa","htf_requested":get_htf_chain(tf),
                    "htf_trends":{},"htf_unavailable":{},"htf_zones":[]}
            for higher in get_htf_chain(tf):
                history = frames[higher]
                limit = history.timestamp.searchsorted(now-durations[higher], side="right")
                key = (higher, int(limit))
                if key not in htf_cache:
                    h = history.iloc[max(0,limit-bars):limit].reset_index(drop=True)
                    hf = detect(h, higher, full=False)
                    htf_cache[key] = (hf["structure"].trend, higher_timeframe_zones(h, higher, hf["order_blocks"], hf["fvg"]))
                trend, zones = htf_cache[key]
                mtfa["htf_trends"][higher] = trend
                mtfa["htf_zones"].extend(zones)
            old_facts = {k:v for k,v in facts.items() if k != "sweeps"}
            old = old_plan(df, interval=tf, mtfa=mtfa, **old_facts)
            new = build_trade_plan(df, interval=tf, mtfa=mtfa, exit_policy="staged", **facts)
            quality = new.get("setup_quality", {})
            old_score = None
            if old["action"] != "wait":
                ranked = rank_entry_zones(df,old["trend_direction"],mtfa=mtfa,
                                         **{k:v for k,v in facts.items() if k not in {"liquidity","premium_discount"}})
                match = next((z for z in ranked if
                              abs(z["bottom"]-old["entry_zone"]["bottom"])<1e-7 and
                              abs(z["top"]-old["entry_zone"]["top"])<1e-7),None)
                old_score = match["score"] if match else None
            if quality:
                evidence_scores.append({"symbol":symbol,"tf":tf,"timestamp":now.isoformat(),**quality})
            future = frame.iloc[i+1:i+109]
            if len(future)<108 or (future.timestamp.diff().dropna()!=durations[tf]).any():
                continue
            for version in versions:
                if i <= busy[version]:
                    continue
                plan = dict(old) if version.startswith("old") else new
                c = counter[version]; c["decisions"] += 1
                if plan["action"] == "wait":
                    c["waits"]+=1; continue
                c["plans"]+=1
                if version.startswith("old_staged"):
                    plan["targets"] = staged_targets(plan["trend_direction"],plan["entry_level"],plan["stop_loss"],facts["liquidity"],df)
                    # Preserve old T1 even if it came from the dealing-range fallback.
                    if not plan["targets"] or plan["targets"][0]["price"]!=plan["take_profit"]:
                        plan["targets"]=[{"price":plan["take_profit"],"fraction":1.0}]
                outcome = simulate(plan,future,staged="staged" in version,breakeven=not version.endswith("no_be"))
                if outcome is None:
                    c["unfilled"]+=1
                    busy[version] = i+12
                    continue
                busy[version] = i+outcome["bars"]
                trades.append({"symbol":symbol,"tf":tf,"version":version,"timestamp":now.isoformat(),
                               "score":old_score if version.startswith("old") else quality.get("score"),"direction":plan["action"],
                               "entry":plan["entry_level"],"target_count":len(plan.get("targets",[])),**outcome})
            if number%250==0:
                print(f"{symbol} {tf}: {number}/{len(selected)}",flush=True)
        counts.extend({"symbol":symbol,"tf":tf,"version":v,**c} for v,c in counter.items())
    pd.DataFrame(trades).to_csv(output/f"{symbol}-trades.csv",index=False)
    (output/f"{symbol}-manifest.json").write_text(json.dumps(manifests,indent=2))
    (output/f"{symbol}-counts.json").write_text(json.dumps(counts,indent=2))
    (output/f"{symbol}-quality.json").write_text(json.dumps(evidence_scores,indent=2))
    return trades, counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",nargs="+",default=["BTCUSDT","ETHUSDT","SOLUSDT"])
    p.add_argument("--start",default="2025-05-15");p.add_argument("--end",default="2025-06-15")
    p.add_argument("--stride",type=int,default=1);p.add_argument("--bars",type=int,default=1000)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--output",required=True);p.add_argument("--cache",required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    root=Path(__file__).resolve().parents[2]
    sources=[*root.glob("src/core/engines/*.py"),*root.glob("src/core/use_cases/market_analysis/*.py"),Path(__file__)]
    config["source_sha256"]={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    config.update(fee_bps_per_side=10,slippage_bps_per_side=2,hold_bars=96,expiry_bars=12,
                  baseline_sha256=hashlib.sha256(Path(__file__).with_name("legacy_trade_plan.py").read_bytes()).hexdigest())
    (out/"config.json").write_text(json.dumps(config,indent=2))
    jobs=[(s,args.cache,args.output,args.start,args.end,args.stride,args.bars) for s in args.symbols]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run_symbol,jobs))
    trades=pd.DataFrame([t for rows,_ in results for t in rows])
    trades.to_csv(out/"trades.csv",index=False)
    if not trades.empty:
        trades["win"]=trades.r>0
        trades["loss"]=trades.r<0
        for groups,name in ((["version","symbol","tf"],"by-market"),(["version"],"summary"),(["version","score"],"by-score")):
            report=trades.groupby(groups).agg(trades=("r","size"),win_rate=("win","mean"),avg_r=("r","mean"),
                                             avg_gross_r=("gross_r","mean"),losing_trade_rate=("loss","mean"))
            report.to_csv(out/f"{name}.csv")
            print(name,report.to_string(),flush=True)
    (out/"counts.json").write_text(json.dumps([c for _,counts in results for c in counts],indent=2))


if __name__=="__main__":
    main()
