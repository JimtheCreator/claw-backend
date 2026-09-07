"""Report stop distance and its relationship to net R for recorded fills."""
import argparse
import json
from pathlib import Path

import pandas as pd


def build(input_csv, output):
    rows=pd.read_csv(input_csv)
    rows=rows[rows.version.isin(["old","new_single"])].copy()
    rows["planner"]=rows.version.map({"old":"old","new_single":"new"})
    rows["stop_distance"]=abs(rows.entry-rows.initial_stop)
    rows["stop_bps"]=rows.stop_distance/rows.entry*10000
    rows["modeled_round_trip_cost_r"]=((rows.entry+rows.entry)*12/10000)/rows.stop_distance
    rows.to_csv(output/"existing-stop-distance-trades.csv",index=False)
    summary=[];bins=[]
    for planner,group in rows.groupby("planner"):
        summary.append(dict(planner=planner,n=len(group),minimum_stop_bps=group.stop_bps.min(),
            median_stop_bps=group.stop_bps.median(),maximum_stop_bps=group.stop_bps.max(),
            pearson_stop_bps_net_r=group.stop_bps.corr(group.r) if len(group)>2 else None,
            spearman_stop_bps_net_r=group.stop_bps.rank().corr(group.r.rank()) if len(group)>2 else None))
        if len(group)>3:
            group=group.assign(quartile=pd.qcut(group.stop_bps,4,duplicates="drop"))
            for label,part in group.groupby("quartile",observed=True):
                bins.append(dict(planner=planner,bin=str(label),n=len(part),stop_bps_min=part.stop_bps.min(),
                    stop_bps_max=part.stop_bps.max(),mean_net_r=part.r.mean(),mean_gross_r=part.gross_r.mean()))
    pd.DataFrame(summary).to_csv(output/"existing-stop-distance-summary.csv",index=False)
    pd.DataFrame(bins).to_csv(output/"existing-stop-distance-quartiles.csv",index=False)
    (output/"existing-stop-distance-summary.json").write_text(json.dumps(summary,indent=2))
    return summary,bins


def main():
    p=argparse.ArgumentParser();p.add_argument("input_csv");p.add_argument("output")
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary,bins=build(args.input_csv,out)
    print(json.dumps({"summary":summary,"quartiles":bins},indent=2))


if __name__=="__main__":main()
