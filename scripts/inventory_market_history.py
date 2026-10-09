"""Inventory existing legacy market history for an explicit, resumable copy.

Run while legacy-only writers are stopped. Reads no provider data or secrets.
Refuses to overwrite an existing inventory.
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository


def inventory(repo):
    base=('from(bucket: '+json.dumps(repo.bucket)+') |> range(start: 0) '
          '|> filter(fn: (r) => r._measurement == "market_data" and r._field == "close") '
          '|> group(columns: ["symbol","interval"])')
    items={}
    for aggregate in ('first','last','count'):
        for table in repo.query_api.query(base+' |> '+aggregate+'()'):
            for row in table.records:
                key=(row.values['symbol'],row.values['interval'])
                item=items.setdefault(key,dict(symbol=key[0],interval=key[1]))
                if aggregate in item:raise ValueError('Ambiguous legacy series identity')
                item[aggregate]=int(row.get_value()) if aggregate=='count' else row.get_time().isoformat()
                if len(items)>300000:raise ValueError('Inventory exceeds bounded group count')
    if not items or any(set(row)!={'symbol','interval','first','last','count'} for row in items.values()):
        raise ValueError('Source changed or inventory incomplete')
    return sorted(items.values(),key=lambda row:(row['symbol'],row['interval']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    load_dotenv()
    repo=InfluxDBMarketDataRepository(verify_connection=False,timeout_ms=60000)
    try:rows=inventory(repo)
    finally:repo.client.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(rows,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(groups=len(rows),symbols=len({r['symbol'] for r in rows}),
                         stored_candles=sum(r['count'] for r in rows),output=str(args.output))))


if __name__=='__main__':main()
