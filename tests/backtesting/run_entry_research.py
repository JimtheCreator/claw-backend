"""Run the preregistered hourly study; persist selection before holdout access.

PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_entry_research --help
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tests.backtesting.run_trade_plans import download_month
from tests.backtesting.research_entry_models import prepare, signals, replay

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT']
MODELS = ['structure_break', 'sweep_reclaim', 'trendline_break']
WINDOWS = {'dev': ('2024-01-01', '2025-01-01'),
           'validation': ('2025-01-01', '2026-01-01'),
           'holdout': ('2026-03-01', '2026-09-01')}


def fingerprint():
    paths = [Path(__file__), Path(__file__).with_name('research_entry_models.py'),
             Path('docs/strategy-research-protocol-2026-09-08.md')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def stats(trades):
    if not trades:
        return dict(n=0, mean_net_r=None, mean_gross_r=None, mean_stress_r=None,
                    win_rate=None, profit_factor=None, weekly_ci95=None, positive_symbols=0)
    f = pd.DataFrame(trades)
    loss = -f.loc[f.net_r < 0, 'net_r'].sum()
    profit = f.loc[f.net_r > 0, 'net_r'].sum()
    f['week'] = pd.to_datetime(f.entry_time, utc=True).dt.tz_localize(None).dt.to_period('W').astype(str)
    weekly = f.groupby('week').net_r.agg(['sum', 'count']).to_numpy()
    if len(weekly) >= 8:
        rng = np.random.default_rng(20260908)
        draw = rng.integers(0, len(weekly), size=(2000, len(weekly)))
        totals = weekly[draw].sum(axis=1)
        ci = np.quantile(totals[:, 0]/totals[:, 1], [.025, .975]).tolist()
    else:
        ci = None
    return dict(n=len(f), mean_net_r=float(f.net_r.mean()), mean_gross_r=float(f.gross_r.mean()),
                mean_stress_r=float(f.stress_r.mean()), win_rate=float((f.net_r > 0).mean()),
                profit_factor=float(profit/loss) if loss else None, weekly_ci95=ci,
                median_stop_bps=float(f.stop_bps.median()),
                positive_symbols=int((f.groupby('symbol').net_r.mean() > 0).sum()))


def eligible(s):
    return (s['n'] >= 100 and s['mean_net_r'] > 0 and s['profit_factor'] is not None
            and s['profit_factor'] >= 1.10 and s['positive_symbols'] >= 5
            and s['weekly_ci95'] is not None and s['weekly_ci95'][0] > 0)


def load_symbol(symbol, phase, cache):
    # Only load the development archive range until development is frozen.
    end = '2024-12' if phase == 'dev' else '2026-08'
    months = pd.period_range('2023-11', end, freq='M').astype(str)
    def fetch(month):
        return download_month(symbol, '1h', month, cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(fetch, months))
    f = pd.concat([c[0] for c in chunks]).sort_values('timestamp').reset_index(drop=True)
    if f.timestamp.duplicated().any() or (f.timestamp.diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError(f'{symbol}: duplicate or missing hourly candles')
    numbers = f[['open', 'high', 'low', 'close', 'volume']].to_numpy()
    if not np.isfinite(numbers).all() or (numbers[:, :4] <= 0).any() or (f.volume < 0).any():
        raise ValueError(f'{symbol}: invalid price/volume')
    if (f.high < f[['open', 'close', 'low']].max(axis=1)).any() or (f.low > f[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError(f'{symbol}: invalid OHLC ordering')
    return f, [c[1] for c in chunks]


def summarize(checkpoints, window):
    result = {}
    for model in MODELS:
        for on in (False, True):
            key = f'{model}/mtfa_{"on" if on else "off"}'
            rows = [r for c in checkpoints for r in c['trades'].get(window, {}).get(key, [])]
            result[key] = stats(rows)
            result[key]['by_symbol'] = {s: stats([r for r in rows if r['symbol'] == s]) for s in SYMBOLS}
            result[key]['by_direction'] = {s: stats([r for r in rows if r['direction'] == s]) for s in ('long', 'short')}
            months = sorted({r['entry_time'][:7] for r in rows})
            result[key]['by_month'] = {m: stats([r for r in rows if r['entry_time'][:7] == m]) for m in months}
    return result


def write_report(out, summary, selected):
    lines = ['# Hourly strategy research results', '',
             f'Frozen development selection: **{selected or "NONE"}**. Production changed: **NO**.', '',
             'Net R includes 10bps fee + 2bps slippage per side. Stress adds 3bps/side and 5bps/day carry.',
             'Eight spot-data symbols; shorts are hypothetical. This is not an account-return backtest.', '',
             '| Window | Policy | Trades | Win % | Gross R | Net R | PF | Stress R | Weekly 95% mean-R interval |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for window, policies in summary.items():
        for key, s in policies.items():
            def fmt(k):
                return f'{s[k]:.3f}' if s.get(k) is not None else 'n/a'
            ci = s.get('weekly_ci95')
            bound = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'n/a'
            win = f'{100*s["win_rate"]:.1f}' if s['win_rate'] is not None else 'n/a'
            lines.append(f'| {window} | {key} | {s["n"]} | {win} | {fmt("mean_gross_r")} | {fmt("mean_net_r")} | {fmt("profit_factor")} | {fmt("mean_stress_r")} | {bound} |')
    lines += ['', 'No post-holdout replacement winner is allowed. All six reported arms consume the holdout.',
              'See summary.json for each symbol, direction and month; checkpoint JSONs contain every trade, observed entry evidence and archive hashes.',
              'See the preregistration for limits: reused development/validation history, survivor universe, correlated markets, fixed 2R target, no actual futures funding or live fill validation.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['dev', 'later'], required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = fingerprint()
    frozen = args.output/'frozen-selection.json'
    if args.phase == 'later':
        selection = json.loads(frozen.read_text())
        if selection['hashes'] != hashes:
            raise ValueError('Research code/protocol changed after selection; do not silently reuse holdout')
    checkpoints = []
    windows = ['dev'] if args.phase == 'dev' else ['validation', 'holdout']
    for symbol in SYMBOLS:
        path = args.output/f'{args.phase}-{symbol}.json'
        if path.exists():
            c = json.loads(path.read_text())
            if c['hashes'] != hashes:
                raise ValueError('Checkpoint code mismatch')
            checkpoints.append(c)
            print(f'{args.phase} {symbol}: checkpoint reused', flush=True)
            continue
        print(f'{args.phase} {symbol}: loading verified hourly history', flush=True)
        raw, manifest = load_symbol(symbol, args.phase, args.cache)
        f = prepare(raw)
        c = dict(symbol=symbol, hashes=hashes, manifests=manifest, candles=len(raw), trades={w: {} for w in windows})
        for model in MODELS:
            for on in (False, True):
                key = f'{model}/mtfa_{"on" if on else "off"}'
                candidates = signals(f, model, mtfa=on)
                for window in windows:
                    start, end = (pd.Timestamp(d, tz='UTC') for d in WINDOWS[window])
                    trades = replay(f, candidates, start, end)
                    c['trades'][window][key] = [dict(symbol=symbol, policy=key, **r) for r in trades]
        path.write_text(json.dumps(c, indent=2, allow_nan=False))
        checkpoints.append(c)
        print(f'{args.phase} {symbol}: saved {sum(len(v) for w in c["trades"].values() for v in w.values())} strategy-trades', flush=True)
    summary = {w: summarize(checkpoints, w) for w in windows}
    if args.phase == 'dev':
        choices = [k for k, s in summary['dev'].items() if eligible(s)]
        selected = sorted(choices, key=lambda k: (-summary['dev'][k]['mean_net_r'], k))[0] if choices else None
        frozen.write_text(json.dumps(dict(selected=selected, hashes=hashes, production_default=False), indent=2))
        (args.output/'dev-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    else:
        selected = selection['selected']
        summary = {**json.loads((args.output/'dev-summary.json').read_text()), **summary}
        passes = bool(selected and all(eligible(summary[w][selected]) and summary[w][selected]['mean_stress_r'] > 0 for w in windows))
        (args.output/'decision.json').write_text(json.dumps(dict(selected=selected,
            qualifies_for_forward_paper_study=passes, production_default=False,
            decision='forward paper candidate only' if passes else 'reject promotion'), indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_report(args.output, summary, selected)
    print(f'Complete; frozen selection: {selected}; report: {args.output/"report.md"}', flush=True)


if __name__ == '__main__':
    main()
