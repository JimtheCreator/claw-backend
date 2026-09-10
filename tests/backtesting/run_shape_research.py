"""Preregistered standalone shapes, finer execution, immutable source freeze."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import traceback

import pandas as pd

from core.use_cases.market_analysis.shape_features import shape_features
from core.use_cases.market_analysis.shape_brain import evaluate_shapes, SHAPES
from core.use_cases.market_analysis.strategy_features import top_down_context, interval_offset
from core.use_cases.market_analysis.strategy_risk import evaluate_risk, StrategyRiskPolicy
from tests.backtesting.run_brain_research import load_symbol, WINDOWS
from tests.backtesting.run_trade_plans import resample_closed
from tests.backtesting.run_entry_research import stats, eligible, SYMBOLS

ARMS = (*SHAPES, 'unblended_arbitration')
ROLES = dict(macro='1d', intermediate='4h', execution='1h')


def fingerprint():
    paths = [Path(__file__), Path('docs/master-vision-protocol-2026-09-11.md')]
    paths += list(Path('src/core/use_cases/market_analysis').glob('shape_*.py'))
    paths += [Path('src/core/use_cases/market_analysis')/f'{n}.py' for n in
              ('strategy_brain', 'strategy_features', 'strategy_risk', 'execution_tier', 'momentum_history')]
    paths += [Path('tests/backtesting')/f'{n}.py' for n in ('run_brain_research', 'run_trade_plans', 'run_entry_research', 'research_entry_models')]
    paths += list(Path('src/core/engines').glob('*.py'))
    paths += [Path('src/core/config')/f'{n}.py' for n in ('mtfa_ladder', 'execution_ladder')]
    paths += [Path('src/common/utils/ohlcv_prep.py')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def execute_candidate(candidate, future, interval='1h', hold=48):
    if len(future) < hold or hold < 1:
        return None
    sign = 1 if candidate['direction'] == 'long' else -1
    planned = candidate['risk']
    entry, stop, target = float(future.open.iloc[0]), planned['stop'], planned['target']
    risk = evaluate_risk(entry, stop, target, sign, StrategyRiskPolicy(), target_source=planned['target_source'])
    if not risk['eligible']:
        return None
    unit = risk['risk_per_unit']
    for n, bar in enumerate(future.iloc[:hold].itertuples(index=False)):
        if (bar.low <= stop if sign == 1 else bar.high >= stop):
            price, why = (min(stop, bar.open) if sign == 1 else max(stop, bar.open)), 'stop'
        elif (bar.high >= target if sign == 1 else bar.low <= target):
            price, why = target, 'target'
        elif n == hold-1:
            price, why = bar.close, 'session' if candidate.get('exit_at_session_end') else 'time'
        else:
            continue
        duration = ((pd.Timestamp(bar.timestamp)+interval_offset(interval))-future.timestamp.iloc[0])/pd.Timedelta(days=1)
        gross = sign*(price-entry)/unit
        cost, stress = (entry+price)*.0012/unit, (entry+price)*.0015/unit+entry*.0005*duration/unit
        return dict(entry=entry, stop=stop, target=target, exit_price=float(price), exit=why,
                    gross_r=float(gross), net_r=float(gross-cost), stress_r=float(gross-stress),
                    stop_bps=risk['stop_distance_bps'], bars=n+1,
                    entry_time=future.timestamp.iloc[0].isoformat(),
                    exit_time=(pd.Timestamp(bar.timestamp)+interval_offset(interval)).isoformat())


def collect_symbol(frame, contexts, windows):
    trades = {w: {a: [] for a in ARMS} for w in windows}
    counts = {w: Counter() for w in windows}
    busy = {w: {a: -1 for a in ARMS} for w in windows}
    bounds = {w: tuple(pd.Timestamp(t, tz='UTC') for t in WINDOWS[w]) for w in windows}
    rows = frame.to_dict('records')
    for i, row in enumerate(rows):
        now = row['available_at']
        window = next((w for w, (a, b) in bounds.items() if a <= now < b), None)
        if window is None:
            continue
        counts[window]['execution_decisions'] += 1
        brain = evaluate_shapes(row, roles=ROLES, htf_available=True, reactions=contexts[i])
        counts[window]['arbitration/'+brain['arbitration']['status']] += 1
        attempts = {}
        for c in brain['candidates']:
            key = c['strategy']
            counts[window][key+'/'+c['status']] += 1
            counts[window][key+'/signal_fired'] += int(c.get('signal_fired', False))
            for why in c.get('failed_checks', []):
                counts[window][key+'/failed/'+why.split(':')[0]] += 1
            if c['status'] == 'eligible':
                attempts[key] = c
        selected = brain['arbitration']['selected_strategy']
        if selected:
            attempts['unblended_arbitration'] = attempts[selected]
        for arm, c in attempts.items():
            hold = 48
            if c.get('exit_at_session_end'):
                remaining = row['timestamp'].floor('D')+pd.Timedelta(days=1)-now
                hold = min(hold, int(remaining/pd.Timedelta(hours=1)))
            if i <= busy[window][arm] or hold < 1 or now+pd.Timedelta(hours=hold) > bounds[window][1]:
                continue
            fill = execute_candidate(c, frame.iloc[i+1:i+1+hold], hold=hold)
            if fill is None:
                counts[window][arm+'/fill_rejected'] += 1
                continue
            busy[window][arm] = i+fill['bars']
            trades[window][arm].append(dict(strategy=c['strategy'], direction=c['direction'],
                signal_time=now.isoformat(), checks=c['checks'], optional_cvd=c.get('optional_cvd'),
                chart_evidence=c.get('chart_evidence'), context=c.get('context'), roles=ROLES, **fill))
    return trades, {w: dict(v) for w, v in counts.items()}


def summarize(checkpoints, windows):
    result = {}
    for w in windows:
        result[w] = {}
        for arm in ARMS:
            rows = [t for c in checkpoints for t in c['trades'][w][arm]]
            value = stats(rows)
            value['by_symbol'] = {s: stats([t for t in rows if t['symbol'] == s]) for s in SYMBOLS}
            value['by_direction'] = {d: stats([t for t in rows if t['direction'] == d]) for d in ('long', 'short')}
            value['cvd_cohorts'] = {s: stats([t for t in rows if (t.get('optional_cvd') or {}).get('status') == s])
                                    for s in ('aligned', 'opposed', 'neutral', 'unavailable')}
            result[w][arm] = value
    return result


def write_report(output, summaries, selected):
    lines = ['# Master-vision standalone shapes — historical research', '',
             f'Frozen development winner: **{selected or "NONE"}**. Production promotion: **NO**.',
             'All periods were inspected previously; there is no untouched historical holdout.',
             'Roles: daily macro / requested 4h intermediate / observed 1h execution. Eight symbols, stride 1.',
             '10+2bps per side; stress 10+5bps and 5bps/day carry. Shorts hypothetical. No account-return claims.', '',
             '| Window | Shape | Trades | Win % | Gross R | Net R | PF | Stress R | Weekly 95% CI |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for w, arms in summaries.items():
        for arm, s in arms.items():
            fmt = lambda k: f'{s[k]:.3f}' if s.get(k) is not None else 'n/a'
            ci = s['weekly_ci95']
            interval = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'n/a'
            win = f'{100*s["win_rate"]:.1f}' if s['win_rate'] is not None else 'n/a'
            lines.append(f'| {w} | {arm} | {s["n"]} | {win} | {fmt("mean_gross_r")} | {fmt("mean_net_r")} | {fmt("profit_factor")} | {fmt("mean_stress_r")} | {interval} |')
    lines += ['', 'Each candidate is isolated before arbitration. Agreeing shapes retain separate plans; no blending or learned rank.',
              'CVD cohorts, per-symbol/direction outcomes, rejection counts and every fill are in summary/checkpoint JSON.',
              'CVD cohort differences are associations, not causal effects or calibrated confidence.',
              'No post-evaluation replacement winner. A reserved prospective window has not happened yet.',
              'Expanding historical structural scope differs from an arbitrary app lookback; this is not an exact replay of every UI request.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['dev', 'later'], required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    hashes = fingerprint()
    freeze = args.output/'source-freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text()) != hashes:
            raise ValueError('Source/protocol changed after freeze; refuse outcome-driven rerun.')
    elif args.phase == 'dev':
        freeze.write_text(json.dumps(hashes, indent=2))  # BEFORE market outcomes.
    else:
        raise ValueError('Development/source freeze must exist first.')
    selection_file = args.output/'frozen-selection.json'
    if args.phase == 'later':
        selection = json.loads(selection_file.read_text())
        if selection['hashes'] != hashes:
            raise ValueError('Frozen selection hash mismatch')
    windows = ['dev'] if args.phase == 'dev' else ['evaluation', 'reused_diagnostic']
    checkpoints = []
    for symbol in SYMBOLS:
        path = args.output/f'{args.phase}-{symbol}.json'
        if path.exists():
            c = json.loads(path.read_text())
            if c['hashes'] != hashes:
                raise ValueError('Checkpoint source mismatch')
        else:
            print(f'{args.phase} {symbol}: verifying archives and finer execution features', flush=True)
            raw, manifests = load_symbol(symbol, args.phase, args.cache)
            frame = shape_features(raw, '1h')
            contexts = top_down_context(frame, resample_closed(raw, '4h', 4), resample_closed(raw, '1D', 24),
                                        middle_interval='4h', anchor_interval='1d')
            trades, counts = collect_symbol(frame, contexts, windows)
            c = dict(symbol=symbol, hashes=hashes, manifests=manifests, counts=counts,
                     trades={w: {a: [dict(symbol=symbol, **t) for t in rows] for a, rows in arms.items()} for w, arms in trades.items()})
            path.write_text(json.dumps(c, indent=2, allow_nan=False))
        checkpoints.append(c)
        print(f'{symbol}: '+', '.join(f'{w} {a}={len(t)}' for w, arms in c['trades'].items() for a, t in arms.items()), flush=True)
    summaries = summarize(checkpoints, windows)
    if args.phase == 'dev':
        choices = [s for s in SHAPES if eligible(summaries['dev'][s]) and summaries['dev'][s]['mean_stress_r'] > 0]
        selected = min(choices, key=lambda s: (-summaries['dev'][s]['mean_net_r'], s)) if choices else None
        selection_file.write_text(json.dumps(dict(selected=selected, hashes=hashes, production_promoted=False), indent=2))
        (args.output/'dev-summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    else:
        selected = selection['selected']
        summaries = {**json.loads((args.output/'dev-summary.json').read_text()), **summaries}
        passed = bool(selected and eligible(summaries['evaluation'][selected]) and summaries['evaluation'][selected]['mean_stress_r'] > 0)
        (args.output/'decision.json').write_text(json.dumps(dict(selected=selected, evaluation_passed=passed,
            untouched_holdout=False, production_promoted=False, reason='No untouched completed window; no production promotion.'), indent=2))
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    write_report(args.output, summaries, selected)
    print(f'Completed {args.output/"report.md"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
