"""Frozen independent-strategy comparison. Research only, no promotion switch.

PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_brain_research --help
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

from core.config.mtfa_ladder import get_htf_chain
from core.use_cases.market_analysis.strategy_brain import evaluate_brain, evaluate_strategies, FAMILIES
from core.use_cases.market_analysis.strategy_features import local_features, top_down_context, validate_frame
from core.use_cases.market_analysis.strategy_risk import evaluate_risk, StrategyRiskPolicy
from tests.backtesting.run_trade_plans import download_month, resample_closed
from tests.backtesting.run_entry_research import stats, eligible, SYMBOLS
from tests.backtesting.research_entry_models import Signal, execute

ARMS = ('smc_isolated', 'momentum_isolated', 'fallback_missing_momentum_stress', 'arbitration_on', 'arbitration_off')
WINDOWS = {'dev': ('2024-01-01', '2025-01-01'), 'evaluation': ('2025-01-01', '2026-01-01'),
           'reused_diagnostic': ('2026-03-01', '2026-09-01')}


def fingerprints():
    files = list(Path('src/core/use_cases/market_analysis').glob('strategy_*.py'))
    files += [Path(__file__), Path('docs/regime-brain-protocol-2026-09-09.md'),
              Path('src/core/config/mtfa_ladder.py'), Path('src/common/utils/ohlcv_prep.py')]
    for name in ('regime', 'tsmom', 'swing_structure', 'market_structure', 'order_block', 'fvg'):
        files.append(Path(f'src/core/engines/{name}_engine.py'))
    for name in ('run_trade_plans', 'run_entry_research', 'research_entry_models'):
        files.append(Path(f'tests/backtesting/{name}.py'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_symbol(symbol, phase, cache):
    end = '2024-12' if phase == 'dev' else '2026-08'
    months = pd.period_range('2023-01', end, freq='M').astype(str)
    def fetch(month):
        return download_month(symbol, '1h', month, cache, include_taker_buy=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(fetch, months))
    raw = pd.concat([c[0] for c in chunks], ignore_index=True)
    gaps = raw.index[(raw.timestamp.diff() != pd.Timedelta(hours=1)) & raw.timestamp.diff().notna()]
    # Binance's March 2023 outage is in warmup, not the experiment. Restart
    # all state after it; never fabricate a candle or bridge returns across it.
    if len(gaps):
        if raw.timestamp.iloc[gaps[-1]] >= pd.Timestamp('2024-01-01', tz='UTC'):
            raise ValueError(f'{symbol}: missing bars inside evaluation period')
        raw = raw.iloc[gaps[-1]:].reset_index(drop=True)
    raw = validate_frame(raw, '1h')
    if (raw.timestamp < pd.Timestamp('2024-01-01', tz='UTC')).sum() < 6049:
        raise ValueError(f'{symbol}: insufficient contiguous pre-development warmup')
    if raw.taker_buy_volume.isna().any():
        raise ValueError(f'{symbol}: incomplete actual taker history')
    return raw, [c[1] for c in chunks]


def replay_arm(frame, candidates, start, end):
    busy, trades = -1, []
    for i, candidate, regime in candidates:
        now = frame.available_at.iloc[i]
        if i <= busy or now < start or now+pd.Timedelta(hours=48) > end:
            continue
        future = frame.iloc[i+1:i+49]
        if len(future) < 48:
            continue
        sign = 1 if candidate['direction'] == 'long' else -1
        risk = candidate['risk']
        policy = StrategyRiskPolicy(minimum_stop_bps=risk['minimum_stop_bps'])
        fill = evaluate_risk(float(future.open.iloc[0]), risk['stop'], risk['target'], sign, policy,
                             target_source=risk['target_source'])
        if not fill['eligible']:
            continue
        signal = Signal(i, candidate['strategy'], sign, risk['stop'], risk['target'], candidate['checks'])
        result = execute(signal, future)
        if result is None:
            continue
        busy = i+result['bars']
        trades.append(dict(strategy=candidate['strategy'], direction=candidate['direction'],
                           signal_time=now.isoformat(), evidence=candidate['checks'],
                           context=candidate.get('context'), regime=regime, target_source=risk['target_source'], **result))
    return trades


def generate_candidates(f, contexts):
    candidates = {a: [] for a in ARMS}
    counts = {w: Counter() for w in WINDOWS}
    bounds = {w: tuple(pd.Timestamp(s, tz='UTC') for s in dates) for w, dates in WINDOWS.items()}
    for i, row in enumerate(f.to_dict('records')):
        window = next((w for w, (a, b) in bounds.items() if a <= row['available_at'] < b), None)
        if window is None:
            continue
        counts[window]['decisions'] += 1
        counts[window]['complete_momentum'] += int(row['momentum_complete'])
        counts[window]['real_flow'] += int(row['real_flow'])
        on = evaluate_brain(row, mtfa_enabled=True, htf_available=True, htf_reactions=contexts[i])
        off = evaluate_brain(row, mtfa_enabled=False)
        stress = evaluate_strategies({**row, 'momentum_complete': False}, mtfa_enabled=False)[2]
        isolated = dict(smc_isolated=on['candidates'][0], momentum_isolated=off['candidates'][1],
                        fallback_missing_momentum_stress=stress)
        for arm, result in (('arbitration_on', on), ('arbitration_off', off)):
            choice = result['arbitration']['selected_strategy']
            counts[window][arm+'/'+result['arbitration']['status']] += 1
            if choice:
                isolated[arm] = next(c for c in result['candidates'] if c['strategy'] == choice)
        for c in on['candidates'][:2]+[off['candidates'][2]]:
            counts[window][c['strategy']+'/'+c['status']] += 1
            for missing in c.get('failed_checks', []):
                counts[window][c['strategy']+'/failed/'+missing.split(':')[0]] += 1
        for arm, candidate in isolated.items():
            if candidate['status'] == 'eligible':
                candidates[arm].append((i, candidate, on['regime']))
    return candidates, counts


def summary(checkpoints, window):
    result = {}
    for arm in ARMS:
        trades = [t for c in checkpoints for t in c['trades'][window][arm]]
        result[arm] = stats(trades)
        result[arm]['by_symbol'] = {s: stats([t for t in trades if t['symbol'] == s]) for s in SYMBOLS}
        result[arm]['by_direction'] = {d: stats([t for t in trades if t['direction'] == d]) for d in ('long', 'short')}
        result[arm]['by_regime'] = {r: stats([t for t in trades if t['regime']['trend'] == r])
                                  for r in ('trending', 'non_trending', 'unknown')}
    return result


def report(output, summaries, selected):
    lines = ['# Independent brain v1 — research results', '',
             f'Frozen development selection: **{selected or "NONE"}**. Production promotion: **NO**.', '',
             'All windows reuse previously inspected market periods. None is a pristine holdout.',
             'Hourly eight-symbol spot history, true taker flow; hypothetical shorts, not account returns.',
             '10bps fees + 2bps slippage per side; stress 10+5bps and 5bps/day carry.',
             'Fallback stress deliberately removes momentum; it is not natural full-history coverage.', '',
             '| Window | Arm | N | Win % | Mean gross R | Mean net R | PF | Stress R | 95% weekly mean-R interval |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for window, arms in summaries.items():
        for arm, values in arms.items():
            def fmt(key):
                return f'{values[key]:.3f}' if values.get(key) is not None else 'n/a'
            ci = values['weekly_ci95']
            ci_text = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'n/a'
            win = f'{100*values["win_rate"]:.1f}' if values['win_rate'] is not None else 'n/a'
            lines.append(f'| {window} | {arm} | {values["n"]} | {win} | {fmt("mean_gross_r")} | {fmt("mean_net_r")} | {fmt("profit_factor")} | {fmt("mean_stress_r")} | {ci_text} |')
    lines += ['', 'Rules/protocol/source hashes, per-symbol/direction/regime metrics and every fill are in the adjacent JSON files.',
              'Arbitration is reported diagnostically; it cannot rescue a failed independently-tested strategy.',
              'Historical feature scope is expanding from 2023 warmup; live local structure uses the requested chart lookback.',
              'Prefix equivalence tests protect causality, but these different warmup scopes are not identical replay of an app request.',
              'No tuned replacement or live promotion is authorized by these results.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['dev', 'later'], required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes, checkpoints = fingerprints(), []
    frozen = args.output/'frozen-selection.json'
    if args.phase == 'later':
        selection = json.loads(frozen.read_text())
        if selection['hashes'] != hashes:
            raise ValueError('Code/protocol changed after freeze')
    windows = ['dev'] if args.phase == 'dev' else ['evaluation', 'reused_diagnostic']
    for symbol in SYMBOLS:
        path = args.output/f'{args.phase}-{symbol}.json'
        if path.exists():
            checkpoint = json.loads(path.read_text())
            if checkpoint['hashes'] != hashes:
                raise ValueError('Checkpoint code mismatch')
        else:
            print(f'{args.phase} {symbol}: loading and computing causal features', flush=True)
            raw, manifests = load_symbol(symbol, args.phase, args.cache)
            f = local_features(raw, '1h')
            contexts = top_down_context(f, resample_closed(raw, '4h', 4), resample_closed(raw, '1D', 24),
                                        middle_interval='4h', anchor_interval='1d')
            entries, counts = generate_candidates(f, contexts)
            checkpoint = dict(symbol=symbol, hashes=hashes, manifests=manifests,
                              counts={w: dict(counts[w]) for w in windows}, trades={})
            for window in windows:
                a, b = (pd.Timestamp(s, tz='UTC') for s in WINDOWS[window])
                checkpoint['trades'][window] = {arm: [dict(symbol=symbol, **t) for t in replay_arm(f, entries[arm], a, b)] for arm in ARMS}
            path.write_text(json.dumps(checkpoint, indent=2, allow_nan=False))
        checkpoints.append(checkpoint)
        print(f'{symbol}: '+', '.join(f'{w} {sum(len(t) for t in checkpoint["trades"][w].values())} arm-fills' for w in windows), flush=True)
    summaries = {w: summary(checkpoints, w) for w in windows}
    if args.phase == 'dev':
        # Select independent natural strategies only. Forced missing-data stress
        # and combined arbitration are not substitutes for isolated validation.
        choices = [a for a in ('smc_isolated', 'momentum_isolated') if eligible(summaries['dev'][a])
                   and summaries['dev'][a]['mean_stress_r'] > 0]
        selected = min(choices, key=lambda a: (-summaries['dev'][a]['mean_net_r'], a)) if choices else None
        frozen.write_text(json.dumps(dict(selected=selected, hashes=hashes, production_promoted=False), indent=2))
        (args.output/'dev-summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    else:
        selected = selection['selected']
        summaries = {**json.loads((args.output/'dev-summary.json').read_text()), **summaries}
        (args.output/'decision.json').write_text(json.dumps(dict(selected=selected, production_promoted=False,
            untouched_holdout=False, reason='Reused data cannot authorize promotion; inspect all independent arms.'), indent=2))
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    report(args.output, summaries, selected)
    print(f'Completed: {args.output/"report.md"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Project logging installs an exception hook; do not hide a failed study
        # when routine detector logging is suppressed.
        traceback.print_exc()
        raise SystemExit(1)
