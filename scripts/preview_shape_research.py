"""Research-only shape charts. No live planner, uploads, orders or projections of timing.

Uses the earliest development fill per shape, without selecting by its outcome.
The preview rebuilds that decision from its prefix; no future candles are drawn.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import logging
from pathlib import Path
from textwrap import wrap
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from core.use_cases.market_analysis.shape_brain import evaluate_shapes, SHAPES
from core.use_cases.market_analysis.shape_features import shape_features
from core.use_cases.market_analysis.strategy_features import interval_offset, top_down_context
from tests.backtesting.run_brain_research import load_symbol
from tests.backtesting.run_trade_plans import resample_closed
from tests.backtesting.run_shape_research import ROLES, fingerprint

GREEN, RED, AMBER, BG = '#36d9a5', '#ff596e', '#ffd078', '#101722'


def price(v):
    return f'{v:,.2f}' if abs(v) >= 10 else f'{v:.6g}'


def evidence_text(fact):
    group = fact['group']
    tf = fact.get('timeframe', '')
    if group == 'own_return_horizons':
        return f'{tf} own returns: '+', '.join(f'{k.removeprefix("own_return_")} bars {v:+.2%}' for k, v in fact['returns'].items())
    if group == 'session_vwap_band':
        return f'{tf} session VWAP {price(fact["vwap"])}; sigma {price(fact["sigma"])}; stretch {fact["z_score"]:+.2f} SD.'
    if group == 'macro_poi':
        return f'{tf} POI {price(fact["price_zone"][0])}–{price(fact["price_zone"][1])}; known at {fact["available_at"]}.'
    if group == 'intermediate_reaction':
        return f'{tf} zone reaction closed at {fact["observed_at"]} before the execution trigger.'
    if group == 'optional_cvd':
        return f'{tf} genuine CVD: {fact["status"]}; optional diagnostic, NOT an entry gate.'
    prefix = 'PASS' if fact.get('passed') else 'FAIL'
    return f'{prefix} · {tf} {group.replace("_", " ")}' + (': '+fact['reason'] if fact.get('reason') else '.')


def shape_figure(symbol, requested, execution, candidate, roles):
    c = deepcopy(candidate)
    now = pd.Timestamp(c['execution_time'])
    frames = []
    for frame, interval, count in ((requested, roles['intermediate'], 60), (execution, roles['execution'], 24)):
        f = frame.copy()
        f['closed_at'] = pd.to_datetime(f.timestamp, utc=True)+interval_offset(interval)
        frames.append(f[f.closed_at <= now].tail(count))
    if any(f.empty for f in frames):
        raise ValueError('Both requested and execution closed histories are required.')
    fig = make_subplots(rows=1, cols=2, column_widths=[.6, .4], horizontal_spacing=.07,
                        subplot_titles=[f'Requested {roles["intermediate"]} · closed candles',
                                        f'Actual {roles["execution"]} execution · price levels, not an arrival time'])
    fig.update_layout(template='plotly_dark', width=1800, height=1050,
                      paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
                      margin=dict(l=65, r=115, t=155, b=370),
                      font=dict(family='Arial', size=19),
                      title=dict(text=f'<b>{escape(symbol)} · {escape(c["strategy"])}</b><br>'
                                 f'<sup>RESEARCH ONLY · {escape(c["status"].upper())} · '
                                 f'{escape(str(roles.get("macro") or "No macro"))} / {roles["intermediate"]} / {roles["execution"]} · '
                                 f'{now.strftime("%Y-%m-%d %H:%M UTC")}</sup>', x=.035, y=.97),
                      meta=dict(research_only=True, production_promoted=False, roles=roles,
                                execution_time=now.isoformat(), chart_evidence=c['chart_evidence']))
    for col, frame in enumerate(frames, 1):
        fig.add_trace(go.Candlestick(x=frame.closed_at, open=frame.open, high=frame.high,
            low=frame.low, close=frame.close, increasing_line_color=GREEN,
            decreasing_line_color=RED, name='Observed closed candles'), row=1, col=col)
        fig.update_xaxes(rangeslider_visible=False, gridcolor='#223043', row=1, col=col)
        fig.update_yaxes(gridcolor='#223043', tickformat=',.2f' if frame.close.iloc[-1] >= 10 else '.5g',
                         row=1, col=col)
    risk = c.get('risk') or {}
    levels = [risk.get(k) for k in ('entry', 'stop', 'target')]
    if c['status'] == 'eligible' and all(v is not None for v in levels):
        entry, stop, target = levels
        end = now+10*interval_offset(roles['execution'])
        sign = 1 if c['direction'] == 'long' else -1
        for edge, color in ((target, GREEN), (stop, RED)):
            fig.add_shape(type='rect', x0=now, x1=end, y0=min(entry, edge), y1=max(entry, edge),
                          fillcolor=color, opacity=.17, line_width=0, row=1, col=2)
        for value, text, color in ((target, 'TP', GREEN), (entry, 'Entry reference', AMBER), (stop, 'SL', RED)):
            fig.add_shape(type='line', x0=now, x1=end, y0=value, y1=value, line=dict(color=color, width=1.5), row=1, col=2)
            fig.add_annotation(x=end, y=value, text=f'{text} {price(value)}', showarrow=False,
                               xanchor='right', yshift=13, bgcolor=BG, font=dict(color=color, size=18), row=1, col=2)
        last = frames[1].iloc[-1]
        fig.add_annotation(x=now, y=last.low if sign == 1 else last.high,
                           text='Long ▲' if sign == 1 else 'Short ▼', showarrow=False,
                           yshift=-28 if sign == 1 else 28, borderpad=6,
                           bgcolor=GREEN if sign == 1 else RED, font=dict(color='white', size=20), row=1, col=2)
        fig.update_xaxes(range=[frames[1].closed_at.iloc[0], end+interval_offset(roles['execution'])], row=1, col=2)
        caption = f'Next {roles["execution"]} open only; shown entry is the closed-signal reference, not a guaranteed fill. '
        caption += f'Recheck fixed SL/TP after any gap. Gross target {risk["gross_target_r"]:.2f}R; stop {risk["stop_distance_bps"]:.1f} bps.'
    else:
        caption = 'No eligible entry: '+('; '.join(c.get('failed_checks', [])) or c.get('reason', 'No matching shape.'))
    for fact in c['chart_evidence']:
        if fact['group'] == 'macro_poi':
            fig.add_shape(type='rect', x0=max(frames[0].closed_at.iloc[0], pd.Timestamp(fact['available_at'])),
                          x1=now, y0=fact['price_zone'][0], y1=fact['price_zone'][1],
                          fillcolor=AMBER, opacity=.12, line_width=0, row=1, col=1)
        if fact['group'] == 'intermediate_reaction':
            reaction = frames[0][frames[0].closed_at == pd.Timestamp(fact['observed_at'])]
            if not reaction.empty:
                bar = reaction.iloc[-1]
                fig.add_trace(go.Scatter(x=[bar.closed_at.isoformat()], y=[bar.close], mode='markers',
                    marker=dict(size=15, color=AMBER, symbol='circle-open', line_width=2), name='Observed 4h POI reaction'), row=1, col=1)
        if fact['group'] == 'session_vwap_band':
            fig.add_shape(type='line', x0=now, x1=now+10*interval_offset(roles['execution']),
                          y0=fact['vwap'], y1=fact['vwap'], line=dict(color=GREEN, width=1), row=1, col=2)
    last = frames[1].iloc[-1]
    if c['strategy'] == 'smc_location_v1' and pd.notna(last.get('break_level')):
        fig.add_shape(type='line', x0=frames[1].closed_at.iloc[-3], x1=now,
                      y0=last.break_level, y1=last.break_level, line=dict(color=AMBER, width=2), row=1, col=2)
        fig.add_annotation(x=frames[1].closed_at.iloc[-3], y=last.break_level,
                           text=f'{roles["execution"]} {escape(last.break_kind)} {price(last.break_level)}',
                           showarrow=False, xanchor='right', yshift=-15, font=dict(color=AMBER, size=18), row=1, col=2)
    ledger = [caption]+[evidence_text(f) for f in c['chart_evidence']]
    text = '<br>'.join(escape(line) for item in ledger for line in wrap(item, 145))
    # No cap: every material fact is present. Extra lines grow the canvas.
    line_count = text.count('<br>')+1
    fig.update_layout(height=max(1050, 735+line_count*26), margin_b=max(370, line_count*26+90))
    fig.add_annotation(xref='paper', yref='paper', x=0, y=0, yshift=-85, xanchor='left', yanchor='top',
                       text=text, align='left', showarrow=False, font=dict(size=18))
    fig.add_annotation(xref='paper', yref='paper', x=0, y=0, yshift=-110-line_count*25,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=15, color=AMBER),
                       text='Unvalidated historical illustration · Earliest development fill, not selected for a winning outcome · No projected arrow')
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    args = p.parse_args()
    if json.loads((args.study/'source-freeze.json').read_text()) != fingerprint():
        raise ValueError('Decision source differs from the study; refuse misleading preview.')
    checkpoints = [json.loads(p.read_text()) for p in sorted(args.study.glob('dev-*USDT.json'))]
    for family in SHAPES:
        rows = [t for c in checkpoints for t in c['trades']['dev'][family]]
        if not rows:
            continue
        trade = min(rows, key=lambda t: (t['signal_time'], t['symbol']))
        raw, _ = load_symbol(trade['symbol'], 'dev', args.cache)
        now = pd.Timestamp(trade['signal_time'])
        raw = raw[raw.timestamp+pd.Timedelta(hours=1) <= now].reset_index(drop=True)
        requested, macro = resample_closed(raw, '4h', 4), resample_closed(raw, '1D', 24)
        f = shape_features(raw, '1h')
        contexts = top_down_context(f, requested, macro, middle_interval='4h', anchor_interval='1d')
        brain = evaluate_shapes(f.iloc[-1].to_dict(), roles=ROLES, htf_available=True, reactions=contexts[-1])
        c = next(c for c in brain['candidates'] if c['strategy'] == family)
        assert c['status'] == 'eligible' and c['direction'] == trade['direction']
        fig = shape_figure(trade['symbol'], requested, f, c, ROLES)
        path = args.study/f'preview-{family}.png'
        pio.write_image(json.loads(fig.to_json()), path, scale=1)
        (args.study/f'preview-{family}.json').write_text(json.dumps(dict(symbol=trade['symbol'], **brain), indent=2, allow_nan=False))
        print(path, flush=True)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
