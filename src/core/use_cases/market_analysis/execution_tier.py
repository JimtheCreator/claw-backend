"""Bounded finer-timeframe discovery for shadow research; never synthesizes bars."""
import asyncio
import numpy as np
import pandas as pd

from core.config.execution_ladder import get_execution_chain
from core.config.mtfa_ladder import get_htf_chain
from core.use_cases.market_analysis.strategy_features import interval_offset, validate_frame, momentum_features, top_down_context
from core.use_cases.market_analysis.momentum_history import load_momentum_history, MomentumHistory, required_momentum_bars
from core.use_cases.market_analysis.shape_features import shape_features
from core.use_cases.market_analysis.shape_brain import evaluate_shapes, SHAPES, VERSION


def qa_mode(value):
    if value not in {'auto', 'off'}:
        raise ValueError('SMC_SHAPE_MTFA_MODE must be auto or off (internal QA only).')
    return value


def settled_frame(raw, interval, as_of):
    frame = pd.DataFrame([[pd.to_datetime(k[0], unit='ms', utc=True), *map(float, k[1:6]),
                           float(k[9]) if len(k) > 9 else np.nan] for k in raw],
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'])
    frame = frame[frame.timestamp+interval_offset(interval) <= as_of].reset_index(drop=True)
    frame = validate_frame(frame, interval)
    if len(frame) < 65 or frame.timestamp.iloc[-1]+2*interval_offset(interval) <= as_of:
        raise ValueError('Insufficient or stale closed history')
    return frame


async def fetch_closed(symbol, interval, as_of, fetch_page):
    raw = await asyncio.wait_for(fetch_page(symbol=symbol, interval=interval, limit=1000,
        end_time=int(as_of.value//10**6)-1, max_retries=1), timeout=10)
    return settled_frame(raw, interval, as_of)


async def discover_execution(symbol, interval, as_of, fetch_page):
    failures = {}
    for finer in get_execution_chain(interval):
        try:
            return finer, await fetch_closed(symbol, finer, as_of, fetch_page), failures
        except Exception as error:
            failures[finer] = type(error).__name__
    return None, None, failures


async def analyze_shape_shadow(symbol, interval, candles, *, as_of, fetch_page, cache=None, htf_mode='auto'):
    htf_mode = qa_mode(htf_mode)
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize('UTC') if as_of.tzinfo is None else as_of.tz_convert('UTC')
    # Request-scoped objects only; no public checkbox, inherited HTF map or cache
    # of decision state can influence this availability boundary.
    middle = candles.copy()
    middle['timestamp'] = pd.to_datetime(middle.timestamp, utc=True)
    middle = middle[middle.timestamp+interval_offset(interval) <= as_of]
    middle = validate_frame(middle, interval)
    finer, execution, failures = await discover_execution(symbol, interval, as_of, fetch_page)
    if finer is None:
        return dict(version=VERSION, mode='shadow', production_promoted=False,
                    roles=dict(macro=None, intermediate=interval, execution=None),
                    candidates=[dict(strategy=s, status='unavailable', reason='No valid finer execution history.') for s in SHAPES],
                    arbitration=dict(action='wait', status='no_execution_data', selected_strategy=None),
                    execution_failures=failures)
    try:
        history = await load_momentum_history(symbol, finer, execution, fetch_page, cache=cache)
    except Exception as error:
        # A cache/provider failure belongs to momentum availability, not to the
        # independent VWAP or SMC shapes. Cancellation still propagates.
        history = MomentumHistory(execution, required_momentum_bars(finer), False, 0,
                                  f'Momentum recovery failed: {type(error).__name__}.')
    local = await asyncio.to_thread(shape_features, execution, finer)
    momentum = await asyncio.to_thread(momentum_features, history.candles, finer)
    row = local.iloc[-1].to_dict()
    # VWAP/flow/structure stay on the actual execution snapshot. Only own-return
    # horizon inputs may use the dedicated extended history.
    row.update(momentum_complete=history.complete, momentum_sign=int(momentum.momentum_sign.iloc[-1]))
    from core.engines.tsmom_engine import TSMOMEngine
    for horizon in TSMOMEngine._interval_lookback_defaults[finer]:
        row[f'own_return_{horizon}'] = (float(history.candles.close.iloc[-1]/history.candles.close.iloc[-1-horizon]-1)
                                      if len(history.candles) > horizon else None)
    macro_interval, macro, macro_error = None, None, None
    chain = get_htf_chain(interval) if htf_mode == 'auto' else []
    if chain:
        macro_interval = chain[0]
        try:
            macro = await fetch_closed(symbol, macro_interval, as_of, fetch_page)
        except Exception as error:
            macro_error = type(error).__name__
    reactions = []
    middle_fresh = middle.timestamp.iloc[-1]+2*interval_offset(interval) > as_of
    context_ready = macro is not None and middle_fresh
    if context_ready:
        try:
            reactions = (await asyncio.to_thread(top_down_context, local, middle, macro,
                         middle_interval=interval, anchor_interval=macro_interval))[-1]
        except Exception as error:
            context_ready, macro_error = False, f'Context computation: {type(error).__name__}'
    output = evaluate_shapes(row, roles=dict(macro=macro_interval, intermediate=interval, execution=finer),
                             htf_available=context_ready, reactions=reactions,
                             qa_htf_off=htf_mode == 'off')
    output.update(symbol=symbol, as_of=as_of.isoformat(), execution_failures=failures,
                  momentum_history={**history.metadata(), 'interval_source': 'execution_timeframe_only', 'interval': finer})
    if htf_mode != 'off':
        output['context_availability'] = dict(macro=macro is not None, intermediate=middle_fresh, error=macro_error)
    return output
