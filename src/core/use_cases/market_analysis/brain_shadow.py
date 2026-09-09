"""Opt-in integration boundary. A failed research branch never erases a chart."""
import asyncio

from core.config.mtfa_ladder import get_htf_chain
from core.use_cases.market_analysis.strategy_features import local_features, momentum_features, top_down_context
from core.use_cases.market_analysis.strategy_brain import evaluate_brain
from core.use_cases.market_analysis.momentum_history import load_momentum_history


def brain_policy(value):
    if value not in {'legacy', 'shadow_v1'}:
        raise ValueError('SMC_BRAIN_POLICY must be legacy or shadow_v1; no strategy is promoted.')
    return value


async def analyze_brain_shadow(symbol, interval, candles, *, mtfa_enabled, htf_frames, fetch_page, cache=None):
    history = await load_momentum_history(symbol, interval, candles, fetch_page, cache=cache)
    # Keep local structural scope identical to the requested chart. Only the
    # momentum horizons use the extended history; never swap chart timeframe.
    local = await asyncio.to_thread(local_features, candles, interval)
    momentum = await asyncio.to_thread(momentum_features, history.candles, interval)
    row = local.iloc[-1].to_dict()
    row.update(momentum.iloc[-1].to_dict())
    row['momentum_complete'] = history.complete
    chain = get_htf_chain(interval)[:2] if mtfa_enabled else []
    available = len(chain) == 2 and all(tf in htf_frames for tf in chain)
    reactions = []
    if available:
        middle, anchor = chain
        contexts = await asyncio.to_thread(top_down_context, local, htf_frames[middle], htf_frames[anchor],
                                          middle_interval=middle, anchor_interval=anchor)
        reactions = contexts[-1]
    output = evaluate_brain(row, mtfa_enabled=mtfa_enabled, htf_available=available, htf_reactions=reactions)
    output['momentum_history'] = history.metadata()
    return output
