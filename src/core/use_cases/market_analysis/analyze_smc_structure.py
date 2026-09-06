import asyncio
from typing import Dict

import pandas as pd
from fastapi.concurrency import run_in_threadpool

from common.logger import logger
from core.engines.swing_structure_engine import SwingStructureEngine
from core.engines.market_structure_engine import MarketStructureEngine
from core.engines.liquidity_engine import LiquidityEngine
from core.engines.fvg_engine import FVGEngine
from core.engines.order_block_engine import OrderBlockEngine
from core.use_cases.market_analysis.setup_evidence import higher_timeframe_zones


async def analyze_smc_structure(ohlcv_df: pd.DataFrame, interval: str) -> Dict:
    """
    Orchestrates the SMC structural primitives for a single symbol/interval.

    Every engine here (Swing, MarketStructure, Liquidity) is pure CPU-bound
    pandas/numpy work with no I/O of its own. Marking them `async def`
    would be misleading and wouldn't actually stop them blocking - there's
    nothing inside them to await, so an async def full of sync pandas work
    blocks the event loop exactly as much as a sync def does. The
    non-blocking boundary belongs HERE, at the orchestration layer, not
    inside the engines: every sync engine call is dispatched through
    run_in_threadpool so it runs off the event loop thread, freeing the
    loop to keep serving other requests (or, once the MTFA layer exists,
    other concurrent HTF fetches) while this one grinds through pandas.

    Dependency shape:
        swings -> market_structure
        swings -> liquidity
    market_structure and liquidity both depend only on swings and not on
    each other, so once swings are computed they run concurrently via
    asyncio.gather instead of one after another - a real concurrency win
    that falls out of actually reasoning about the dependency graph,
    not from reflexively marking everything async.

    This will grow as more engines land (FVG, OB, Inversion FVG, etc.) -
    same pattern applies to each: compute the engines with no remaining
    dependencies concurrently, in threadpool, every time.
    """
    swing_engine = SwingStructureEngine(interval=interval)
    swings = await run_in_threadpool(swing_engine.detect_swings, ohlcv_df)

    structure_engine = MarketStructureEngine(interval=interval)
    liquidity_engine = LiquidityEngine(interval=interval)

    structure, liquidity = await asyncio.gather(
        run_in_threadpool(structure_engine.detect_structure, ohlcv_df, swings),
        run_in_threadpool(liquidity_engine.map_liquidity, ohlcv_df, swings),
    )

    fvg, order_blocks = await asyncio.gather(
        run_in_threadpool(FVGEngine(interval=interval).detect_fvgs, ohlcv_df),
        run_in_threadpool(OrderBlockEngine(interval=interval).detect_order_blocks, ohlcv_df, structure),
    )
    zones = await run_in_threadpool(higher_timeframe_zones, ohlcv_df, interval, order_blocks, fvg)
    logger.debug(
        f"[analyze_smc_structure] interval={interval} "
        f"swings={len(swings.swings)} structure_events={len(structure.events)} "
        f"liquidity_pools={len(liquidity.pools)}"
    )

    return {
        "interval": interval,
        "swings": swings,
        "market_structure": structure,
        "liquidity": liquidity,
        "order_blocks": order_blocks,
        "fvg": fvg,
        "poi_zones": zones,
    }
