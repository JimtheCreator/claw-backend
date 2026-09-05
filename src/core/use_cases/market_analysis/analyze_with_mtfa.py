import asyncio
from typing import Awaitable, Callable, Dict

import pandas as pd

from common.logger import logger
from core.config.mtfa_ladder import get_htf_chain
from core.use_cases.market_analysis.analyze_smc_structure import analyze_smc_structure

CandleFetcher = Callable[[str, str], Awaitable[pd.DataFrame]]

async def analyze_with_mtfa(
    symbol: str,
    interval: str,
    mtfa_enabled: bool,
    candle_fetcher: CandleFetcher,
) -> Dict:
    """
    Top-level orchestrator: fetches the requested timeframe and, if MTFA
    is on, every higher timeframe in its ladder chain, then runs the SMC
    structural pipeline (analyze_smc_structure) on each.

    Takes `candle_fetcher` as an injected dependency rather than importing
    a concrete Binance/InfluxDB/Redis client directly - this module
    doesn't need to know how candles get fetched, only that it can await
    one call per (symbol, interval) pair. Wire your existing candle-fetch
    function in at the call site.

    `context` is "standalone" when MTFA is off OR when the requested
    interval sits at the top of the ladder (no higher timeframe
    configured) - both cases mean the same thing to a consumer (no HTF
    context was consulted), regardless of which reason produced it.

    Requested-TF and every HTF fetch+analysis run FULLY CONCURRENTLY via
    asyncio.gather - all independent I/O/CPU work, same reasoning as
    analyze_smc_structure's own internal concurrency, just one layer up.

    Failure isolation is asymmetric on purpose:
      - If the REQUESTED timeframe's fetch or analysis fails, that
        exception propagates. There's no meaningful partial result to
        return without the timeframe the caller actually asked for.
      - If an HTF's fetch or analysis fails, it's caught, logged, and
        that HTF is simply excluded from `htf` and `htf_trend_alignment`
        - one bad HTF (a transient DB hiccup, a symbol not listed on a
        higher-TF venue, whatever) doesn't take down the whole response.

    `htf_trend_alignment`: per HTF, whether its Market Structure trend
    agrees with the requested TF's trend (True/False, or None if either
    trend is undetermined). This SURFACES disagreement as data - it does
    NOT suppress, downweight, or otherwise alter the requested-TF result.
    Whether product behavior should hide, flag, or ignore a disagreeing
    HTF is a decision this function deliberately leaves to the caller;
    baking in a suppression policy here would assume an answer that
    hasn't actually been made.
    """
    htf_chain = get_htf_chain(interval) if mtfa_enabled else []

    if not htf_chain:
        requested_result = await analyze_smc_structure(await candle_fetcher(symbol, interval), interval)
        return {
            "symbol": symbol,
            "interval": interval,
            "context": "standalone",
            "requested": requested_result,
            "htf": {},
            "htf_trend_alignment": {},
        }

    tasks = [_fetch_and_analyze(symbol, interval, candle_fetcher)] + [
        _fetch_and_analyze(symbol, tf, candle_fetcher) for tf in htf_chain
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    requested_outcome = outcomes[0]
    if isinstance(requested_outcome, Exception):
        logger.error(f"[analyze_with_mtfa] Requested TF fetch/analysis failed for {symbol} {interval}: {requested_outcome}")
        raise requested_outcome
    requested_result = requested_outcome

    htf_by_interval: Dict[str, Dict] = {}
    for tf, outcome in zip(htf_chain, outcomes[1:]):
        if isinstance(outcome, Exception):
            logger.warning(
                f"[analyze_with_mtfa] HTF fetch/analysis failed for {symbol} {tf}: "
                f"{outcome} - excluding this HTF, other timeframes unaffected."
            )
            continue
        htf_by_interval[tf] = outcome

    requested_trend = requested_result["market_structure"].trend
    htf_trend_alignment: Dict[str, bool] = {}
    for tf, res in htf_by_interval.items():
        htf_trend = res["market_structure"].trend
        if requested_trend is None or htf_trend is None:
            htf_trend_alignment[tf] = None
        else:
            htf_trend_alignment[tf] = requested_trend == htf_trend

    logger.debug(
        f"[analyze_with_mtfa] symbol={symbol} interval={interval} htf_chain={htf_chain} "
        f"resolved_htfs={list(htf_by_interval)} alignment={htf_trend_alignment}"
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "context": "mtfa",
        "requested": requested_result,
        "htf": htf_by_interval,
        "htf_trend_alignment": htf_trend_alignment,
    }


async def _fetch_and_analyze(symbol: str, interval: str, candle_fetcher: CandleFetcher) -> Dict:
    df = await candle_fetcher(symbol, interval)
    return await analyze_smc_structure(df, interval)