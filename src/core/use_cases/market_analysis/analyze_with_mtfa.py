import asyncio
from typing import Awaitable, Callable, Dict, Optional

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
    requested_result: Optional[Dict] = None,
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

    `context` is explicit: ``disabled`` when the user turned MTFA off and
    ``no_higher_timeframe`` at the top of the ladder. Consumers must not
    have to infer the reason higher-timeframe data is absent.

    When the caller has not already produced the requested-TF result, it
    and every HTF fetch+analysis run concurrently. The live task passes its
    requested result so the analysis is based on the same candle snapshot
    it sends to the client and no duplicate fetch can create a race.

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
        if requested_result is None:
            requested_result = await analyze_smc_structure(
                await candle_fetcher(symbol, interval), interval
            )
        return {
            "symbol": symbol,
            "interval": interval,
            "context": "disabled" if not mtfa_enabled else "no_higher_timeframe",
            "requested": requested_result,
            "htf": {},
            "htf_requested": [],
            "htf_unavailable": {},
            "htf_trend_alignment": {},
        }

    task_intervals = htf_chain if requested_result is not None else [interval, *htf_chain]
    tasks = [_fetch_and_analyze(symbol, tf, candle_fetcher) for tf in task_intervals]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # The task already has the requested-TF result. Reusing it keeps the
    # MTFA comparison tied to the exact candle snapshot delivered to the
    # client and avoids a duplicate data fetch. Direct callers may omit it.
    if requested_result is None:
        requested_outcome = outcomes[0]
        if isinstance(requested_outcome, Exception):
            logger.error(
                f"[analyze_with_mtfa] Requested TF fetch/analysis failed for "
                f"{symbol} {interval}: {requested_outcome}"
            )
            raise requested_outcome
        requested_result = requested_outcome
        htf_outcomes = outcomes[1:]
    else:
        htf_outcomes = outcomes

    htf_by_interval: Dict[str, Dict] = {}
    htf_unavailable: Dict[str, str] = {}
    for tf, outcome in zip(htf_chain, htf_outcomes):
        if isinstance(outcome, Exception):
            logger.warning(
                f"[analyze_with_mtfa] HTF fetch/analysis failed for {symbol} {tf}: "
                f"{outcome} - excluding this HTF, other timeframes unaffected."
            )
            htf_unavailable[tf] = type(outcome).__name__
            continue
        htf_by_interval[tf] = outcome

    requested_trend = requested_result["market_structure"].trend
    htf_trend_alignment: Dict[str, Optional[bool]] = {}
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
        "htf_requested": htf_chain,
        "htf_unavailable": htf_unavailable,
        "htf_trend_alignment": htf_trend_alignment,
    }


async def _fetch_and_analyze(symbol: str, interval: str, candle_fetcher: CandleFetcher) -> Dict:
    df = await candle_fetcher(symbol, interval)
    return await analyze_smc_structure(df, interval)
