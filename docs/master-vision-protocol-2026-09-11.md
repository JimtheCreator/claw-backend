# Master-vision shapes v2 — preregistration, 2026-09-11

Implements `claw-backend-master-vision-spec.md`. Freeze this file and the complete
source fingerprint before outcomes. No detector edits, LLM calls, orders, sizing,
or live promotion. The old `shadow_v1` files remain a reproducible rejected
baseline; **momentum_flow_v1 is not in the new pool**. New policy `shadow_v2` is
opt-in research only; default `legacy` and live chart decisions stay unchanged.

## Three jobs, with honest strategy-specific applicability

Requested interval is the intermediate analysis/chart interval. Macro is its
nearest configured higher rung. Execution is a distinct, strictly finer series,
selected nearest-first from a separate declared downward ladder. 4h research
therefore means **1d macro → 4h intermediate → 1h execution**. All candles must
have closed by decision time. Macro zones must exist before the intermediate
reaction; that reaction must close before the execution candle opens. Reuse the
existing SMC predicate unchanged on that execution candle. No same-bar reaction
can retroactively authorize a trigger. No downsampling into invented fine bars.

Standalone TSMOM and VWAP do not need macro zones or macro agreement. Their
structural/zone role is N/A or their own VWAP band, not a hidden HTF vote. They
still evaluate their own trigger on the genuinely finer closed series. Requiring
a separate BOS for either would recreate the unwanted bundle. A persistent
TSMOM sign may fire again after an earlier paper trade closes; no channel-cross
prerequisite is smuggled in. Missing finer history means unavailable, not a
same-timeframe fallback. 1m has no supported finer execution series: explicit
unavailability, never pretend 1m serves two jobs.

## Frozen independent rules (no parameter search)

* **smc_location_v1**: same three checks and risk as the prior implementation:
  anchor POI, intermediate reaction, current execution displacement BOS/CHoCH.
  Existing top_down_context supplies causal POIs/reactions. Structural stop
  plus 0.25 execution ATR; nearest opposing confirmed execution pivot target.
  Only frame roles change; no new confirmation condition is added.
* **tsmom_v1**: all existing 21/63/252-day own-return horizon signs must be
  available and unanimously positive or negative on the execution series.
  Those signs are the entire signal. No channel, VWAP, CVD, SMC, regime or HTF
  requirement. Stop 2 execution ATR14 from reference close; target 2R from it.
  These ATR stops/short holds and horizon agreement are **project hypotheses**,
  not a replication of AQR's monthly 12-month-excess-return futures strategy.
* **vwap_reversion_v1**: execution close at/beyond ±2 volume-weighted population
  standard deviations of typical price around cumulative UTC-session VWAP.
  Complete session origin, >=3 execution bars, positive variance/volume required.
  Above upper band → short; below lower → long. No reversal candle/BOS required.
  Target is VWAP **frozen at signal**, not tomorrow's moving VWAP. Stop is beyond
  the farther of the ±3SD band and current price, plus 0.25 execution ATR14.
  Exit no later than the UTC session boundary. No assertion that a 2SD excursion
  must revert. This declared weighted-price variance is not claimed identical
  to TradingView's documented standard deviation of cumulative VWAP values.

Each uses the existing account-independent risk evaluator: 30bps stop floor,
1.5 minimum gross target R, 10bps fee + 2bps slippage per side, no widening to
pass costs. Risk failure is distinct from the strategy signal not firing. Max
48 execution bars (VWAP additionally session-bounded). Next-open fill with fixed
signal-time SL/TP; recheck risk after gaps. Both-hit bars stop first; adverse stop
gaps use the open; no favorable TP gap benefit. Costs are on traded notional.
Stress: 10+5bps per side plus 5bps/day carry, scaled by actual bar duration.
Intrabar execution is an OHLC approximation, not tick-level fill validation.

True taker CVD is optional **diagnostic confirmation** (+1 aligned, -1 opposed,
0 unavailable), never an eligibility/arbitration gate or win probability.
Missing/proxy flow earns no confirmation. RSI/MACD divergence is not used by
these candidates; no new unvalidated veto. Report CVD-aligned/opposed/unavailable
cohorts independently, not a combined confidence score. No optional observation
may silently change any strategy's trigger, prices, eligibility or selection.

## Availability, arbitration, and product boundary

New shadow adapter automatically fetches actual macro and finer data. An
internal `SMC_SHAPE_MTFA_MODE=auto|off` controls QA only; it never reads the public
MTFA checkbox. OFF does not fetch, score or emit HTF context. SMC becomes
unavailable if macro/middle history is missing; standalone shapes remain testable.
No inheritance of an earlier request's frames. Full execution momentum history
uses the existing persistent cache and bounded backfill, without shorter horizons.

Frozen arbitration is **surface-the-conflict / no blending**. One eligible shape
is selected. Opposite directions → conflict/WAIT. Multiple agreeing shapes →
agreement with each complete plan shown, but **no single selected plan** until
one independently validated strategy exists. Do not rank by invented confidence,
regime preferences or this run's evaluation returns. Empty pool → plain no setup.
The old fallback remains baseline-only; trendline-only is a separately optional
future experiment, not inserted into these shapes.

Part 3.6 takes precedence over shipping UI changes: public toggle removal and a
new customer-facing planner are **not deployed** before validation. Future public
contract is auto availability, with the requested chart interval preserved and
the finer execution interval/time clearly identified. Shadow contains per-shape
reasoning/price/time facts for research preview; cannot overwrite the live plan.

## Evaluation and promotion bar

Eight symbols: BTC/ETH/BNB/XRP/ADA/DOGE/LINK/LTC USDT spot candles, hourly execution
stride 1, requested 4h. Checksum-verified Binance archives. Warmup 2023 after its
last pre-development gap; require full 6,049-bar execution momentum warmup.
Development 2024; freeze selection **before loading later archives**. Evaluation
2025; reused diagnostic March–August 2026. Report each isolated strategy plus
unblended arbitration, per symbol/direction, signal/failure counts, net/gross R,
win rate, PF, weekly-cluster bootstrap CI and CVD cohorts. One open paper trade
per symbol/arm; no overlapping same-arm fills. No claim of account returns or
actual spot short availability. No tuning after inspecting outcomes.

Development AND evaluation must each satisfy: >=100 fills; mean net R >0;
weekly-cluster 95% lower bound >0; PF>=1.10; >=5/8 positive-mean symbols; positive
mean stress R. Select at most one independent development winner by mean net R,
strategy-name tie break. If none pass, freeze NONE; evaluation cannot choose a
replacement. Arbitration is diagnostic and cannot rescue a rejected shape.

**All these historical periods have been inspected before. No historical result
here authorizes promotion.** A genuinely untouched prospective paper window is
still required. Reserve 2026-10-01 through 2027-01-01 UTC for a candidate frozen
before October, subject to the same bar. If there is no development/evaluation
winner, that window cannot promote one. A reservation is not a scheduled monitor
or proof it has run. Future validation must use archived inputs and immutable
decision-time versions, not hindsight chart screenshots.

## Source boundaries

[AQR original paper summary](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
supports own-return momentum, not our extra gates or crypto profitability.
[AQR original monthly data description](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data)
states 12-month lookback / one-month holding; this experiment is an adaptation.
[TradingView VWAP documentation](https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/)
documents VWAP/bands, not validated mean-reversion expectancy for this rule.
[IG top-down example](https://www.ig.com/sg/trading-strategies/introduction-to-multi-time-frame-analysis-220929)
illustrates context then a finer trigger; it does not require every timeframe
to agree. The attachment's claim that every detector is proven correct remains
an assertion, not certification by this implementation work.
