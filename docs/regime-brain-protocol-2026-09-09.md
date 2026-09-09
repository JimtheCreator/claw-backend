# Independent-strategy brain v1 — frozen specification, 2026-09-09

Implements Architecture B in the supplied `brain-regime-aware-smc-momentum-architecture-v2.md`.
The attachment is a proposal, not proof that detectors or strategies are profitable.
No LLM, order placement, account sizing, learned weights or production promotion.
`SMC_BRAIN_POLICY=shadow_v1` opts into diagnostics only; default `legacy` leaves
decisions and request cost unchanged. Unknown policy names fail configuration.
The new brain never calls the old entry gate and never changes `indicators_v1`.

## Fixed independent hypotheses (no parameter search)

All input observations are closed candles. Requested interval stays unchanged.
Each strategy records individual checks, missing data, entry trigger and risk.

* **smc_location_v1**: MTFA ON, first two higher rungs of the configured ladder
  both present (middle/anchor). Anchor confirmed OB or FVG, still not invalidated
  by a close beyond its far edge, formed within 180 anchor bars. Anchor's own
  structural bias must agree with the zone; other timeframe trends are NOT votes.
  A completed middle candle in the last three middle bars must have touched the
  already-known zone and closed back outside its near edge in the intended
  direction. After that reaction, the current requested candle must close a
  BOS/CHoCH in that direction with body >=0.8 prior ATR14 and >=60% of range.
  No pending retest is interpreted as an observed entry. The three mandatory
  jobs are anchor location, middle reaction, and local displacement break.
* **momentum_flow_v1**: available in either MTFA state. All three existing TSMOM
  horizons must be present and agree in sign. Its OWN trigger is the first close
  beyond the previous 20-bar HIGH/LOW channel (not a fractal BOS). Confirm with
  directional UTC-session VWAP side, >=3 bars in that session, and real taker
  delta in direction on the trigger AND summed over the last three bars. All
  three deltas must be genuine; proxies cannot qualify. VWAP/CVD are confirmations
  of this independent trigger, not extra vetoes on SMC. Divergence and volume
  profile are not mandatory: the document provides no validated new veto rule.
* **local_fallback_v1**: only if MTFA is OFF and complete TSMOM is unavailable.
  All FOUR checks mandatory: current BOS/CHoCH, the same displacement definition,
  a preceding opposite-side confirmed-pivot wick/reclaim in the preceding ten
  bars, and a still-valid local OB/FVG overlap linked to the current break leg
  (formation within three bars). Local zone age <=64 bars. This is explicitly
  a confirmed-pivot sweep, not a claim of institutional stop orders. Missing
  evidence is a rejection, never a free pass. TSMOM disagreement is NOT missing
  data and cannot activate this fallback. MTFA fetch failure while ON does not
  silently change the user's mode or activate Tier 3.

## Shared risk contract

Reference entry is the trigger close; historical fill is NEXT open, subject to
the same risk checks after gaps. Stop beyond latest confirmed opposing local
pivot plus 0.25 ATR14. No artificial stop widening to clear fees. SMC/fallback
target the nearest unconsumed opposing confirmed local pivot. Momentum uses a
declared 2R target, explicitly a risk multiple, NOT observed liquidity. Minimum
remaining gross RR 1.5. Stop floors 30bps SMC/momentum and 60bps fallback. Single
full exit, no breakeven, staging or position size. Max 48 requested bars.
Fees 10bps + slippage 2bps per side; stress 10+5bps and 5bps/day carry. Return
stop bps, target R, estimated cost R at stop/target and net target R; none is an
expectancy estimate. Stops gap adversely; both-hit bars stop first. Reject fills
with invalid stop/target/cost geometry. Each symbol/arm has at most one position.

## Regime and arbitration, frozen before outcomes

Wilder ADX14 >=25 = trending, <25 = non-trending, insufficient = unknown. ATR14
as percent of price with a trailing 252-bar percentile separately describes
volatility (>=75 expanding/high, <=25 contracting/low, otherwise normal). These
are unvalidated descriptive hypotheses, not proof of a strategy's suitability.
Opposing eligible directions ALWAYS yield conflict/WAIT, with both ledgers.
For eligible strategies agreeing on direction: trending prefers momentum, then
SMC, then fallback; otherwise SMC, then momentum, then fallback. No weights,
first-match, performance claims, random choice or missing-indicator vote.
Regime is local-only. MTFA OFF strips HTF inputs before evaluation and emits no
HTF zones/trends/derived prices. A disabled strategy has only a reason, no HTF
evidence checklist. Pool evaluation is independently testable before arbitration.

## Data and evaluation

Keep TSMOM's existing interval horizons (21/63/252 days, NOT literal 1/3/12
calendar months on a 24/7 market). Do not shrink them. Unknown intervals fail
explicitly rather than inherit daily windows. Momentum history is a dedicated
same-timeframe cache, timestamp-bounded and contiguous, independent of chart
lookback and MTFA. Eight 1000-bar pages / 30 seconds maximum foreground recovery;
shorter intervals need a separately invoked cache warmup. Cache survives workers.
Missing/partial/gapped coverage is reported and cannot claim complete momentum.

Hourly research: BTC/ETH/BNB/XRP/ADA/DOGE/LINK/LTC USDT, stride 1; warmup 2023,
development 2024, evaluation 2025, reused diagnostic March–August 2026. HTF middle
4h / anchor 1d (first two configured rungs; weekly not a required third vote).
Binance checksum-verified archives, real field 9, prefix-causal features and HTF
availability at CLOSE time. Each strategy first independently; additionally
report frozen arbitration ON/OFF and natural OFF coverage. Report isolated Tier3
with deliberately unavailable momentum as a STRESS cohort, not natural coverage.
Data preflight found a one-hour March 24, 2023 Binance gap. Discard warmup before
the final pre-development gap and restart every feature; require >=6,049
contiguous warmup bars before 2024. Any gap inside an evaluation window aborts
that symbol. No fabricated candles or returns spanning exchange outages.
At least 100 fills, mean net R and weekly-cluster 95% lower bound >0, PF>=1.10,
5/8 profitable symbols, positive stress net R required in dev and evaluation.
Persist source/protocol hashes and dev selection before later-window simulation.
No replacement winner after evaluation. All arms are experimental.

**There is no untouched historical holdout here.** Prior studies consumed 2024,
2025 and March–August 2026. Report reuse plainly; no positive reused result can
promote a strategy. A subsequent preregistered, venue-specific forward paper
window is required. Do not label a small September sample independent validation.

## Primary-source interpretation

[AQR's original TSMOM research](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
supports studying own-return momentum, not hourly crypto profitability.
[IG's top-down example](https://www.ig.com/sg/trading-strategies/introduction-to-multi-time-frame-analysis-220929)
distinguishes reaction location from an observed trigger, not universal TF votes.
[Binance's kline contract](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)
provides base taker-buy volume at index 9. Institutional-intent explanations in
the attachment are hypotheses, not facts inferable from OHLCV.
