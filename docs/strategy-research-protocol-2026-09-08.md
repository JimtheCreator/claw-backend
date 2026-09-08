# Strategy recovery study — declared before results, 2026-09-08

Objective: discover whether a small, explicit family of causal entry models
deserves further validation. Not a profitability promise or a deployment switch.
The current production planner and all evidence/exit defaults remain unchanged.

## Source interpretation

- [IG's multi-timeframe example](https://www.ig.com/sg/trading-strategies/introduction-to-multi-time-frame-analysis-220929)
  separates higher-timeframe location from the lower-timeframe trigger and warns
  against waiting for every timeframe to align. Support alone is not an entry.
  Its 1h/4h pairing motivates this study; its examples are not performance evidence.
- [IG on completed bars](https://www.ig.com/uk/view-ig/matching-time-frames-to-build-a-trading-system--37480-170329)
  motivates aligning context by bar CLOSE time, never unfinished HTF candles.
- [CME on support/resistance](https://www.cmegroup.com/education/courses/trading-and-analysis/support-and-resistance.hideSubnav.educationIframe.html)
  describes pivots and trendlines as possible reaction areas, not guaranteed reversals.
- [FXOpen's SMC description](https://fxopen.com/blog/en/the-smart-money-concept-basics-and-strategies/amp/)
  supplies pattern vocabulary, not proof that OHLCV reveals institutional intent.
- [Time-series momentum research](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
  concerns much longer horizons and other markets. It does not validate hourly
  crypto SMC or the rejected indicator gate.
- [Bailey et al. on overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)
  motivates limiting trials and freezing selection before later-window outcomes.

## Fixed experiment

Eight preselected, currently surviving liquid spot symbols: BTC, ETH, BNB, XRP,
ADA, DOGE, LINK, LTC (all USDT). This has survivorship and cross-symbol dependence;
it is not a random universe or eight independent replications.
Requested timeframe 1h, HTF 4h, stride ONE. No threshold search.

Development: 2024-01-01 to 2025-01-01. Validation: 2025-01-01 to 2026-01-01.
These histories overlap earlier inspected data and are not pristine holdouts.
Final temporal holdout: 2026-03-01 to 2026-09-01, beyond prior studies' cutoff.
Hourly archive warmup starts 2023-11-01. All archives SHA256 verified.

Three distinct research entry models, each independently ON/OFF (six trials):

1. `structure_break`: existing confirmed BOS/CHoCH on the signal bar. Stop beyond
   the latest confirmed opposing local pivot plus 0.25 ATR(14).
2. `sweep_reclaim`: wick crosses a prior 24-bar extreme and closes back inside;
   within the next three bars, close breaks the sweep candle's other extreme.
   Stop beyond the sweep extreme plus 0.25 ATR. This is a price-pattern proxy,
   not proof of stop orders or institutions. An invalidated sweep cannot trigger.
3. `trendline_break`: join the last two confirmed lower highs (long) or higher
   lows (short). Trade the first close through the extrapolated line after both
   anchors became knowable; maximum second-anchor age 24 bars. Stop beyond
   the latest confirmed opposing local pivot plus 0.25 ATR.

MTFA ON additionally requires 4h structural direction agreement and a price
reaction location: during the last 12 hourly candles, price must have intersected
the latest available 4h support (long)/resistance (short), or its latest same-side
broken structural level, within 0.25 of 4h ATR. These are structural POIs, not
new OB/FVG detections. OFF never uses HTF features. No indicator checklist gates.

Every signal enters at the NEXT hourly open, not a future retest or the already
known closing price. Invalid/gapped-through stops or targets cancel entry. Target
is fixed 2R from the signal close, not claimed to be observed liquidity. Require
at least 1.5R remaining at the next open. 48-bar maximum hold, full exit, no BE,
no overlapping positions within each symbol/policy, no new entry on an exit bar.
These research execution rules differ from production retest execution; results
must not be called an apples-to-apples improvement over the old backtest.
Both-hit bars stop first; stop gaps fill at the adverse open. Charge 10bps fee
and 2bps adverse slippage per side. Report stress at 10+5bps per side and an
additional conservative 5bps/day carry sensitivity for all directional trades.
Spot-data shorts are hypothetical: no venue funding, borrow, liquidation or
order-book replay. Long-only and short-only outcomes must also be reported.

All positions close inside each window, with 48-hour end embargo; no cross-window
outcomes. Missing hourly data aborts the symbol instead of bridging a gap.

## Selection and rejection (frozen before holdout)

Development eligibility: >=100 fills, positive mean net R, profit factor >=1.10,
at least 5/8 symbols positive, and lower 95% weekly-cluster bootstrap bound on
mean R >0. Pick the highest mean R among eligible policies, lexical tie break.
Freeze one policy or NONE before computing validation and final results.

Promotion remains FALSE in this study. Even the selected policy must pass both
later windows with >=100 fills, positive lower bootstrap bound, PF >=1.10,
5/8 profitable symbols, and positive stressed mean R before being a candidate
for venue-specific forward paper testing. No replacement winner may be chosen
from validation or holdout. Report all six policies for diagnosis only; they
consume this holdout for future research. Include mean gross/net R, net win rate,
profit factor, median stop bps, monthly results and per-symbol results. Weekly
clusters pool simultaneous markets to avoid pretending correlated trades are
independent. Bounds are descriptive, not a guarantee or a full correction for
multiple testing. Do not sum symbol risk into an account-return claim.

No changes to `indicators_v1`, `evidence_policy`, MTFA isolation, or production
planner thresholds. A losing result means reject, not tune until it wins.
