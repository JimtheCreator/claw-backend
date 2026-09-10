# Top-down analysis audit and validation — 2026-09-06

## Long/Short candle badges, no projection arrow — 2026-09-10

`conditional-forecast-v7` removes the directional projection line and arrow
entirely. The user's reference meant a green **Long ▲** candle badge or a red
**Short ▼** candle badge, not a diagonal path to TP. Long is positioned below
the latest candle low; Short above its high, with the compact notched label
style. Only the existing supported scenario determines direction. The same
change applies to the experimental next-move presentation (`next-move-v3`).

Current-reference TP/SL shading, price flags, confirmation levels, entry status,
evidence, and planner decisions are unchanged. A scenario-only badge is explicitly
not entry confirmation; invalid/missing scenarios do not gain a Long/Short badge.
There are no historical signal badges fabricated from the reference image.
Observed BOS/CHoCH annotation pointers remain evidence annotations, not forecasts.
All **149 unit tests passed** with the existing unrelated price-alert import
failure excluded. Tests assert removal of projection traces, badge direction and
placement, retained TP/SL geometry, and unchanged input plans. The public-data
preview now detects rendered scenarios by their badge, not by a removed line.
ETH 4h (Long) and BNB 1h (Short) real-data PNGs were visually verified, with
ON/OFF smoke renders completed for both. Artifacts are in workspace
`outputs/conditional-forecast-v7` and `outputs/conditional-forecast-v7-bnb`.

## Current-candle shading and price flags — 2026-09-10

At the user's explicit request, `conditional-forecast-v6` changes the chart's
**visual reference** from pending activation to the latest closed candle. The
green area measures current close to the unchanged scenario target; the red area
measures current close to unchanged invalidation. The solid illustrative arrow
starts at that same candle/time/price. Neither the reference nor the arrow is an
approved entry, a claim that confirmation has occurred, or a calibrated price
prediction. The planner, its gates, and all price levels remain unchanged.

This fixes the ETH 4h example where activation 2,523.30 and TP 2,523.97 left only
0.67 of reward shading despite the snapshot close being well below both. The
chart keeps the original **activation-based** R:R and its below-minimum warning;
it does not replace that ratio with the visually larger current-to-target ratio.
Current reference, confirmation, TP and SL/invalidation use separate left-pointing
colored flags. Collision spacing moves labels only; leaders anchor displaced
labels to the exact price. There is no invented retest leg or moved target.

If the latest close has already reached/passed the target or invalidation, no
current-reference arrow or reversed reward/risk boxes are drawn; the chart
explicitly requests reassessment. This check is presentation-only and does not
mutate the stored plan. Missing/invalid levels still cannot fabricate a forecast.
**149 unit tests passed**, excluding the same unrelated price-alert import
failure. Tests include mirrored long/short geometry, current/activation separation,
near-identical TP/confirmation labels, MTFA isolation, and unchanged input plans.
Actual ETHUSDT 4h ON/OFF PNGs were rendered and visually inspected under workspace
`outputs/conditional-forecast-v6`. No claim of a new trading edge follows from
these presentation changes.
The local worker was confirmed idle and gracefully restarted with unchanged
queues/pool/concurrency; it reported ready on this renderer. Fresh requests are
needed because existing chart images are immutable.

## Forecast layout follow-up — 2026-09-10

`conditional-forecast-v5` removes the eight-candle visual gutter: the forecast
arrow and both TP/SL bands start exactly at NOW on the time axis. The arrow still
starts at the supplied activation **price**, not a newly invented executable
entry at the last close. No planner levels, evidence or eligibility changed.
The route is now a continuous solid arrow with a larger directional head and no
starting dot; labels are offset clear of the arrowhead. The experimental
next-move presentation uses the same arrow style (`next-move-v2`).
The shaded future pane and explicit conditional captions distinguish forecasts
from observations; the obsolete solid-facts/dashed-forecast legend is removed.
Tests cover the boundary at 1m/1h/4h/1d, both MTFA modes, long/short directions,
unchanged input plans and serialized PNG payloads. ETHUSDT 4h ON/OFF PNGs were
rendered for visual verification; fresh analyses are required to replace old
immutable images.
Verification: **144 unit tests passed** (same unrelated price-alert collection
failure excluded). The idle local worker was gracefully restarted with its
existing queues/pool/concurrency and reported ready on the updated renderer.

## Forecast visibility and TP/SL rendering — 2026-09-10

The immediate display defect was in `analysis_chart_presentation.py`: a valid
conditional scenario could include a trigger, target and invalidation, but the
renderer deliberately showed only its trigger. WATCH therefore hid useful
scenario information. `conditional-forecast-v4` restores that supplied path
without promoting the scenario to an approved entry.

- `trade_plan.py` preserves a separate `forecast_scenario` before entry-gate
  early returns. Where higher-timeframe validation is unavailable or mixed,
  an available local structural scenario is explicitly local-only and does not
  populate order levels or change WAIT into BUY/SELL. Actual confirmed pivots
  and an unswept target are required; missing evidence does not invent levels.
- The chart draws green target/reward and red stop/invalidation regions only
  for correctly ordered, complete levels. Its dashed path begins at the
  activation level in the future pane, **not at today's price**: it does not
  assert that price will reach a pending entry or invent a retest sequence.
  Existing confirmation/retest requirements remain visible in the caption.
- Pending setups say ENTRY PENDING; non-approved scenarios say NO ENTRY
  APPROVED. Targets and invalidation are scenario levels, not placed orders.
  Invalid/incomplete geometry has no fabricated arrow or risk boxes. A gross
  reference reward/risk below the planner's 1.5R minimum is called out explicitly,
  not concealed by favorable-looking shading.
- Observed structural facts remain separate from the dashed conditional
  forecast, whose timing and outcome are not claimed. Entry/exit thresholds,
  evidence policy, indicator gates and research promotion rules are unchanged.

Verification: **136 unit tests passed**, excluding the existing unrelated
`test_price_alert_manager.py` collection failure (legacy notification import).
Public closed Binance BTCUSDT/ETHUSDT/BNBUSDT 1h snapshots rendered successfully
with MTFA ON and OFF: all six had a supported forecast. Final BTC/BNB PNGs were
also visually checked for readable labels, price bounds and future-only risk
regions. These are rendering checks, **not evidence of profitable forecasts**.
The reproducible read-only smoke tool is `scripts/preview_analysis_forecasts.py`;
workspace artifacts are under `outputs/conditional-forecast-v4` and
`outputs/conditional-forecast-v4-final`. Previously generated PNGs are immutable;
fresh analysis requests are needed after the worker loads this renderer.
The single local worker was confirmed idle, gracefully restarted with the same
queues/pool/concurrency, and reported ready with the updated renderer loaded.

## Independent regime-aware brain — implemented and tested, 2026-09-09

The attached Architecture B proposal is implemented as a separate deterministic
strategy pool, **not a rewrite or another indicator gate on the old planner**.
The exact preregistration is `docs/regime-brain-protocol-2026-09-09.md`.
Its causal stories are hypotheses; the attachment's assertion that every
detector is correct is not an independent certification of that claim.

### What changed

| File | Responsibility |
|---|---|
| `strategy_brain.py` | Independent HTF-location SMC, momentum/flow, and missing-context local fallback; each has its own eligibility ledger. Conflict always yields WAIT. |
| `strategy_features.py` | Shared causal features for live/replay; existing pivot/BOS/OB/FVG detectors, own 20-bar momentum trigger, true taker delta, UTC VWAP, three-job anchor/middle/local context. |
| `regime_engine.py` | Wilder ADX14 and trailing relative-ATR percentile; regime separate from MTFA availability. |
| `strategy_risk.py` | Structural stop + 0.25 ATR, declared 30/60bps floors, target source, gross/cost/net target R; no account sizing or automatic trades. |
| `momentum_history.py` | Persistent, same-interval SQLite cache, closed-snapshot cutoff, contiguous full horizons, bounded async exchange recovery. |
| `brain_shadow.py` / `tasks.py` | Opt-in diagnostics attached as `trade_plan.brain_shadow`; cannot replace the existing plan/chart. OFF ignores HTF inputs. |
| `scripts/data/warm_momentum_history.py` | Explicit resumable prewarm for longer histories, using the shared Binance rate limiter. |
| `tests/backtesting/run_brain_research.py` | Isolated strategies first; frozen dev selection, five diagnostic arms, per-fill evidence and archive/source hashes. |

SMC uses the first two higher rungs: middle reaction at an already-available
anchor OB/FVG, followed by a current local displacement break. It does NOT demand
universal timeframe-direction agreement. Momentum can enter the research pool
with MTFA ON or OFF and does NOT require BOS; it requires its own channel-break
trigger, all configured horizon signs agreeing, directional UTC VWAP, and genuine
trigger/three-bar taker delta. These are this experiment's rules, not validated
institutional behavior. Divergence/Profile are not new vetoes. Tier 3 requires
all four declared local checks and a 60bps floor, only with MTFA OFF and complete
momentum unavailable. Mixed momentum is not missing history and cannot unlock it.

Risk targets for SMC/fallback are confirmed opposing pivots, not a claim to know
resting orders. Momentum's 2R target is labeled a risk multiple, not liquidity.
The old planner, `indicators_v1`, evidence policy and MTFA isolation code were
not modified. New research cannot be promoted through an environment variable.

### History recovery verified

Real public Binance BTC/ETH/BNB 1h caches were warmed to **6,049 closed contiguous
bars each**, including real field-9 taker-buy volume. All three needed six pages
beyond the initial snapshot. Subsequent ON/OFF adapter checks reused the cache
with zero additional historical pages. Each strategy's momentum reading was
identical ON/OFF. All six current snapshots still rejected entry; a working
history cache is not evidence of a profitable strategy or an always-on signal.

Foreground history recovery is bounded at eight pages / 30 seconds. A 1m full
horizon needs 362,881 bars; it is NOT covered by eight pages, and requires the
explicit prewarm command. Short listings/gaps/budget exhaustion remain explicit
unavailability. No window is silently shortened or replaced with an HTF series.
The old horizons represent 21/63/252 days, not literal calendar months in 24/7
crypto. The new strategy requires all three; the legacy engine's partial-horizon
behavior remains unchanged.

### Frozen research results: no promotion

Eight symbols, hourly stride 1, development 2024, evaluation 2025, reused
diagnostic March–August 2026. Costs 10bps fee + 2bps slippage **per side**; stress
10+5bps plus 5bps/day carry. No outcome-driven threshold changes. Development
selection **NONE** was persisted before later-window simulation.

| Independent strategy | 2024 N / mean net R | 2025 N / mean net R | Reused 2026 N / mean net R |
|---|---:|---:|---:|
| HTF-location SMC | 6 / −0.906 | 15 / −0.686 | 4 / −0.480 |
| Momentum/flow | 402 / +0.048 | 443 / −0.007 | 226 / −0.219 |
| Local fallback, forced missing-momentum stress | 5 / +0.341 | 8 / −0.200 | 3 / −0.025 |

Momentum's developmental improvement does not establish an edge: its weekly
95% interval was [−0.091, +0.183]R, then the mean turned negative in both later
windows. It failed the predeclared eligibility bar. SMC and fallback remain
**drastically under-sampled**; positive fallback development R from five fills
is not validation. Do not relax gates after seeing these outcomes and call the
same windows an untouched test.

Arbitration ON averaged +0.037R / −0.022R / −0.223R across those windows; OFF
matched standalone momentum because complete horizons were available and Tier3
was naturally disabled. Combining strategies did not rescue them. The ADX rule
is a deterministic routing implementation, not an empirically validated router.
See per-regime/symbol/direction breakdowns and every simulated fill in workspace
`outputs/independent-brain-v1/{report.md,summary.json,dev-*.json,later-*.json}`.
Independent samples must not be added to overlapping arbitration arms as if
those were new trades. Forced fallback stress is not natural fallback coverage.

All these historical periods have been inspected before. **No genuinely untouched
holdout exists in this run.** Historical features expand from contiguous 2023
warmup; the live adapter keeps the user's local chart lookback and extends only
momentum history. This is not an exact replay of an individual app request.
The March 2023 archive gap was handled by restarting warmup after it, not by
fabricating missing candles. No evaluation-window gaps were bridged.

### Operation and verification

`SMC_BRAIN_POLICY` defaults to `legacy` (no additional fetch or changed decision).
`shadow_v1` enables stored diagnostic candidates after a normal worker restart;
there is deliberately no unvalidated production mode. This turn did not change
the running worker's policy or restart it. The user-facing chart still uses the
existing planner. An optional shadow failure cannot prevent chart delivery.

From the backend root, explicit cache warmup:

```sh
PYTHONPATH=src:. .venv/bin/python -m scripts.data.warm_momentum_history --symbols BTCUSDT ETHUSDT BNBUSDT --interval 1h
```

Replay with the same `--cache` and `--output` directories, first `--phase dev`,
then `--phase later`, via `python -m tests.backtesting.run_brain_research`.
Code/protocol hashes must match the frozen selection and checkpoints.

New tests cover prefix equivalence, actual HTF reactions known only after middle
closes, poisoned HTF OFF isolation, no-BOS momentum, proxy/partial CVD exclusion,
Wilder initialization, persistent history/cursors/gaps/timeouts, gap-fill risk
revalidation and agreement/conflict arbitration. The full unit suite currently
has an unrelated collection failure in `test_price_alert_manager.py` importing
the nonexistent `infrastructure.notifications`; excluding that legacy file
allows the remaining tests to run. The failing alert import was not changed.
Final verification: **128 tests passed** with that one collection-broken file
excluded; compilation and `git diff --check` also passed.

## Entry rejection no longer erases market context — 2026-09-09

Read-only inspection of saved September 8 results confirmed three different
causes, not a broken SSE stream: ETH 1m exited on mixed HTF context before any
scenario existed; BTC 1h/4h exited on local/HTF disagreement; BTC MTFA OFF failed
the local setup evidence checklist. The previous renderer also deliberately
removed unsupported approach projections. Together these left users with WAIT
and little visual explanation. This was a usability regression, not evidence
that a directional trade should have been approved.

`market_read.py` now builds requested-timeframe observations before the planner's
entry-gate returns: nearest unbroken confirmed swing support/resistance, latest
observed BOS/CHoCH and its age, structural direction, and the next structural
check. No HTF data enters this local read. Provisional pivots and already-broken
levels are excluded. Missing levels remain missing rather than invented.

`market-read-v3` plots at most those two local levels and the last observed break,
even when an incomplete/mixed HTF gate prevents the entry checklist from running.
It keeps the blocking reason visible and distinguishes local trend from entry
permission. Very distant local levels are disclosed in the caption rather than
flattening the candles; existing evidence-anchor caps remain disclosed.
It does not turn a structural checkpoint into a BUY/SELL target or draw an
unsupported approach arrow. No entry threshold, HTF rule, indicator policy or
trade economics changed. This is not a profitable-strategy fix.

Live public-data diagnostics covered BTCUSDT, ETHUSDT and BNBUSDT, 1h, with MTFA
ON and OFF, using 750 requested bars per timeframe (not an exact replay of each
saved app request). All SIX remained entry-WAIT under the unchanged rules; all
retained market context. An actual ETH PNG was rendered and visually inspected.
Artifacts: workspace `outputs/market-read-v3-smoke.json` and
`outputs/market-read-v3-eth.png`. New tests cover early-gate returns, observed
levels, unconfirmed/broken pivot exclusion, unchanged entry rejection and MTFA
OFF isolation. Subscriber-ready trading signals remain unvalidated.

## Taker-volume ingestion and rejected-gate containment — 2026-09-08

Confirmed gap: real Binance taker-buy volume was discarded before CVDEngine.
There is no live `data_loader.py` in this repository. The actual path is
`market_data.py`/background backfills/websocket persistence -> MarketDataEntity
-> InfluxDB -> `data_access.py` -> closed-candle DataFrame -> CVDEngine.

That path now preserves nullable `taker_buy_volume`: REST kline **index 9** and
websocket kline **V**, both base-asset volume (not index 10 / quote volume).
[Binance kline reference](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints).
Real per-candle delta remains `2 * taker_buy_volume - volume`. Invalid, negative,
over-total and missing values become unknown, not zero. Genuine zero is valid.
Influx writes the field when known and both raw chronological/reverse queries
retrieve it. Analysis bypasses downsampling; legacy display-only downsampled
records do not claim genuine taker totals.

Existing Influx records are not magically repaired by a schema change. Analysis
now makes at most one optional exchange read (<=1000 candles, one retry attempt,
15-second bound) when its requested snapshot lacks flow. It only attaches values
to identical timestamp/OHLCV observations and queues recovered closed records
through the existing persistence task. Failed/mismatched recovery leaves unknown
flow and does not turn an optional feature into a failed OHLCV request. No bulk
database deletion or historical full-store migration was performed.

CVDEngine continues to label any price-direction fallback explicitly. Result
metadata counts genuine versus approximate points; each cumulative point now
also records real/proxy/mixed provenance. A real delta after a missing candle
must not make a mixed cumulative sum look genuine. `indicator_evidence` still
excludes proxies from CVD confirmation.

The archive helper `download_month(..., include_taker_buy=True)` provides the
same field for future richer-input studies. Resampling sums it only with complete
coverage; missing constituent flow produces missing aggregate flow, not a partial
total. Default six-column historical reads remain unchanged to preserve previous
study input contracts. Previous hourly research had no CVD feature and no ML
training, so its negative results were not trained on a CVD proxy. They have not
been relabelled or rerun as richer-data results.

Gate audit: `analyze_smc_structure.py` only orchestrates structural detectors;
it has no VWAP veto. Under default `smc_v2`, VWAP/Profile are shadow evidence and
cannot alter ranking, eligibility, score or mandatory count. They contributed to
an extra standalone-count gate only in the rejected `indicators_v1` experiment;
neither was individually mandatory. The live task now explicitly selects
`smc_v2`, ignoring `SMC_EVIDENCE_POLICY` so an environment override cannot enable
that rejected gate. Explicit offline research calls retain `indicators_v1`
unchanged for reproducibility. No unspecified ML decision path was added.

Regression coverage includes REST cold fetch and refresh, websocket persistence,
Influx line-protocol/readback, formatter-to-CVD, optional cache recovery and failure,
archive field 9 and partial-resample semantics, proxy exclusion, and unchanged
default eligibility with VWAP/Profile agreeing, disagreeing or unavailable.
A read-only live Binance BNBUSDT smoke check verified three closed 1h candles:
three genuine CVD points, zero proxy points, all matching the field-9 formula.
This validates ingestion correctness, not CVD predictive value or profitability.

## Strategy recovery research — 2026-09-08: no profitable replacement found

The user correctly identified that delivery infrastructure is not a trading
edge. This study changes neither the UI nor production decisions. It adds a
fast, reproducible research path for entry mechanisms, reusing the existing
SwingStructureEngine and MarketStructureEngine. Its rules and rejection criteria
were written in `docs/strategy-research-protocol-2026-09-08.md` before results.

### Research interpretation

[IG's top-down worked example](https://www.ig.com/sg/trading-strategies/introduction-to-multi-time-frame-analysis-220929)
uses higher-timeframe location and a subsequent lower-timeframe event. It does
not establish that price will recover merely because a higher chart is bullish.
[CME's support/resistance discussion](https://www.cmegroup.com/education/courses/trading-and-analysis/support-and-resistance.hideSubnav.educationIframe.html)
treats pivots and trendlines as possible reaction areas, not certain destinations.
These suggest testable rules, not verified crypto returns.

The existing planner combines detections into a pending setup, but does not
observe the subsequent live entry/exit lifecycle. SMC vocabulary, a confluence
count, or a persuasive forecast image cannot bridge that gap. This experiment
therefore measures actual next-open entries and subsequent stop/target/time
outcomes. It tests reduced structural, sweep-reclaim and trendline models—not
every discretionary SMC/ICT interpretation, not a new OB/FVG combination, and
not a wholesale replacement of the original pipeline.

### Fixed comparison

- BTC, ETH, BNB, XRP, ADA, DOGE, LINK, LTC USDT; 1h execution / 4h context;
  every hourly close (stride 1). MTFA ON requires direction AND structural POI
  location; OFF has neither dependency. Each of three entry models has ON/OFF.
- Development calendar 2024; validation calendar 2025; final holdout March–August
  2026. Earlier studies inspected portions of development/validation. The final
  window is beyond their cutoff, but is now consumed by this experiment.
- Next-open fills, original structural invalidation plus 0.25 ATR buffer,
  fixed signal-time 2R target (not invented liquidity), 1.5R remaining minimum
  after an opening gap, maximum 48 hours. No retracement order, BE or staged exit.
- Costs: 10bps commission plus 2bps adverse slippage per side. Stress: 10+5bps
  per side plus 5bps/day carry. Spot history shorts remain hypothetical; no actual
  futures funding, leverage liquidation, borrow or exchange fill validation.
- Six policies only. Development selection required >=100 fills, positive mean
  net R and weekly-cluster lower bound, PF >=1.10 and 5/8 positive symbols.
  **NONE qualified.** That decision was persisted with source/protocol hashes
  before later-window simulation. No holdout winner was substituted.

### Final holdout results (March 1–September 1, 2026, exclusive end)

| Entry model | MTFA | Trades | Net win rate | Mean gross R | Mean net R | Profit factor | Stress net R |
|---|---|---:|---:|---:|---:|---:|---:|
| Structure break | OFF | 831 | 39.4% | +0.032 | -0.100 | 0.826 | -0.160 |
| Structure break | ON | 541 | 40.9% | +0.077 | -0.046 | 0.915 | -0.105 |
| Sweep then reclaim | OFF | 867 | 38.3% | +0.037 | -0.162 | 0.774 | -0.239 |
| Sweep then reclaim | ON | 329 | 39.2% | +0.121 | -0.091 | 0.871 | -0.170 |
| Trendline break | OFF | 845 | 39.8% | +0.040 | -0.174 | 0.743 | -0.255 |
| Trendline break | ON | 442 | 37.6% | +0.050 | -0.213 | 0.711 | -0.305 |

**Reject all six for production promotion.** All were net-negative in development,
validation AND final holdout. MTFA improves some final-window comparisons but
not consistently: structure ON worsened validation (-0.149R versus OFF -0.102R).
Trendlines did not supply a profitable fix. Larger samples address the earlier
N=2/N=16 problem but do not imply independence across markets or policy arms.

Median holdout stop distances were 126–232bps across these arms. Despite much
wider typical stops than the earlier low-timeframe study, costs still exceed
the small average gross gains; widening the sampling timeframe alone did not
create sufficient edge. Structure ON's holdout mean-R weekly 95% bootstrap
interval is [-0.194, +0.106], not a confidently positive edge. Sweep ON's is
[-0.232, +0.047]. Other full statistics, positive-symbol counts, monthly results
and observed long/short fill subsets are in the report. Direction subsets are
descriptive, not separate operational long-only/short-only backtests.

These entry/execution rules differ from production's retest rules. Do not call
their less-negative R an apples-to-apples improvement over prior SMC backtests.
No profitability percentage, account return, or automatic trading permission
follows from this study. A new hypothesis needs a new preregistration and fresh
confirmation data; do not adjust these rules until this holdout looks good.

### Artifacts and verification

`tests/backtesting/research_entry_models.py` contains causal feature availability,
independent entry models, HTF POI checks and execution replay.
`tests/backtesting/run_entry_research.py` downloads checksum-verified archives,
persists selection, writes every simulated trade and reports clustered bounds.
Outputs are under the Codex workspace's `outputs/hourly-strategy-research-v1`:
`report.md`, `summary.json`, `frozen-selection.json`, `decision.json`, per-symbol
development/later checkpoints, and shared-engine/data-loader source provenance.

New tests verify prefix/full-history agreement, no unfinished 4h leakage,
poisoned-HTF OFF isolation, next-open entry, stop-first ambiguous bars, adverse
gaps, fees, 48-hour expiry and non-overlapping/window-contained outcomes.
Combined focused suite: **78 passed**. No production default, evidence policy,
MTFA isolation rule, renderer, worker or iOS behavior changed in this study.

```sh
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_entry_research --phase dev --cache "$CACHE" --output "$OUTPUT"
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_entry_research --phase later --cache "$CACHE" --output "$OUTPUT"
```

## Standalone evidence extension — 2026-09-07

### Decision and data audit

`tasks.py` previously computed all five standalone engines but omitted their results from `build_trade_plan`. It now passes each result separately. `trade_plan.py` forwards them to `setup_evidence.py`; every selected candidate exposes `indicator_evidence` with a separate nullable `passed`, `available`, reason and snapshot timestamp. These are closed-signal observations, **not observations of the future entry candle**. The existing rejection-then-retest remains pending; no claim that future order flow or divergence has already been checked.

| Group | Exact predeclared check | Important limitation |
|---|---|---|
| `vwap_side` | Last closed price strictly above daily-UTC VWAP for longs, below for shorts | Side-of-VWAP, not a detected reclaim; correlated with price direction |
| `volume_profile_side` | Last closed price strictly above window POC for longs, below for shorts | Engine distributes OHLCV volume over each candle's range; not actual trade-by-price data |
| `tsmom_alignment` | Available combined return-horizon signal has the trade's sign | Default 5m needs at least 6,049 bars; 15m at least 2,017. Never shrink horizons to manufacture availability |
| `cvd_break_confirmation` | Real taker-buy minus taker-sell delta has the trade's sign on the linked break candle | Candle-direction proxy cannot earn this point; ordinary analysis OHLCV currently drops taker-buy data |
| `no_opposing_divergence` | Valid RSI/MACD readings, at least 35 bars, and no opposing event whose second pivot became confirmed within the last 12 bars | Uses the live detector's two-right-bar confirmation. Absence of an event is not positive buying/selling pressure |

Unknown inputs remain `null`, never a free point or an assertion that an indicator disagrees. Equal VWAP/POC or zero momentum/delta does not agree. Separate feature names enable separate tests; they do not establish statistical independence. `CVDEngine` also needed a correctness fix: a missing taker-buy value in a partially populated column must not become zero buying and therefore falsely imply pure selling. Invalid/missing rows now use an explicitly tagged proxy, excluded from decision confirmation.

Binance archives contain genuine taker-buy base volume as field 9 (zero-based); the ordinary six-column history loader and live formatter omit it. [Binance archive specification](https://github.com/binance/binance-public-data). Research with richer archived inputs must be labeled separately from deployment-equivalent inputs. Adding a predictor does not repair a missing ingestion field.

### Opt-in policy, declared before looking at this run's outcomes

`SMC_EVIDENCE_POLICY=smc_v2` remains the production default. It measures the new groups but preserves old ranking, score, threshold and eligibility. `SMC_EXIT_POLICY` remains independent and defaults to `single`.

`SMC_EVIDENCE_POLICY=indicators_v1` is experimental:

- MTFA active: preserve old mandatory linked break, displacement, HTF POI reaction and at least 4/5 SMC groups; additionally require at least **one positive** standalone group. Minimum total 5/10.
- MTFA inactive: preserve linked break, displacement and at least 3/4 local SMC groups; additionally require at least **two positive** standalone groups. Minimum total 5/9. This is deliberately stricter than merely removing HTF confirmation.
- Positive groups are VWAP, profile, TSMOM and genuine CVD. Absence of opposing divergence cannot satisfy that positive-evidence gate. An observed opposing divergence vetoes the experimental setup; unavailable divergence is disclosed and is neither a point nor a veto.
- The maximum includes unavailable standalone checks to expose missing coverage; unknowns do not lower thresholds. The HTF POI group, by contrast, is completely absent when HTF is inactive. An enabled interval with no higher ladder also uses the standalone requirements.
- No new threshold or exit default is promoted on a small or non-independent sample. There are no LLM calls or automatic order execution.

### MTFA isolation, call site through output

`tasks.py` constructs a fresh summary per request and fetches HTF data only inside `if mtfa_enabled`. The planner previously echoed the caller's entire dictionary, allowing stale fields to leak into the result despite not selecting from them. It now replaces disabled context at the boundary with exactly `{"enabled": false, "context": "disabled"}` and deep-copies enabled context to prevent later caller mutation.

All HTF touches reviewed: `_market_context` returns local before trend/alignment access when disabled; pullback/intermediate confirmation is reachable only with active HTF context; `rank_entry_zones` gates POI iteration and HTF evidence insertion with strict boolean activation; candidate enumeration uses only the provided requested-timeframe confluence/OB/FVG entities, never `htf_zones`. No HTF zone is copied into the entry list. Disabled output has `selected_htf_poi: null`, no HTF annotation, no HTF evidence group or score contribution, and no echoed HTF trends/zones. The chart heading additionally ignores HTF fields when disabled. Local engine inputs remain the caller's requested-timeframe entities, as wired by the task.

Tests run the same setup ON then OFF with deliberately retained stale trends and zones, compare OFF to a clean OFF request under both policies, assert no HTF-derived output, and verify no mutation of input or prior output. Other tests cover independent real/proxy CVD, missing coverage, two-positive OFF requirements, causal divergence confirmation and unchanged default execution versus a frozen planner.

### Reproducibility and study limits

Frozen baseline: repository commit `054067c552cf80e196a769c0a708355ee6eaf87d`, copied into `tests/backtesting/evidence_v2_trade_plan.py` and `evidence_v2_setup.py` (only the planner's evidence import redirects to the frozen helper). SHA-256 respectively `2db437a0b9b5b5dce27660f7cdbaac4a9d034194b5fbc529c1e6ab30b18c861e` and `21e74b8f6fc9755f41f3188233c41d3246bef9892c25460e8e5027d94e998b49`. The older pre-SMC baseline remains untouched.

`tests.backtesting.run_indicator_evidence` compares frozen v2 and indicator policy with MTFA ON/OFF, identical single-target management and existing conservative simulator/cost assumptions. Each feature is measured on the actual selected zone of frozen-baseline fills, not a different candidate or only the successful experimental subset. Reports include separate true/false/unavailable counts, mean realized R, win rate and point-biserial correlation. Correlation is undefined without feature variation or sufficient observations, not zero. No overlapping positions/pending orders within a strategy/market. Development and test are independent runs with fresh order state, and the final 108 possible future bars are buffered at each window end: development outcomes cannot consume test-period prices. Stride 3 samples analysis opportunities rather than every candle; results are specific to that sampling schedule. Per-symbol/window checkpoints allow reporting to be rebuilt with `--summarize-only` without recomputing detections.

Development: June 15–July 15, 2025; temporal test: July 15–August 15, 2025; BTC/ETH/SOL, 5m/15m. These windows reuse previously inspected history and **are not an untouched confirmation set**. Rules were fixed before examining this extension's results; no threshold sweep. The 1,000-bar deployment-equivalent research snapshot leaves CVD and low-TF TSMOM unavailable. `enrich_indicator_cohort` separately replays and verifies each frozen fill before measuring genuine archived CVD and longer-history TSMOM. Those shadow covariates do not change the deployable-input policy backtest and cannot justify enabling unavailable inputs in production.

### Completed indicator study: do not promote the new gate

The final run used independent window state and boundary buffers. BTC/ETH/SOL, 5m/15m; development June 15–July 15, temporal test July 15–August 15, 2025; every third analysis opportunity, 1,000-bar snapshots, 10 bps fees plus 2 bps adverse slippage per side. Figures are net of those modeled costs.

| Window | MTFA | Policy | Fills | Net win rate | Mean realized R |
|---|---|---|---:|---:|---:|
| Development | OFF | Frozen v2 | 19 | 10.53% | -2.0872 |
| Development | OFF | Indicators v1 | 6 | 0% | -2.1997 |
| Development | ON | Frozen v2 | 1 | 0% | -2.9359 |
| Development | ON | Indicators v1 | 1 | 0% | -2.9359 |
| Test | OFF | Frozen v2 | 16 | 31.25% | -1.2119 |
| Test | OFF | Indicators v1 | 4 | 0% | -3.1808 |
| Test | ON | Frozen v2 | 1 | 100% | +2.3201 |
| Test | ON | Indicators v1 | 1 | 100% | +2.3201 |

**Decision: reject promotion of the new mandatory indicator gate.** The OFF result is worse in both windows; its test sample retained four losses and did not retain the baseline's winners. ON selected the same single fill in each window and supplies essentially no statistical evidence. Neither policy has demonstrated profitability or professional-grade reliability. Keep `smc_v2` as default; retain `indicators_v1` only as an explicitly requested research mode. Do not silently invert failed filters after seeing test results.

#### Each feature versus actual frozen-baseline outcomes

OFF is the only cohort large enough even for descriptive comparisons (19 development, 16 test fills). Correlations are point-biserial correlations of each boolean with net R, not causal effects, calibrated probabilities, or significance claims. Pooling symbols/timeframes and examining multiple features adds confounding and selection risk.

| Feature | Dev correlation | Test correlation | Test TRUE: N / mean R | Test FALSE: N / mean R | Interpretation |
|---|---:|---:|---|---|---|
| VWAP side | +0.1010 | -0.4118 | 10 / -2.0777 | 6 / +0.2310 | Weak positive development association did not persist; no demonstrated positive predictive value |
| Volume profile side | -0.3209 | -0.1200 | 10 / -1.4642 | 6 / -0.7914 | Passing was associated with worse returns in both windows; do not make it mandatory by default |
| No opposing divergence | -0.0961 | -0.1992 | 10 / -1.6308 | 6 / -0.5138 | This particular absence/veto rule did not improve selection |
| Genuine CVD on break, richer-input shadow | -0.0321 | Undefined | 16 / -1.2119 | 0 / undefined | Every test break passed, so this check had no discriminating variation; no added predictive value demonstrated |
| TSMOM alignment, longer-history shadow | -0.2239 | -0.3683 | 5 / -2.6945 | 11 / -0.5380 | Alignment did not identify better outcomes in this sample; not promoted |

In the deployment-equivalent study CVD and TSMOM were unavailable for **every** baseline fill, not false. The richer-input shadow study reconstructed all 37 frozen fills (35 OFF, 2 ON) and verified their realized R before attaching authentic archive delta and longer-history momentum. It did not change entries or exits, nor rerun the experimental policy with richer data. Actual TSMOM history lengths and available horizons are recorded per fill; they are longer than current deployment-equivalent snapshots, so those results do not establish live availability. ON feature correlations are undefined in each window because N=1.

Full artifacts are in `outputs/indicator-evidence-v1` under the Codex workspace: `summary.csv`, `trades.csv`, `per-feature.csv`, `richer-input-shadow-trades.csv`, `richer-input-shadow-features.csv`, source/checksum manifests and six per-window checkpoints. The first attempt had a report-comprehension error and produced no usable saved results; the final checkpointed rerun is the source of every number in this section. A reporting-recovery regression test now covers that path.

Focused validation covers data recovery, old and new trade-plan gates, MTFA isolation, chart presentation, historical snapshot handling, frozen hashes, real/proxy CVD and report reconstruction. An adequate untouched forward/holdout study is still required before any promotion. No worker was restarted and no environment default was changed.

Reproduce from the backend root (replace the two absolute directory arguments for another machine):

```sh
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_indicator_evidence --cache /Users/apple/Documents/Codex/2026-09-05/meticulously-check-the-added-analysis-feature-2/outputs/market-history --output /Users/apple/Documents/Codex/2026-09-05/meticulously-check-the-added-analysis-feature-2/outputs/indicator-evidence-v1 --start 2025-06-15 --split 2025-07-15 --end 2025-08-15 --stride 3
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.enrich_indicator_cohort --cache /Users/apple/Documents/Codex/2026-09-05/meticulously-check-the-added-analysis-feature-2/outputs/market-history --output /Users/apple/Documents/Codex/2026-09-05/meticulously-check-the-added-analysis-feature-2/outputs/indicator-evidence-v1
```

## Verdict before validation

### Next-move execution and presentation (2026-09-07)

**Superseded after user testing:** the new execution policy below introduced
additional entry restrictions and caused excessive WAIT results. The requested
change was to shorten the existing chart, not replace the trade selection rules.
The live analysis task now explicitly calls the original `retest` planner again;
`SMC_EXECUTION_POLICY=next_move` no longer changes that task. The experimental
module remains available only for explicit code-level research calls.

### Misleading approach projection corrected (2026-09-08)

`first-leg-v1` was defective: it chose UP/DOWN from `trigger > current_price`
and replaced WAIT with NEXT PROJECTED MOVE. A pending short entry above price,
or a bullish recovery condition in a local downtrend, consequently looked like
a supported buy toward that level. Geometry-only tests initially encoded this
incorrect behavior; passing them did not validate the forecast's reasoning.

The replacement `checkpoint-v2` retains the shortened chart but draws only a
neutral horizontal watch level. It does not draw a current-price-to-trigger
trajectory, assign that approach a direction, or label the checkpoint as a
profit target. WAIT stays WAIT; original long/short actions are explicitly
labelled conditional setups with confirmation pending (the retest planner does
not verify entry execution). The original reason and confirmation requirement
remain visible, along with local structural trend and separate HTF context.
There is still no drawn retest or subsequent target path. Original evidence
thresholds, entry/stop/target calculations, and MTFA rules are unchanged.

Regression coverage includes both bullish and bearish real planner pullback
results, WAIT checkpoints, and pending short-above-price/long-below-price
entries. No such result may produce an approach arrow or a NEXT PROJECTED MOVE
headline. Inputs must remain unchanged. Existing saved PNGs are immutable;
request a fresh analysis after the worker loads this renderer.

Prior experiment: the analysis task defaulted to `SMC_EXECUTION_POLICY=next_move`. This is a new,
explicitly experimental execution policy, implemented in `next_move_plan.py` and
dispatched by `build_trade_plan`. `SMC_EXECUTION_POLICY=retest` retains the prior
planner for comparisons and rollback. The frozen old/evidence-v2 backtests still
refer to their original entry rules; their performance numbers do **not** validate
this execution change. No old backtest was relabelled as a next-move result.

The product now evaluates a trade from the latest closed price to the nearest
relevant exit. It no longer draws a future journey to an entry level followed by
a separate trade. Each MTFA mode computes its own result from its snapshot.
Levels are never averaged or copied between ON/OFF requests.

The reasoning sequence is explicit:

1. Preserve local structure. A recent observed BOS/CHoCH supplies the candidate
   direction; HTF bias alone cannot generate a reversal or an approach arrow.
2. Find the nearest untouched liquidity/pivot or fresh opposing OB/FVG near edge.
   Active, available HTF obstacles can shorten this exit. Never skip a nearby
   obstacle to obtain better reward/risk. These are inferred chart levels, not
   a claim that actual orders exist there or that price must reach/stop there.
3. Apply the existing `smc_v2`/`indicators_v1` evidence definitions unchanged.
   Require the selected zone's evidence to belong to the current structure event.
   MTFA context and POI requirements still apply only when enabled.
4. Require an observed entry event on the latest **closed** candle: either the
   displacement break itself, or a directional touch/rejection of its broken
   level after the break. Closes must have held on the correct side since the
   break. An older confirmation, a possible future bounce, or proximity to an
   attractive exit is insufficient. This is an explicit entry rule, not a newly
   optimized indicator checklist.
5. Evaluate current entry location within the existing discount/premium range.
   Stop goes beyond both the originating zone and signal candle, plus the
   existing ATR/price buffer. The stop is never tightened to force approval.
6. Require reward/risk of at least 1.5 **after modeled costs**. Favorable net
   reward is `direction × (target-entry) - (entry+target) × friction`; adverse
   risk is `abs(entry-stop) + (entry+stop) × friction`. Defaults remain estimates:
   `SMC_FEE_BPS_PER_SIDE=10`, `SMC_SLIPPAGE_BPS_PER_SIDE=2`. This is a conservative
   economic eligibility requirement, not a threshold proven profitable by the
   previous study. The rejected minimum-stop experiment remains separately
   opt-in under `SMC_COST_POLICY`; its default is still `none`.
7. If qualified, show BUY/SELL, last-close entry reference, structural stop, and
   one 100% exit. End the illustrative line there. If blocked, show WAIT, the
   watch level if available, and failed facts; draw no directional approach.

`next-move-v1` charts retain the complete policy evidence ledger, including
mandatory failures. Local BOS/CHoCH remains visible when HTF context blocks a
trade. Chart horizontal spacing is schematic; it does not forecast arrival time.
The entry reference is a closed snapshot, not a live executable quote, and the
app does not place orders. Price/cost changes require reassessment before entry.

This workflow follows the use of support/resistance as potential entry and exit
levels described by [IG](https://www.ig.com/en/trading-strategies/support-and-resistance-levels-explained-181219),
and structural invalidation/risk budgeting discussed by
[CME](https://www.cmegroup.com/education/courses/trade-and-risk-management/proper-position-size).
Those sources support the reasoning framework, not these particular thresholds
or SMC heuristics. There is no meaningful basis for a “90–99% professional”
rating. Remaining limits include uncalibrated evidence, venue costs, live quotes,
portfolio sizing, execution latency, and unvalidated outcomes for the new policy.
Mixed HTF context still returns WAIT; this does not represent every trader's
approach to countertrend trades. The existing indicator policy and MTFA isolation
implementations were preserved.

Focused tests cover BUY/SELL symmetry, latest-close entry, single exit, nearest
obstacle selection, old/future triggers, net cost rejection, disabled stale HTF
data, ON/OFF local evidence, and absence of a WAIT projection. A bounded smoke
check evaluates and repeats 200 BTC/ETH historical 5m snapshots for deterministic,
JSON-safe output. It is **not an outcome backtest**. Synthetic BUY/WAIT image
fixtures and a historical chart are saved alongside `next-move-smoke.json` in
the workspace outputs for visual review.
All 200 smoke snapshots returned WAIT (zero executed setups); therefore this
check provides no realized-return evidence. Synthetic fixtures verify that both
BUY and SELL can qualify, but do not establish real-market signal frequency.

The original pipeline is deterministic, not random-number-based. Its apparently arbitrary output comes from weak selection and missing evidence propagation. The detectors encode price-action heuristics; they do not establish institutional intent or predictive profitability. No result in this document should be called professional-grade or a calibrated forecast without independent validation.

## Research: what top-down analysis actually contributes

- Start with wider context and **price location**, then seek lower-timeframe behavioral change. IG's example explicitly does not assume support will hold: a lower-timeframe break and retest provide the trigger. Its warning against requiring every timeframe to agree also matters: universal alignment is not the definition of top-down analysis. [IG, Introduction to multi-time frame analysis](https://www.ig.com/sg/trading-strategies/introduction-to-multi-time-frame-analysis-220929).
- Annotate each timeframe's levels and conditions, then combine them into a trade thesis. A small-chart bearish move inside a larger bullish context may be a correction; the context alone does not establish that it will reverse. [OANDA, multi-timeframe entries and exits](https://www.oanda.com/us-en/skills-and-insights/education/technical-analysis/price-charts-and-candlesticks/analysis-multi-timeframe-better-entries-exits/).
- A stop must express invalidation, not an arbitrary cash amount; position size follows stop distance and the trader's risk budget. This backend currently has no portfolio risk budget, so it should not prescribe position size. [CME, Proper Position Size](https://www.cmegroup.com/education/courses/trade-and-risk-management/proper-position-size).
- Scaling out is a management choice, not proof of increased expectancy. Partial profit reduces remaining exposure but also reduces participation in large winners. A stop moved to entry can remove a runner during an ordinary retracement. [IG discussion of taking profits and scaling](https://www.ig.com/en/news-and-trade-ideas/trading-mistakes--failing-to-take-profits-on-a-trend-230905).
- More filters, variants and parameter searches increase selection bias. Record failed versions and examine separate history instead of presenting the best-looking trial as evidence. [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

These sources motivate a workflow, not the particular SMC thresholds below. OB/FVG geometry and inferred liquidity are not direct observations of bank orders. Five checklist points are not five independent statistical predictors or a win probability.

## File-by-file audit

| File | Original behavior | Extension / remaining limitation |
|---|---|---|
| `analyze_with_mtfa.py` | Concurrent injected fetchers, requested-snapshot reuse, isolated HTF failures are good. Returned complete HTF objects, but decision summary reduced them to direction. | Keep asynchronous orchestration and incomplete-context WAIT. HTF orchestration now supplies POIs through `analyze_smc_structure`. |
| `analyze_smc_structure.py` | Only swings, structure and liquidity. | Compute actual HTF OB/FVG entities off the event loop; serialize their availability time, price bounds and mitigation state. |
| `tasks.py` | Discarded HTF spatial evidence; local sweep detector unused by planner. Infrastructure initialization could fail before terminal-failure handling. | Propagate `htf_zones` and sweeps, preserve requested-TF chart, publish failure even if database failure persistence fails. Treat unavailable data as an expected task exception. |
| `trade_plan.py` | Not literally first zone: confluence-source priority, then distance. Nonetheless no linked trigger or quality grade. Confirmation only prose; fixed nearest target and gross 1.5R threshold. Checked CURRENT price location although order was proposed at a retracement. | Score linked evidence, gate mandatory facts, inspect entry location, retain explicit pending confirmation, preserve single-target compatibility and expose staged levels. Profitability remains unvalidated. |
| `setup_evidence.py` (new) | No equivalent. | One point each for linked recent BOS/CHoCH, displacement, preceding sweep, OB/FVG overlap, HTF POI reaction (MTFA only). No repeated points for duplicate zones or multiple aligned HTFs. |
| `swing_structure_engine.py` / `market_structure_engine.py` | Confirmed fractal pivots and close-through structure breaks; these are meaningful causal primitives. | Keep unchanged. Choice of fractal window remains heuristic. CHoCH is a first opposing break, not proof of sustained reversal. |
| `liquidity_engine.py` / `liquidity_sweep_engine.py` | Inferred swing clusters, not exchange order-book liquidity. Sweeps could be timestamped before a pivot's right-hand confirmation bars existed. | Carry `confirmed_index` and do not recognize a sweep before that time. Snapshot clustering is still not a complete chronological order-flow model. |
| `fvg_engine.py` | Three-candle nonoverlap; called displacement, but no directional body/ATR strength test. | Preserve geometry; decision layer separately tests breakout displacement. |
| `order_block_engine.py` | Last opposite candle before an observed BOS/CHoCH, binary mitigation on first touch. | Preserve detector convention. HTF POIs can be touched-but-not-close-invalidated; do not claim these are fresh untouched OBs. Availability is break CLOSE, not original OB candle time. |
| `imbalance_order_block_engine.py` | Pure overlap can join zones from unrelated legs. | Give overlap one point only and separately demand a recent linked structure event. Overlap alone cannot pass. |
| `premium_discount_engine.py` | Latest confirmed high/low range, which can be degenerate or not the trader's intended impulse. | Preserve conservative invalid-range WAIT; evaluate the pending entry inside the selected half of the range. This range heuristic still needs independent testing. |
| `analysis_chart_presentation.py` | Reduced clutter, but no observed evidence explaining the projected path. | Plot up to five truthful time/price anchors, disclose any omitted anchor count, and list every policy-relevant fact in the decision ledger; distinguish observed facts from an illustrative pending path. No heatmap. |
| `data_access.py` / `market_data.py` / Binance `client.py` | Empty cache already fell back to Binance. Broad wrapping hid cause; swallowed provider errors looked like empty data; pooled client could belong to an expired Celery event loop; zero-volume candles rejected; recent-priority refresh returned old snapshot; persistence failure could discard usable source candles. | Per-call analysis exchange session on current loop, explicit source errors, accept zero-volume candles, merge refreshed data, reject stale analysis, persistence queue failure does not destroy fetched candles. Exact original upstream failure cannot be inferred from the supplied traceback alone. |

Additional data-integrity findings:

- InfluxDB `market_db.py` could downsample reverse history with `aggregateWindow(fn: first)` on each field. Analysis now explicitly disables this display optimization: it can change timeframe and does not preserve OHLC semantics. Analysis skips repeated health-probe handshakes and uses a bounded query timeout; existing display callers retain their behavior.
- Redis `rate_limiter.py` previously used the global async connection and failed open on errors. Analysis now owns a current-loop Redis connection and fails closed. Other callers retain their existing behavior. Kline request weight is corrected to the documented 2. [Binance endpoint reference](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market).
- BNBUSDT 1m cold-source recovery was tested twice with real exchange requests: each returned 720 candles over 12h (719 closed at request time). Persistence dispatch was stubbed so these verification calls did not write InfluxDB. This confirms present recovery, not the unknowable exact original upstream exception.

Dead legacy `analysis_structure/main_analysis_structure.py`, `enhanced_pattern_api.py`, `trader_aware_*`, `analysis_pipeline.py`, `trend_detector.py`, `zone_detector.py`, `pattern_scanner.py`, `candle_confirmer.py` and `scorer.py` are not used for this implementation or baseline.

## Predeclared deterministic policy

`evidence-v1`: structure event no older than 12 requested-timeframe bars and tied to candidate origin/formation (within 3 bars); directional breakout body at least 0.8 previous-bar ATR and at least 60% of the bar's range; optional sweep within 10 bars before break, after pool confirmation. MTFA-on additionally requires an overlapping, available, aligned HTF POI touched before the break and not subsequently close-invalidated. Threshold 4/5 on, 3/4 off. Structure and displacement are mandatory; HTF POI mandatory on. Scores are uncalibrated evidence counts.

`evidence-v2`: correct location assessment from the current breakout close to the proposed retracement entry. This correction was made after the first development run and BEFORE the separate June 15–August 15 evaluation. No score threshold was relaxed to obtain trades. The original large-timeframe mixed/incomplete-context WAIT policy was preserved, not claimed to represent every professional's style.

Entry remains conditional: a closed directional zone rejection arms a subsequent retest, expires after 12 bars, and cancels at invalidation. Live snapshots do not place or manage orders. Staged research variant uses 50% T1, 50% next untouched liquidity; only one available target means 100% there. Price-breakeven starts next bar after T1 in the OHLC simulation; transaction costs mean it is not economic breakeven.

## Research log and reproducibility

- Baseline: byte-for-byte frozen original `trade_plan.py` in `tests/backtesting/legacy_trade_plan.py`; SHA-256 recorded in each run's `config.json`.
- Development run: May 15–June 15, 2025; BTCUSDT, ETHUSDT, SOLUSDT; 5m and 15m; every third decision close; 1,000-bar rolling windows. Old: 14 fills, 21.43% net wins, average -1.1165R; new v1: one pending plan, zero fills. This is not validation of the filter. Quality scores were concentrated at 0–2, with only one 4-point candidate.
- Separate evaluation: June 15–August 15, 2025; same markets, every decision close; selection rules frozen before inspecting this period. Completed results below. Additional exit variants isolate single T1, staged+BE, and staged without BE. The default exit choice was made AFTER seeing results, and is not another independently validated strategy.
- Archive source: Binance public spot monthly ZIPs, verified using the published SHA-256 checksums; timestamps normalize milliseconds/microseconds. [Binance archive format and checksums](https://github.com/binance/binance-public-data).
- Costs: 10 bps fee + 2 bps adverse slippage each side, including partial/time exits. These are assumptions, not the user's actual fee tier. [Binance fee calculation](https://www.binance.com/en-IN/support/faq/detail/e85d6e703b874674840122196b89780a).
- No same-close fills; a rejection and limit fill cannot occur on the same bar. Stop wins if stop and target share a bar; stop gaps fill adversely at the open. Do not credit favorable entry-bar excursions. Max holding 96 bars with mark-to-close time exit. Trades and pending orders do not overlap within each variant/market.
- HTFs are built only from complete source bars and are available only after their close. Every detection uses a historical prefix. Resampling drops incomplete higher candles rather than filling gaps with synthetic prices. The same input windows are used for old/new.
- Limitations: a handful of crypto spot markets is not all markets/regimes. Paper shorts are directional tests, not executable unlevered spot positions; borrowing/funding is not modeled. Legacy confirmation was prose, so its precise trading policy never existed: the common rejection-then-retest model is an explicit operationalization. Fees, sizes, order queues, intrabar paths and latency need venue-specific forward testing. Raw hit rate is not evidence of profitability. No parameter sweep was performed.

The harness reports net-positive trade rate, average net realized R, gross R and net-negative trade rate. The latter is an operational losing-trade rate, NOT a calibrated detector false-positive probability. Empty score bins and zero-trade strategies must be reported as undefined, not 0% accuracy or proof of safety.

## Completed separate-history results

June 15–August 15, 2025, BTCUSDT/ETHUSDT/SOLUSDT, 5m and 15m. 70,272 scheduled analysis closes; 10 bps fee + 2 bps adverse slippage per side. No strategy has demonstrated positive net expectancy here.

| Variant | Filled trades | Net win rate | Mean realized net R |
|---|---:|---:|---:|
| Frozen old planner, single target | 30 | 20% | -1.4516 |
| Old entries, staged + price BE | 30 | 20% | -1.3207 |
| Old entries, staged, original stop | 30 | 20% | -1.3373 |
| New evidence filter, single target | 2 | 50% | -0.3084 |
| New evidence filter, staged + price BE | 2 | 0% | -1.9325 |
| New evidence filter, staged, original stop | 2 | 0% | -2.1822 |

Matched-entry counterfactuals reproduce the above exit means on these samples: improvement on old entries is slight and remains negative; staging materially worsens the two new trades. **Do not promote partials/breakeven as the default.** Implementation defaults to single T1. `SMC_EXIT_POLICY=staged` or `staged_no_be` explicitly enables the research alternatives. There is no automatic order execution. More history and a separate confirmation sample are needed before promoting any exit policy.

New filter: 95% Wilson interval for net win rate is approximately **9.5%–90.5%**. With two trades, neither 50% win rate nor the difference from the old average establishes an edge. The single-target mean's day-resampled interval spans roughly -2.94R to +2.32R; such a tiny bootstrap is descriptive, not reliable inference. The old mean's day-resampled interval is roughly -2.55R to -0.41R under the simulator assumptions.

### Does confluence predict outcome?

Score the **actual selected price zone**, not whichever alternative the new planner prefers. The initial harness's old-trade score column described the latter; use `verified-trades.csv` / `verified-summary.json`, which rebuild every recorded plan and correct this attribution. The decision logic and realized returns were not changed by this reporting correction.

| Score of old selected zone | Trades | Mean net R |
|---|---:|---:|
| 0 | 5 | -1.5565 |
| 1 | 17 | -1.9696 |
| 2 | 5 | -1.4110 |
| 3 | 2 | -3.1708 |
| 4 | 1 | +11.1145 |

Spearman rank correlation of score with net R is **-0.098** (30 old trades): no demonstrated monotonic relationship. One high-score winner does not validate the scoring system. Both new trades scored 4, so correlation within new trades is **undefined**. Consequently the score remains an evidence-completeness checklist with explicit mandatory gates, NOT a claimed probability or empirically validated quality grade. Do not increase position size based on it.

### Costs dominate tight-stop setups

| Assumed fee + slippage per side | Old mean R | New single-target mean R |
|---|---:|---:|
| 0 + 0 bps | +0.1566 | +2.7521 |
| 4 + 2 bps | -0.6475 | +1.2219 |
| 10 + 2 bps | -1.4516 | -0.3084 |
| 10 + 5 bps | -1.8537 | -1.0735 |

The production planner's gross R gate is NOT an account-specific net R calculation. These results show why gross 1.5R can be inadequate with small stop distances. Do not describe the displayed gross R as expected net return; actual venue fee tier, spread, slippage and shorting costs are necessary for a deployable trade policy.

#### Stop distance investigation — 2026-09-07

For every unique single-target selection in the existing June 15–August 15 dataset, stop distance is `abs(entry - initial_stop) / entry × 10,000`. The complete trade-level table is saved as `existing-stop-distance-trades.csv`; no stopped, target or time exit was omitted.

| Planner | Fills | Stop range | Median stop | Pearson stop-bps/net-R | Spearman stop-bps/net-R |
|---|---:|---:|---:|---:|---:|
| Frozen old | 30 | 5.47–59.27 bps | 19.69 bps | +0.329 | +0.798 |
| Evidence v2 | 2 | 5.75–12.40 bps | 9.07 bps | Undefined | Undefined |

Old-planner stop-distance quartiles were: 5.47–11.41 bps, N=8, mean **-3.4367R**; 11.63–19.19 bps, N=7, **-0.6557R**; 20.19–25.81 bps, N=7, **-1.5014R**; and 28.25–59.27 bps, N=8, **-0.1193R**. This strongly supports the tight-stop/cost hypothesis for the old planner in this small sample, though the non-monotonic middle quartiles and observational grouping prevent a causal claim. Evidence v2 cannot confirm it: its 5.75-bps trade won +2.3201R net while its 12.40-bps trade lost -2.9368R net. N=2 is not a relationship test.

At 12 bps per side, approximate round-trip friction is about 24 price bps. Expressed in R it is roughly `24 / stop_bps`, before target-price differences: about 4R for a 6-bps stop, 2R for 12 bps, and 1R for 24 bps. This explains mechanically how a gross-positive strategy can become net-negative without proving which stop floor will preserve useful trades.

No further confirmation-indicator gate is being tested. The cost study's development window is September 1, 2024–March 1, 2025; temporal test is September 1, 2025–March 1, 2026. The six-month gap keeps the test outside every June–August 2025 result already inspected in this project. Symbols are BTC, ETH, BNB, XRP, ADA, DOGE, LINK and LTC USDT; 5m/15m; stride 2; complete 108-bar outcome buffers; independent order state per window and variant; checksum-verified Binance archives. Predeclared candidate floors are 0/12/18/24/30/40/50 bps. One shared floor is chosen using **only old-planner development results** if it has at least 30 fills, retains at least 25% of baseline fills, and improves mean net R by at least 0.25; maximize mean R, then sample size. Otherwise freeze 0 bps (no promoted filter). The selected floor is then applied unchanged to old and evidence-v2 planners in the test. Using the larger old cohort to select and the same threshold on the sparse new planner tests transfer rather than tuning the new planner's few trades. It still cannot make a sparse new sample adequate.

The development rule selected and froze **30 bps**. This was a development selection, not permission to deploy it. The untouched temporal holdout then produced:

| Planner / policy | Dev fills | Dev mean gross R | Dev mean net R | Test fills | Test win rate | Test mean gross R | Test mean net R |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old, no floor | 193 | +0.1534 | -0.8888 | 220 | 25.00% | +0.1895 | -1.0622 |
| Old, frozen 30-bps floor | 99 | +0.7211 | +0.2414 | 92 | 30.43% | +0.1201 | -0.3822 |
| Evidence v2, no floor | 22 | +0.2555 | -0.8840 | 16 | 12.50% | -0.6771 | -1.6815 |
| Evidence v2, same frozen floor | 9 | +0.3581 | -0.0489 | 9 | 22.22% | -0.4260 | -0.8503 |

**Holdout decision: do not promote the 30-bps policy to the production default.** It reduced the old planner's test loss by 0.6800R and the evidence-v2 loss by 0.8312R, so tight-stop cost amplification is real and persistent. It did not produce positive test expectancy. Evidence v2 has only 16 baseline and nine filtered test fills, which is still plainly inadequate. The old planner's 220 test fills are a useful descriptive expansion, not enough to rescue an economically losing result.

Baseline test stop distance remains positively related to net R: old N=220, median 24.00 bps, Pearson +0.209 and Spearman +0.601; evidence v2 N=16, median 34.93 bps, Pearson +0.842 and Spearman +0.938. Old test stop quartiles had mean gross/net R of **+0.096/-2.456**, **+0.445/-0.826**, **+0.182/-0.601**, and **+0.035/-0.366** from tightest to widest. Thus the tightest old quartile was slightly profitable before costs and catastrophically negative after them, while wider groups still failed to clear realistic friction. Evidence-v2 quartiles were gross-negative in all four buckets, so its holdout failure is not merely a stop-distance problem.

The operational floor replay owns independent pending/trade state: rejecting one tight plan lets a later timestamp become actionable. It is therefore a realistic policy comparison, not a row filter. As a separate matched-baseline diagnostic, retaining only original test fills with stops at least 30 bps gives old N=85 at -0.3113R net and evidence v2 N=8 at -0.7345R. The conclusion is unchanged. About 277,000 scheduled decision closes were examined, but inference is governed by filled-trade counts, not candle count.

The deployable hook is independent of evidence scoring: default `SMC_COST_POLICY=none`; research mode `SMC_COST_POLICY=minimum_stop_bps` reads `SMC_MIN_STOP_BPS`. The latter remains disabled after the failed holdout. Enabled output records the measured stop bps and threshold. Failure returns WAIT before exposing a pending entry scenario; passing setups retain the same stop, entry, target and evidence rules. This filter rejects uneconomic geometry—it does not widen a technically defined invalidation stop.

#### Chart reasoning ledger

`analysis_chart_presentation.py` now separates chart-located observations from the full eligibility ledger. `setup_evidence.py` emits exactly one ledger item for every group active under the chosen evidence policy: structure break, displacement, sweep, OB/FVG overlap, optional HTF POI, and—in `indicators_v1` only—the five standalone groups plus the required standalone-count gate. The scoring and eligibility definitions are unchanged. Each item carries PASSED/FAILED/UNAVAILABLE and whether a failure is mandatory. Missing HTF POI and divergence vetoes therefore remain visible on WAIT charts. The cost policy appends its own mandatory PASS/FAIL item when enabled.

Only observations with truthful price/time coordinates receive arrows. Momentum, divergence availability and count gates are summary facts, not invented points on a candle. Up to five located anchors are drawn to protect candle and forecast readability; if more exist, the chart states how many additional anchors were omitted while listing **all** material facts in the caption. The pending dotted path remains explicitly illustrative and visually separate from observed facts. Presentation version is `evidence-v5`.

### Coverage and limits that still matter

- This is a reproducible **research harness**, not a validated strategy. It is not proof of live fill quality or profitability. New sample size is plainly inadequate.
- Historical tests use 1,000-bar rolling windows on every timeframe. Production HTF lookbacks vary (e.g. 1h 14d, 4h 60d, daily 200d), so this run is a controlled old/new comparison, not exact replication of every possible frontend lookback. Differences must be tested before extrapolating these metrics to live requests.
- Data integrity repairs (raw candles, closed-bar availability, source recovery) are correctness changes retained regardless of profitability. Conservative structure/POI gates are retained as requirements for an evidence-supported diagram, not asserted profitable filters.
- The historical chart example is BTCUSDT 5m at 2025-06-10 12:15 UTC, selected from development history to demonstrate available evidence. It is not evidence of a winning trade. The original staged screenshot is superseded by the default single-target preview.

### Reproduce

Run from the backend, setting OUTPUT and CACHE to your chosen local directories:

```sh
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_trade_plans --start 2025-06-15 --end 2025-08-15 --stride 1 --output "$OUTPUT" --cache "$CACHE"
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.report_trade_plans "$OUTPUT" "$CACHE"
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.stop_cost_report "$OUTPUT/trades.csv" "$STOP_OUTPUT"
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_stop_cost_study --phase dev --start 2024-09-01 --end 2025-03-01 --stride 2 --cache "$CACHE" --output "$STOP_OUTPUT"
PYTHONPATH=src:. .venv/bin/python -m tests.backtesting.run_stop_cost_study --phase test --start 2025-09-01 --end 2026-03-01 --stride 2 --cache "$CACHE" --output "$STOP_OUTPUT"
```

The stop study writes `existing-stop-distance-trades.csv`, `dev-trades.csv`, `test-trades.csv`, per-policy summaries, stop quartiles and relationships, per-symbol counts, manifests, configuration/source hashes, and `frozen-stop-policy.json`. Archive manifests, checksums, configuration, raw trades, matched-exit counterfactuals, fee sensitivity and verified score statistics are written alongside the earlier results. Re-running uses the cached, checksum-verified files. Tests exercise lookahead boundaries, zero-volume candles, source failure, downsample bypass, mandatory evidence gates, exact levels, ambiguous-bar stops and staged execution.
