# Top-down analysis audit and validation — 2026-09-06

## Verdict before validation

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
| `analysis_chart_presentation.py` | Reduced clutter, but no observed evidence explaining the projected path. | At most three selected evidence anchors (HTF reaction, sweep, BOS/CHoCH); two target labels when supported; distinguish observed facts from an illustrative pending path. No heatmap. |
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
```

Archive manifests, checksums, configuration, raw trades, matched-exit counterfactuals, fee sensitivity and verified score statistics are written alongside the results. Re-running uses the cached, checksum-verified files. Tests exercise lookahead boundaries, zero-volume candles, source failure, downsample bypass, mandatory evidence gates, exact levels, ambiguous-bar stops and staged execution.
