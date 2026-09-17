# Geometry qualification and bounded scan workload

Verified locally on 17 September 2026. No production services were deployed,
no live scanner was enabled, and the ten-symbol pilot manifest was not expanded.

## Detector findings and changes

The existing notification workflow fixtures mock detector output. They do not
establish that the detector actually recognizes the supplied candles. A new
offline suite runs actual registered detectors through the scanner's validation,
normalization and recency contract.

The suite reproduced **26 failures out of 100 checks** before these fixes:

| Finding | Change |
|---|---|
| Rectangle touch quality was lowest at the boundary it was supposed to reward | Score by absolute distance from the band; exclude points outside tolerance; reject zero-height ranges |
| Triangle classification changed when the same prices were rescaled | Compare fractional slopes per bar, preserving the prior threshold at a reference price of 100 |
| Expanding boundaries were accepted as a symmetrical triangle | Require falling highs/rising lows and an eligible future intersection; enforce the existing documented apex horizon |
| Exactly horizontal regression targets divided by zero in fit scoring | Use normalized residuals and an explicit numerical-zero variance case for triangle/channel fits |
| Roundoff in near-zero slopes rejected flat channels; low-priced rising/falling channels became horizontal | Classify flatness relative to price and bypass directional slope-ratio comparison only for flat levels |
| Fixed quote-unit range thresholds suppressed small-priced doji and hammer fixtures | Reject numerical zero relative to price; retain the existing proportional geometry/context filters |

The first **100 checks passed**: 25 deterministic cases at four price multipliers
(`1e-7`, `0.01`, `1`, `1000`). Every run must produce the expected variant or no
match, ready coverage, valid anchors/scores, and no numerical runtime warnings.
These corrections affect the shared detector functions, including legacy chart
callers. Strict exception propagation remains opt-in for scanner callers.

That first corpus provided synthetic positive examples for 15 enabled variants:
rectangle; ascending/descending/symmetrical triangle; horizontal/ascending/
descending channel; bullish/bearish engulfing; standard/dragonfly/gravestone doji;
hammer; morning star; evening star. Negative examples cover zero range, trends
instead of consolidation, expanding/parallel triangle boundaries, converging
channels, absent engulfment, large doji bodies, wrong hammer context and failed
star reversals.

## Completed positive fixture coverage

The expanded corpus now includes **all 31 enabled variants**: both ABCD, Bat and
Gartley directions; both head-and-shoulders directions; double/triple tops and
bottoms; bullish/bearish flags; rising/falling wedges, in addition to the first
15 variants above. These are **69 scenarios at four scales, 276 checks**. Every
window also runs all 20 pilot detectors (5,520 evaluations) so an unrelated
detector failure makes the case fail. Recognition expectations apply only to the
named detector; other detectors' matches are not independently labeled.

The first expanded run reproduced **36 failures among 252 checks** before fixes.
Another 24 flag-window checks were added afterward. The final 276 checks pass.

| Reproduced finding | Correction |
|---|---|
| Inverse head-and-shoulders rejected with three troughs and only two peaks | Accept the appropriate pivot counts for either direction |
| Bat final retracement used X–D instead of A–D relative to X–A | Measure `AD_XA`, matching a retracement from A toward X |
| ABCD direction came from A and was reversed at D | Use the final pivot for ABCD direction, keeping five-point behavior unchanged |
| A recent flag was invisible unless its pole occupied the first third of the entire window | Locate recent poles and shorter consolidations using bounded local searches |

The Bat endpoint correction follows the retracement definition in the author's
[Bat description](https://harmonictrader.com/harmonic-patterns/bat-pattern/).
The [AB=CD description](https://harmonictrader.com/harmonic-patterns/abcd-pattern/)
supports the equal-leg construction used in the fixtures. These references do
not establish live accuracy, and the existing harmonic detectors still implement
only a subset of published criteria. For example, Gartley's C ratio remains
restricted to 0.382 ± 0.05, and Bat does not yet enforce every BC/AB=CD
projection condition. Other unqualified harmonic configurations are unchanged.

Sixteen focused harmonic checks also verify ratio scores and legacy output:
the shared ratio scorer now decreases monotonically with absolute ratio error
and rejects nonfinite inputs/invalid tolerances. Previously its mixed absolute
and relative tolerances could reward a worse fit. ABCD maturity now agrees with
its direction; legacy intermediate targets are ordered between D and the existing
B endpoint. Scanner responses continue to omit all targets and stops.

The new flag policy searches 12–60 consolidation bars and at most 120 preceding
pole bars, requires a pole move of at least 3%, directional travel efficiency of
at least 80%, a consolidation no longer than the pole, and at most a 50% close
retracement. It requires two touches on each boundary, fit quality of at least
0.8, parallel or horizontal bounds, and the last close still inside them. Slopes
are normalized to price and horizontal fits do not divide by zero. These are
explicit pilot heuristics, not universal flag definitions or measured probability
thresholds. A breakout-following confirmation detector is not implemented here.

Negative fixtures include wrong harmonic ratios, unequal reversal peaks,
missing heads, parallel wedges, absent flagpoles, wrong flag slope, full pole
retracement and overly long consolidation. Multiple pole locations prevent
reintroducing the old fixed-third split. An early falling-wedge fixture ended
four bars ago and was correctly excluded by recency; the positive fixture was
corrected to a recent mirrored wedge instead of relaxing the scanner contract.

The labels are synthetic engineering expectations with written rationales, not
independently reviewed real-market annotations. Market precision and recall are
explicitly `null` in the report. No win-rate or trading-profit claim follows.

```sh
PYTHONPATH=src:. .venv/bin/python scripts/qualify_scanner_patterns.py
```

Fixtures: `tests/fixtures/scanner/geometry.json` and
`tests/fixtures/scanner_geometry.py`. Per-case results and the detector code
version are written to `logs/scanner-qualification.json`; the pre-fix diagnostic
run is retained locally as `logs/scanner-qualification-before.json`, with the
expanded 252-check baseline in `logs/scanner-qualification-expanded-before.json`.
The report distinguishes fixture inventory from variants whose positive cases
actually pass every scale. The command
exits nonzero if any geometry case fails. Pytest runs the same cases in
`tests/unit/test_scanner_geometry.py`.

## Larger real-service workload

The disposable Redis/Influx/Celery harness now accepts an optional burst:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --burst --report logs/scanner-burst-report.json
```

It seeds **50 symbols across 15m, 1h, 4h and 1d**, using varied synthetic geometry,
price scales and volume. Forty symbols are deliberately synthetic identities;
the other ten use the pilot names with synthetic data. All four latest-due
cutoffs are enqueued together, producing **200 instrument jobs and 4,000 detector
evaluations**, inside the existing 200-stream guard.

Observed on this local machine with two separate workers, each using two prefork
processes, Redis 7.4.10 and InfluxDB 2.7.12:

- All 200 instruments ready; no pending, partial or error coverage.
- 200 computed, zero cache reuse for the burst inputs.
- **9.556 seconds** from dispatch through all four complete snapshots, excluding
  fixture seeding and service/worker startup.
- Sampled Redis queue peaks: 96 `scanner`, 2 `scanner_ingestion`, sampled every
  50 ms. These exclude reserved/in-flight/ETA tasks and can miss shorter peaks.
- Zero external Python socket attempts; no Binance/Massive requests.
- Six runtime cases passed, including the earlier shared-read/reuse/Lua checks.

This rerun completed at 18:30:44 UTC on 17 September 2026 with the expanded
geometry corpus and corrected detectors. The prior 6.987-second run used the
smaller corpus and earlier code; these are different workloads, not a controlled
performance comparison. The 126-second overall integration duration includes a
wait for the next safe candle boundary; the dispatch measurement excludes it.

This is one bounded synthetic workload measurement. It is not a replay of a real
exchange stream, a representative latency distribution, a test of thousands of
symbols, or a 1,000/2,000-user capacity result. The test services and worker
processes were removed afterward. Production coverage and limits remain unchanged.

Qualification-milestone regression: **696 unit tests passed**, with five existing dependency warnings
and the previously documented broken legacy price-alert test excluded. The six
integration cases have three dependency warnings. Remaining gates are recorded,
independently labeled examples, broader harmonic-definition checks,
worker-death/network-failure tests, larger representative workloads and the real
HTTP concurrency/soak test described in the development pipeline.

The subsequent [events/recovery milestone](scanner-events.md) passes 713 unit
and eight integration cases, including two actual worker-child kills. Its latest
report is `logs/scanner-recovery-report.json`; the timings above describe the
earlier qualification run.
