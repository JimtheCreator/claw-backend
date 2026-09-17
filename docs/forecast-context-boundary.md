# Forecast context boundary — 12 September 2026

Previously a local structural scenario could survive a WAIT decision caused by
missing/conflicting HTF data or failed setup evidence. The chart could display
that scenario with a Long/Short badge and TP/SL regions.

The retest planner now emits `primary_scenario` and `forecast_scenario` only
after its existing context, evidence, location, target and risk gates pass.
Every WAIT return clears both forecast fields. `market_read` preserves observed
structure; `structure_watch` carries conditional reassessment text where context
permits a watch, without authorizing a forecast.

The renderer independently requires an eligible setup, a supported market
context, matching action/direction and a setup scenario. It rejects old
scenario-only payloads too. Rejected plans have no directional badge or TP/SL
shading and are labeled MARKET WATCH / FORECAST WITHHELD. Qualified retest
plans say Long pending or Short pending on the badge. The presentation version
is `conditional-forecast-v8`.

MTFA OFF still means local analysis. This change does not turn MTFA on secretly,
add regime/volume filters, change entry/stop/target prices, or promote research
strategies. It enforces existing requirements consistently; it does not
establish higher forecast accuracy or profitability.

Validation: 207 unit tests pass with the pre-existing
`test_price_alert_manager.py` collection failure excluded (missing
`infrastructure.notifications`). Added mirrored long/short planner-to-renderer
tests across HTF, displacement, structure, overlap/sweep, location-availability,
target, R:R and cost-floor failures; eligible ON/OFF setups; and stale renderer
payloads. Four synthetic PNGs were generated through the actual ChartEngine
image path and checked for blocked/pending display behavior. No profitability
backtest was rerun because entry decisions and economics were not changed.

Existing stored chart images are immutable. A worker must load the changed
source, and a fresh analysis must be requested to see the new presentation.
The local worker was confirmed idle (no active, reserved or scheduled tasks),
gracefully stopped, and restarted with its original arguments and environment
preserved. The replacement reported ready on 12 September 2026. No remote
deployment or stored-image rewrite was performed.
