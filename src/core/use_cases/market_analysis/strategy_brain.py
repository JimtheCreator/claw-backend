"""Independent strategy pool and frozen arbitration. No legacy entry gates.

This is an experimental candidate contract, not an order or a live trade signal.
Production integration stores it in shadow; it cannot overwrite the trade plan.
"""
from math import isfinite

from core.engines.regime_engine import RegimeEngine
from core.use_cases.market_analysis.strategy_risk import build_strategy_risk

FAMILIES = ('smc_location_v1', 'momentum_flow_v1', 'local_fallback_v1')


def _finite(value):
    return value is not None and isfinite(value)


def _candidate(family, sign, checks, row, *, context=None):
    risk = build_strategy_risk(row, sign, family) if sign else None
    failures = [k for k, v in checks.items() if v is not True]
    if risk is not None and not risk['eligible']:
        failures.append('risk: '+risk['reason'])
    return dict(strategy=family, status='eligible' if sign and not failures else 'rejected',
                direction='long' if sign == 1 else 'short' if sign == -1 else None,
                checks=checks, failed_checks=failures, risk=risk,
                context=context, entry_timing='next_open_after_observed_trigger',
                validation_status='experimental_not_validated')


def _disabled(family, reason):
    # No skipped HTF point counted false or true; no copied HTF data.
    return dict(strategy=family, status='unavailable', reason=reason)


def evaluate_strategies(row, *, mtfa_enabled, htf_available=False, htf_reactions=None):
    """No mutation; the OFF request boundary ignores all caller HTF objects."""
    result = []
    sign = int(row.get('break_sign') or 0)
    if not mtfa_enabled:
        result.append(_disabled(FAMILIES[0], 'Strategy requires enabled multi-timeframe data.'))
    elif not htf_available:
        result.append(_disabled(FAMILIES[0], 'Anchor or middle closed history is unavailable.'))
    else:
        reactions = htf_reactions or []
        poi = next((p for p in reactions if p['sign'] == sign), None)
        checks = dict(anchor_poi=poi is not None, middle_reaction=poi is not None,
                      local_displacement_break=bool(sign and row.get('displacement')))
        result.append(_candidate(FAMILIES[0], sign, checks, row, context=poi))

    complete = row.get('momentum_complete') is True
    if not complete:
        result.append(_disabled(FAMILIES[1], 'Complete configured momentum horizons unavailable.'))
    else:
        momentum = int(row.get('momentum_sign') or 0)
        trigger = int(row.get('channel_trigger') or 0)
        vw, delta, delta3 = row.get('vwap'), row.get('delta'), row.get('delta3')
        checks = dict(tsmom_horizons_agree=bool(momentum),
                      own_channel_trigger=bool(momentum and trigger == momentum),
                      vwap_confirmation=bool(_finite(vw) and momentum*(row['close']-vw) > 0 and
                                             row.get('session_bars', 0) >= 3 and row.get('session_complete')),
                      genuine_cvd_confirmation=bool(row.get('real_flow') and _finite(delta) and _finite(delta3)
                                                    and momentum*delta > 0 and momentum*delta3 > 0))
        result.append(_candidate(FAMILIES[1], momentum, checks, row))

    if mtfa_enabled or complete:
        result.append(_disabled(FAMILIES[2], 'Fallback needs MTFA OFF and unavailable complete momentum.'))
    else:
        checks = dict(observed_structure_break=bool(sign), displacement=bool(row.get('displacement')),
                      preceding_pivot_sweep=bool(row.get('preceding_sweep')),
                      local_ob_fvg_overlap=bool(row.get('ob_fvg_overlap')))
        result.append(_candidate(FAMILIES[2], sign, checks, row))
    return result


def arbitrate(candidates, regime):
    eligible = [c for c in candidates if c['status'] == 'eligible']
    directions = {c['direction'] for c in eligible}
    if len(directions) > 1:
        return dict(action='wait', selected_strategy=None, status='conflict',
                    reason='Eligible strategies disagree; neither direction overrides the other.',
                    conflicting_strategies=sorted(c['strategy'] for c in eligible))
    if not eligible:
        return dict(action='wait', selected_strategy=None, status='no_eligible_strategy',
                    reason='No independent strategy passed its observed trigger and risk checks.')
    priority = (FAMILIES[1], FAMILIES[0], FAMILIES[2]) if regime['trend'] == 'trending' else FAMILIES
    winner = min(eligible, key=lambda c: priority.index(c['strategy']))
    return dict(action=winner['direction'], selected_strategy=winner['strategy'], status='research_candidate',
                reason='Eligible strategies agree; fixed regime preference selects the research candidate.',
                risk=dict(winner['risk']))


def evaluate_brain(row, *, mtfa_enabled, htf_available=False, htf_reactions=None):
    candidates = evaluate_strategies(row, mtfa_enabled=mtfa_enabled,
                                    htf_available=htf_available, htf_reactions=htf_reactions)
    regime = RegimeEngine.describe(row)
    return dict(version='independent-brain-v1', mode='shadow', production_promoted=False,
                regime=regime, candidates=candidates, arbitration=arbitrate(candidates, regime),
                momentum_history=dict(complete=row.get('momentum_complete') is True),
                warning='Research candidate, not an execution instruction or estimated win probability.')
