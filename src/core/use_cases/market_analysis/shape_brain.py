"""Independent master-vision hypotheses. No route from this pool to live orders."""
from math import isfinite

from core.config.execution_ladder import get_execution_chain
from core.use_cases.market_analysis.strategy_brain import _candidate, _disabled
from core.use_cases.market_analysis.strategy_risk import StrategyRiskPolicy, evaluate_risk

SHAPES = ('smc_location_v1', 'tsmom_v1', 'vwap_reversion_v1')
VERSION = 'master-vision-shapes-v2'


def finite(value):
    try:
        return value is not None and isfinite(float(value))
    except (ValueError, TypeError):
        return False


def flow_confirmation(row, sign):
    delta = row.get('delta3')
    if not sign or not row.get('real_flow') or not finite(delta):
        return dict(status='unavailable', score_adjustment=0, mandatory=False)
    aligned = sign*delta > 0
    return dict(status='aligned' if aligned else 'opposed' if sign*delta < 0 else 'neutral',
                score_adjustment=1 if aligned else -1 if sign*delta < 0 else 0,
                mandatory=False, delta3=float(delta), source='genuine_taker_volume')


def candidate(family, sign, checks, row, stop, target, source):
    risk = evaluate_risk(row.get('close'), stop, target, sign, StrategyRiskPolicy(), target_source=source) if sign else None
    fired = bool(sign and all(v is True for v in checks.values()))
    failures = [k for k, v in checks.items() if v is not True]
    if fired and not risk['eligible']:
        failures.append('risk: '+risk['reason'])
    return dict(strategy=family, status='eligible' if fired and risk['eligible'] else 'rejected',
                signal_fired=fired, direction='long' if sign == 1 else 'short' if sign == -1 else None,
                checks=checks, failed_checks=failures, risk=risk,
                validation_status='experimental_not_validated', entry_timing='next_execution_open')


def surface_agreement(candidates):
    fired = sorted((c for c in candidates if c['status'] == 'eligible'), key=lambda c: c['strategy'])
    directions = {c['direction'] for c in fired}
    base = dict(action='wait', selected_strategy=None, strategies=[c['strategy'] for c in fired])
    if not fired:
        return dict(base, status='no_setup', reason='No independent shape fits with valid risk right now.')
    if len(directions) > 1:
        return dict(base, status='conflict', reason='Independent shapes disagree; neither overrides the other.')
    if len(fired) > 1:
        return dict(base, status='agreement_unranked', direction=fired[0]['direction'],
                    reason='Shapes agree, but their plans stay separate; no validated ranking exists.')
    return {**base, 'action': fired[0]['direction'], 'selected_strategy': fired[0]['strategy'],
            'status': 'research_candidate', 'risk': dict(fired[0]['risk'])}


def evaluate_shapes(row, *, roles, htf_available=False, reactions=None, qa_htf_off=False):
    """All decisions concern CLOSED execution candles; roles are explicit."""
    if roles.get('execution') not in get_execution_chain(roles.get('intermediate')):
        raise ValueError('A declared genuinely finer execution timeframe is required.')
    sign = int(row.get('break_sign') or 0)
    poi = None if qa_htf_off or not htf_available else next((p for p in (reactions or []) if p['sign'] == sign), None)
    if qa_htf_off or not htf_available:
        smc = _disabled(SHAPES[0], 'Macro/intermediate context unavailable.' if not qa_htf_off else 'HTF disabled by internal QA.')
    else:
        # Unchanged SMC predicate/risk; execution now has its own finer candles.
        smc = _candidate(SHAPES[0], sign, dict(anchor_poi=poi is not None, middle_reaction=poi is not None,
                         local_displacement_break=bool(sign and row.get('displacement'))), row, context=poi)
        smc['signal_fired'] = bool(sign and all(smc['checks'].values()))
    momentum = int(row.get('momentum_sign') or 0)
    if row.get('momentum_complete') is not True:
        tsmom = _disabled(SHAPES[1], 'Full execution-timeframe TSMOM horizons unavailable.')
    else:
        atr, close = row.get('atr'), row.get('close')
        stop = close-momentum*2*atr if finite(atr) and finite(close) else None
        target = close+momentum*4*atr if finite(atr) and finite(close) else None
        tsmom = candidate(SHAPES[1], momentum, dict(own_return_horizons_agree=bool(momentum)),
                          row, stop, target, 'declared_2R_not_liquidity')
        tsmom['horizons'] = {k: float(v) for k, v in row.items() if k.startswith('own_return_') and finite(v)}
    sigma, vwap, z = row.get('vwap_sigma'), row.get('vwap'), row.get('vwap_z')
    ready = (bool(row.get('session_complete')) and row.get('session_bars', 0) >= 3 and
             all(finite(x) for x in (sigma, vwap, z)) and sigma > 0)
    if not ready:
        reversion = _disabled(SHAPES[2], 'Complete execution UTC session / VWAP deviation band unavailable.')
    else:
        side = -1 if z >= 2 else 1 if z <= -2 else 0
        atr = row.get('atr')
        far = min(row['close'], vwap-3*sigma) if side == 1 else max(row['close'], vwap+3*sigma)
        stop = far-side*.25*atr if finite(atr) else None
        reversion = candidate(SHAPES[2], side, dict(outside_two_sigma_band=bool(side)), row,
                              stop, vwap, 'execution_session_vwap_frozen_at_signal')
        reversion.update(vwap=float(vwap), sigma=float(sigma), z_score=float(z), exit_at_session_end=True)
    candidates = [smc, tsmom, reversion]
    for c in candidates:
        if c['status'] == 'unavailable':
            continue
        direction = 1 if c['direction'] == 'long' else -1 if c['direction'] == 'short' else 0
        c['optional_cvd'] = flow_confirmation(row, direction)
        c['execution_time'] = row['available_at'].isoformat()
        c['entry_timing'] = 'next_execution_open'
        c['execution_interval'] = roles['execution']
        check_roles = {'anchor_poi': 'macro', 'middle_reaction': 'intermediate'}
        c['chart_evidence'] = [dict(group=k, passed=v, mandatory=True,
                                   timeframe=roles[check_roles.get(k, 'execution')]) for k, v in c['checks'].items()]
        c['chart_evidence'].append(dict(group='risk_contract', passed=bool(c.get('risk') and c['risk']['eligible']),
                                       mandatory=True, timeframe=roles['execution'],
                                       reason=c['risk']['reason'] if c.get('risk') else 'No directional signal.'))
        if c['strategy'] == SHAPES[1]:
            c['chart_evidence'].append(dict(group='own_return_horizons', timeframe=roles['execution'],
                                           returns=dict(c['horizons']), observed_at=c['execution_time']))
        if c['strategy'] == SHAPES[2]:
            c['chart_evidence'].append(dict(group='session_vwap_band', timeframe=roles['execution'],
                                           vwap=c['vwap'], sigma=c['sigma'], z_score=c['z_score'],
                                           lower_two_sigma=c['vwap']-2*c['sigma'], upper_two_sigma=c['vwap']+2*c['sigma'],
                                           observed_at=c['execution_time']))
        c['chart_evidence'].append(dict(group='optional_cvd', timeframe=roles['execution'], **c['optional_cvd']))
        if c['strategy'] == SHAPES[0] and poi:
            c['chart_evidence'] += [dict(group='macro_poi', timeframe=roles['macro'], price_zone=[poi['bottom'], poi['top']],
                                       available_at=poi['available_at']),
                                    dict(group='intermediate_reaction', timeframe=roles['intermediate'], observed_at=poi['reaction_at'])]
    return dict(version=VERSION, mode='shadow', production_promoted=False,
                roles={**roles, 'macro': None if qa_htf_off else roles.get('macro')},
                candidates=candidates, arbitration=surface_agreement(candidates),
                warning='Independent research hypotheses; no candidate has passed untouched validation.')
