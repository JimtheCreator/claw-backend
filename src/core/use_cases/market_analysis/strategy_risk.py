"""Account-independent risk contract for research strategies; no position sizing."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class StrategyRiskPolicy:
    minimum_stop_bps: float = 30.0
    minimum_rr: float = 1.5
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    atr_buffer: float = 0.25
    maximum_hold_bars: int = 48

    def __post_init__(self):
        for value in (self.minimum_stop_bps, self.minimum_rr, self.fee_bps,
                      self.slippage_bps, self.atr_buffer):
            if not isfinite(value) or value < 0:
                raise ValueError('Risk parameters must be finite and non-negative')
        if self.maximum_hold_bars < 1:
            raise ValueError('Hold period must be positive')


def evaluate_risk(entry, stop, target, sign, policy, *, target_source):
    entry, stop, target = [float(v) if v is not None and isfinite(v) else None for v in (entry, stop, target)]
    result = dict(eligible=False, reason=None, entry=entry, stop=stop, target=target,
                  target_source=target_source, unit='R', position_size=None,
                  maximum_hold_bars=policy.maximum_hold_bars,
                  minimum_stop_bps=policy.minimum_stop_bps)
    if sign not in (-1, 1) or any(v is None or not isfinite(v) or v <= 0 for v in (entry, stop, target)):
        result['reason'] = 'Missing or invalid entry, structural stop or target.'
        return result
    risk, reward = sign*(entry-stop), sign*(target-entry)
    if risk <= 0 or reward <= 0:
        result['reason'] = 'Stop or target lies on the wrong side of entry.'
        return result
    rr, bps = reward/risk, risk/entry*10000
    cost = (policy.fee_bps+policy.slippage_bps)/10000
    result.update(risk_per_unit=risk, gross_target_r=rr, stop_distance_bps=bps,
                  target_cost_r=(entry+target)*cost/risk,
                  stop_cost_r=(entry+stop)*cost/risk,
                  net_target_r=rr-(entry+target)*cost/risk,
                  note='R uses initial price risk; cost estimate is not expected profit.')
    if bps + 1e-9 < policy.minimum_stop_bps:
        result['reason'] = f'Stop {bps:.2f}bps below {policy.minimum_stop_bps:g}bps floor.'
    elif rr + 1e-9 < policy.minimum_rr:
        result['reason'] = f'Remaining gross reward {rr:.2f}R below {policy.minimum_rr:g}R.'
    else:
        result.update(eligible=True, reason='Structural risk and declared cost floor passed.')
    return result


def build_strategy_risk(row, sign, family):
    policy = StrategyRiskPolicy(minimum_stop_bps=60.0 if family == 'local_fallback_v1' else 30.0)
    entry, atr = row.get('close'), row.get('atr')
    pivot = row.get('stop_support' if sign == 1 else 'stop_resistance')
    stop = pivot-sign*policy.atr_buffer*atr if pivot is not None and atr is not None else None
    target = row.get('resistance' if sign == 1 else 'support')
    source = 'confirmed_opposing_pivot'
    if family == 'momentum_flow_v1':
        target = entry+sign*2*(sign*(entry-stop)) if stop is not None else None
        source = 'declared_2R_not_observed_liquidity'
    return evaluate_risk(entry, stop, target, sign, policy, target_source=source)
