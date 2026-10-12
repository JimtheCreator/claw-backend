"""Compact push copy; the complete match list stays in authenticated storage."""


import random

_body_bags = {}


def _next_body(kind, choices):
    bag = _body_bags.setdefault(kind, [])
    if not bag:
        bag.extend(range(len(choices)))
        random.shuffle(bag)
    return choices[bag.pop()]


def pattern_summary(deliveries):
    unique = {}
    for row in deliveries:
        match = row['payload']['match']
        market = 'forex' if match['market'] == 'forex' else 'crypto'
        unique.setdefault((market, match['symbol']), dict(match, universe=row['universe']))
    matches = sorted(unique.values(), key=lambda m: (m['market'], m['symbol']))
    return {'matches': matches, 'count_kind': 'newly_detected',
            'crypto_count': sum(m['market'] != 'forex' for m in matches),
            'forex_count': sum(m['market'] == 'forex' for m in matches)}


def summary_body(summary):
    count = len(summary['matches'])
    if count < 1:
        raise ValueError('Cannot notify an empty match group')
    if count == 1:
        return _next_body("single", [
            "A new symbol matches this pattern.",
            "A new match has been detected.",
            "This pattern has been spotted on a new symbol."])
    # Count refers to distinct newly detected symbols in this notification,
    # not the scanner's total active population. Avoid total-active wording.
    choices = [f"{count} symbols now match this pattern.",
               f"{count} symbols are showing this pattern.",
               "New matches detected across multiple symbols."]
    choices.append("This pattern is showing up across multiple markets.")
    return _next_body("multiple", choices)
