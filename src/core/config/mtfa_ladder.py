from typing import Dict, List

# Nearest higher timeframe first. Empty list = top of the ladder, nothing
# higher exists to consult - MTFA degrades to standalone here regardless
# of the toggle, because there's structurally nothing to pull from.
MTFA_LADDER: Dict[str, List[str]] = {
    "1m": ["15m", "1h", "4h"],
    "5m": ["1h", "4h", "1d"],
    "15m": ["1h", "4h", "1d"],
    "30m": ["4h", "1d", "1w"],
    "1h": ["4h", "1d", "1w"],
    "2h": ["1d", "1w"],
    "4h": ["1d", "1w"],
    "6h": ["1d", "1w"],
    "1d": ["1w", "1M"],
    "3d": ["1w", "1M"],
    "1w": ["1M"],
    "1M": [],
}


def get_htf_chain(interval: str) -> List[str]:
    """
    Returns the ordered list of higher timeframes MTFA pulls bias/context
    from for the given interval - nearest first.

    This is the ONLY direction MTFA ever looks: up. Looking down (finer
    timeframes for entry refinement) is a deliberately separate concern
    that never touches this ladder - conflating the two breaks what the
    MTFA toggle means (see analyze_with_mtfa's docstring).

    An interval this config doesn't recognize returns an empty chain,
    same as the top of the ladder - never a guessed default. Silently
    inventing a ladder for an unknown interval would be worse than just
    treating it as standalone-only and letting the caller notice the gap.
    """
    return list(MTFA_LADDER.get(interval, []))