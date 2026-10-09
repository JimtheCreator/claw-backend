"""Session boundaries in exchange-local time; never synthesize missing candles.

Spot FX convention is New York 17:00 Sunday–Friday, including DST. Explicit
provider/instrument closures override it. Metals and other non-FX instruments
must have their own calendar; their hours are not inferred from a C: prefix.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketSession:
    kind: str
    closures: tuple[tuple[datetime, datetime], ...] = ()

    def __post_init__(self):
        if self.kind not in {"crypto", "forex"}:
            raise ValueError("An explicit session calendar is required")
        for start, end in self.closures:
            if start.tzinfo is None or end.tzinfo is None or start >= end:
                raise ValueError("Invalid session closure")

    def is_open(self, when):
        if when.tzinfo is None:
            raise ValueError("Session timestamps must be timezone-aware")
        if any(start <= when < end for start, end in self.closures):
            return False
        if self.kind == "crypto":
            return True
        local = when.astimezone(NY)
        day = local.weekday()
        return day < 4 or (day == 4 and local.hour < 17) or (day == 6 and local.hour >= 17)

    def state(self, when, last_close=None):
        opened = self.is_open(when)
        next_open = None
        if not opened:
            candidate = when.replace(second=0, microsecond=0) + timedelta(minutes=1)
            # A bad holiday configuration cannot create an unbounded search.
            for _ in range(15 * 24 * 60):
                if self.is_open(candidate):
                    next_open = candidate.astimezone(UTC).isoformat()
                    break
                candidate += timedelta(minutes=1)
        return {"market_state": "open" if opened else "closed",
                "next_open": next_open,
                "last_close_timestamp": last_close,
                "session_calendar": "continuous" if self.kind == "crypto" else "fx-new-york-17"}

    def active_bar(self, opened, step):
        start = datetime.fromtimestamp(opened, UTC)
        end = start + timedelta(seconds=step)
        # Scanner intervals never exceed one day. Include partially active
        # Sunday/daily candles without manufacturing weekend-only candles.
        return (self.is_open(start) or self.is_open(end - timedelta(microseconds=1))
                or any(start < stop < end and self.is_open(stop) for _, stop in self.closures))

    def expected_opens(self, cutoff, step, count):
        values = []
        stamp = cutoff - step
        for _ in range(count * 4 + 15 * 86400 // step):
            if self.active_bar(stamp, step):
                values.append(stamp)
                if len(values) == count:
                    return list(reversed(values))
            stamp -= step
        raise ValueError("Session calendar has no sufficient active history")


def closed_session_snapshot(metadata, now):
    """Keep the last completed FX scan readable through the weekly closure.

    An already-stale pre-close snapshot is never relabeled as fresh. This is
    availability of last-session results, not a claim of current live quotes.
    """
    if metadata.get('market') != 'forex':
        return metadata
    from .catalog import INTERVAL_SECONDS
    session = MarketSession('forex')
    state = session.state(now)
    if state['market_state'] != 'closed' or not state['next_open']:
        return metadata
    step = INTERVAL_SECONDS[metadata['interval']]
    cutoff = int(now.timestamp()) // step * step
    last_close = session.expected_opens(cutoff, step, 1)[0] + step
    if datetime.fromisoformat(metadata['data_as_of']).timestamp() < last_close:
        return metadata
    fresh = datetime.fromisoformat(state['next_open']) + timedelta(seconds=step+5)
    return dict(metadata, fresh_until=fresh.isoformat(), session=state)
