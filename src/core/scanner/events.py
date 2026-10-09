"""Observed pattern lifecycle; no forecast or per-user computation."""
import hashlib
import json
from datetime import datetime

from core.scanner.catalog import INTERVAL_SECONDS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lifecycle_transition(previous, metadata, results):
    """Compare consecutive complete observations for each instrument.

    First observation, changed definitions, or a missed close establish a fresh
    baseline, avoiding a burst of false 'new' matches after startup or downtime.
    Same-close corrections update the baseline without emitting new alerts.
    """
    coverage = metadata["coverage"]
    instruments = metadata.get('instrument_coverage')
    if coverage["eligible"] == 0 or coverage['ready'] == 0:
        return None
    if instruments is None and coverage['ready'] != coverage['eligible']:
        return None  # Older snapshots lack the evidence needed for partial delivery.
    ready = None
    if instruments is not None:
        if (len(instruments) != coverage['eligible'] or
                sum(status == 'ready' for status in instruments.values()) != coverage['ready']):
            raise ValueError('Instrument coverage does not match lifecycle totals')
        ready = {f"{metadata['provider']}:{metadata['market']}:{symbol}"
                 for symbol, status in instruments.items() if status == 'ready'}
    scope = [metadata["provider"], metadata["market"], metadata["universe_id"], metadata["interval"]]
    epoch = [metadata["universe_revision"], metadata["detector_version"]]
    cutoff = metadata["data_as_of"]
    current = {}
    for pattern, rows in results.items():
        for row in rows:
            if row["pattern_id"] != pattern or row["interval"] != metadata["interval"]:
                raise ValueError("Pattern lifecycle scope mismatch")
            key = json.dumps([row["instrument_id"], row["pattern_id"]], separators=(",", ":"))
            if key in current:
                raise ValueError("Duplicate instrument/pattern in lifecycle input")
            if ready is not None and row['instrument_id'] not in ready:
                continue  # Partial detector output cannot establish or end a match.
            # Drawing geometry is presentation data, not notification state.
            current[key] = {k: v for k, v in row.items() if k not in ("geometry", "preview")}
    if previous and previous["scope"] != scope:
        raise ValueError("Pattern lifecycle checkpoint scope mismatch")
    if previous and cutoff < previous["cutoff"]:
        return None
    same_close = previous is not None and cutoff == previous["cutoff"]
    step = INTERVAL_SECONDS[metadata['interval']]
    expected_previous = datetime.fromisoformat(cutoff).timestamp() - step
    gap = (previous is not None and not same_close and
           datetime.fromisoformat(previous['cutoff']).timestamp() != expected_previous)
    reason = ("initial" if previous is None else "definition_changed" if previous["epoch"] != epoch
              else "observation_gap" if gap else None)
    prior = previous['matches'] if previous and not reason else {}
    stamps = dict(previous.get('instrument_cutoffs', {})) if previous and not reason else {}
    if ready is not None:
        members = {f"{scope[0]}:{scope[1]}:{symbol}" for symbol in instruments}
        stamps = {instrument: stamp for instrument, stamp in stamps.items() if instrument in members}
        # Retain unavailable instruments without claiming their old matches ended.
        current.update({key: row for key, row in prior.items()
                        if row['instrument_id'] in members - ready})
    observed = ready if ready is not None else {row['instrument_id'] for row in current.values()}
    prior_stamps = dict(stamps)
    stamps.update({instrument: cutoff for instrument in observed})
    state = {"schema_version": 1, "scope": scope, "epoch": epoch, "cutoff": cutoff,
             "matches": current, "instrument_cutoffs": stamps}
    events = []
    batch_id = _digest([scope, epoch, cutoff])

    def event(kind, match=None, **extra):
        value = {"type": kind, "data_as_of": cutoff, **extra}
        if match is not None:
            value["match"] = match
        value["event_id"] = _digest([batch_id, kind, match and match["instrument_id"],
                                     match and match["pattern_id"], match and match["pattern_start"]])
        return value

    if reason:
        events.append(event("baseline_reset", reason=reason))
    elif not same_close:
        for key in sorted(prior.keys() | current.keys()):
            old, new = prior.get(key), current.get(key)
            if ready is not None:
                instrument = (new or old)['instrument_id']
                last_seen = prior_stamps.get(instrument)
                if (instrument not in ready or last_seen is None or
                        datetime.fromisoformat(last_seen).timestamp() != expected_previous):
                    continue  # First/recovered observation establishes a baseline only.
            # End anchors, score and last price may evolve without a new setup.
            changed = old and new and old["pattern_start"] != new["pattern_start"]
            if old and (not new or changed):
                events.append(event("no_longer_detected", old,
                                    reason="replaced" if new else "absent_on_complete_scan"))
            if new and (not old or changed):
                events.append(event("detected", new, new_symbol=not bool(old)))
    batch = {"schema_version": 1, "batch_id": batch_id, "scope": scope,
             "epoch": epoch, "data_as_of": cutoff, "events": events}
    return state, batch
