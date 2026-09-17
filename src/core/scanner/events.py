"""Observed pattern lifecycle; no forecast or per-user computation."""
import hashlib
import json
from datetime import datetime

from core.scanner.catalog import INTERVAL_SECONDS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lifecycle_transition(previous, metadata, results):
    """Compare complete consecutive closes; incomplete coverage cannot end a match.

    First observation, changed definitions, or a missed close establish a fresh
    baseline, avoiding a burst of false 'new' matches after startup or downtime.
    Same-close corrections update the baseline without emitting new alerts.
    """
    coverage = metadata["coverage"]
    if coverage["eligible"] == 0 or coverage["ready"] != coverage["eligible"]:
        return None
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
            current[key] = dict(row)
    state = {"schema_version": 1, "scope": scope, "epoch": epoch, "cutoff": cutoff, "matches": current}
    if previous and previous["scope"] != scope:
        raise ValueError("Pattern lifecycle checkpoint scope mismatch")
    if previous and cutoff < previous["cutoff"]:
        return None
    same_close = previous is not None and cutoff == previous["cutoff"]
    gap = (previous is not None and not same_close and
           (datetime.fromisoformat(cutoff) - datetime.fromisoformat(previous["cutoff"])).total_seconds()
           != INTERVAL_SECONDS[metadata["interval"]])
    reason = ("initial" if previous is None else "definition_changed" if previous["epoch"] != epoch
              else "observation_gap" if gap else None)
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
        prior = previous["matches"]
        for key in sorted(prior.keys() | current.keys()):
            old, new = prior.get(key), current.get(key)
            # End anchors, score and last price may evolve without a new setup.
            changed = old and new and old["pattern_start"] != new["pattern_start"]
            if old and (not new or changed):
                events.append(event("no_longer_detected", old,
                                    reason="replaced" if new else "absent_on_complete_scan"))
            if new and (not old or changed):
                events.append(event("detected", new))
    batch = {"schema_version": 1, "batch_id": batch_id, "scope": scope,
             "epoch": epoch, "data_as_of": cutoff, "events": events}
    return state, batch
