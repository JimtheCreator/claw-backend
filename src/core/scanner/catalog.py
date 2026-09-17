"""Lightweight catalog: importing an API must not import detector CPU libraries."""
import json
from functools import lru_cache
from pathlib import Path

INTERVAL_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


@lru_cache(maxsize=1)
def detector_catalog():
    return json.loads(Path(__file__).with_name("catalog.json").read_text())


def pattern_catalog():
    return sorted(
        [dict(pattern, detector_id=detector["id"], category=detector["category"])
         for detector in detector_catalog() for pattern in detector["patterns"]],
        key=lambda pattern: pattern["display_name"],
    )
