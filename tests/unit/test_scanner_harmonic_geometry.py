"""Numerical invariants behind the full-candle harmonic fixtures."""
import asyncio
import json

import numpy as np
import pytest

from core.use_cases.market_analysis.detect_patterns_engine.harmonic_patterns import (
    calculate_ratio_confidence, detect_abcd, detect_bat,
)
from scripts.qualify_scanner_patterns import CORPUS
from tests.fixtures.scanner_geometry import geometry_rows


@pytest.mark.parametrize("ideal", [.382, .618, .786, .886, 1.0, 1.618])
def test_ratio_score_decreases_with_absolute_error_on_both_sides(ideal):
    for sign in [-1, 1]:
        scores = [calculate_ratio_confidence(ideal + sign * error, ideal)
                  for error in np.linspace(0, .055, 12)]
        assert scores[0] == 1
        assert scores[-1] == 0
        assert all(0 <= score <= 1 for score in scores)
        assert all(a >= b for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("actual,ideal,tolerance", [
    (float('nan'), .618, .05), (float('inf'), .618, .05),
    (.618, float('nan'), .05), (.618, .618, float('inf')),
    (.618, .618, 0), (.618, .618, -.05),
])
def test_ratio_score_rejects_nonfinite_or_invalid_tolerance(actual, ideal, tolerance):
    assert calculate_ratio_confidence(actual, ideal, tolerance) == 0


def _case_data(case_id):
    case = next(case for case in json.loads(CORPUS.read_text())["cases"] if case["id"] == case_id)
    rows = geometry_rows(case["recipe"])
    return {key: [row[key] for row in rows] for key in rows[0]}


@pytest.mark.parametrize("direction,sign", [("bullish", 1), ("bearish", -1)])
def test_abcd_direction_maturity_and_legacy_levels_agree(direction, sign):
    patterns = asyncio.run(detect_abcd(_case_data(f"abcd_{direction}")))
    match = next(p for p in patterns if p["pattern_name"] == f"abcd_{direction}")
    levels = match["key_levels"]
    assert match["end_index"] == 247
    assert levels["is_mature"] and levels["maturity_score"] == 1
    d = levels["points"]["D"]["price"]
    targets = levels["targets"]
    ordered = [targets["stop_loss"], d, targets["target_1"], targets["target_2"], targets["target_3"]]
    assert all(sign * (b - a) > 0 for a, b in zip(ordered, ordered[1:]))


@pytest.mark.parametrize("direction", ["bullish", "bearish"])
def test_bat_measures_completion_retracement_from_a(direction):
    patterns = asyncio.run(detect_bat(_case_data(f"bat_{direction}")))
    match = next(p for p in patterns if p["pattern_name"] == f"bat_{direction}")
    assert match["key_levels"]["ratios"]["AD_XA"] == pytest.approx(.886)
    assert "XD_XA" not in match["key_levels"]["ratios"]
