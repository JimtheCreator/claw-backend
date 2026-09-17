"""Verify scanner coverage includes failures from real registered detectors."""
import asyncio

import numpy as np
import pytest

from core.scanner.engine import scan_instrument
from tests.unit.test_market_scanner import CUTOFF, source


@pytest.mark.parametrize("detector", ["engulfing", "rectangle", "abcd"])
def test_scanner_reports_internal_detector_failure_as_partial_coverage(monkeypatch, detector):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic math dependency failure")
    monkeypatch.setattr(np, "array", fail)
    result = asyncio.run(scan_instrument("BTCUSDT", "15m", CUTOFF, [detector], source()))
    assert result["status"] == "partial"
    assert result["matches"] == []
    assert result["detector_coverage"][detector] == {"evaluated": 0, "errors": 1}
    assert result["issues"] == [{"symbol": "BTCUSDT", "detector_id": detector,
                                 "reason": "detector_error"}]
