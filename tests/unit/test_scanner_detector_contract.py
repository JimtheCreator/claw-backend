"""Failure-contract tests; these are not evidence of market detection accuracy."""
import asyncio

import numpy as np
import pytest

from core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
from core.use_cases.market_analysis.detect_patterns_engine import harmonic_patterns
from core.use_cases.market_analysis.detect_patterns_engine.pattern_registry import (
    strict_detector_errors, strict_errors_enabled,
)


def ohlcv():
    closes = [130.0 - i * 0.12 for i in range(250)]
    return {
        "open": [c + 0.2 for c in closes], "close": closes,
        "high": [c + 0.3 for c in closes], "low": [c - 0.1 for c in closes],
        "volume": [1000.0] * 250,
    }


@pytest.mark.parametrize("name", sorted(initialized_pattern_registry))
def test_internal_failures_propagate_only_through_strict_entrypoint(name, monkeypatch):
    """Inject a math-library failure after input validation, across all detectors."""
    entry = initialized_pattern_registry[name]
    failure = RuntimeError("synthetic detector dependency failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(np, "array", fail)
    monkeypatch.setattr(np, "asarray", fail)

    async def scenario():
        with pytest.raises(RuntimeError) as raised:
            await entry["strict_function"](ohlcv())
        assert raised.value is failure
        assert not strict_errors_enabled()
        if entry["category"] in {"chart", "candlestick"}:
            assert await entry["function"](ohlcv()) is None
        else:
            # Harmonic entrypoints already propagated these outer failures.
            with pytest.raises(RuntimeError) as legacy:
                await entry["function"](ohlcv())
            assert legacy.value is failure

    asyncio.run(scenario())


def test_strict_entrypoint_preserves_valid_match_and_no_match():
    async def scenario():
        entry = initialized_pattern_registry["engulfing"]
        assert await entry["function"](ohlcv()) is None
        assert await entry["strict_function"](ohlcv()) is None
        data = ohlcv()
        for key, values in {
            "open": (101.0, 100.0), "close": (100.3, 102.5),
            "high": (101.2, 102.7), "low": (100.1, 99.8),
        }.items():
            data[key][-2:] = values
        legacy = await entry["function"](data)
        strict = await entry["strict_function"](data)
        assert strict == legacy
        assert strict["pattern_name"] == "bullish_engulfing"

    asyncio.run(scenario())


def test_nested_harmonic_volume_failure_is_not_replaced_by_default_score(monkeypatch):
    swings = [(0, 100.0, "low"), (60, 120.0, "high"),
              (120, 108.0, "low"), (180, 128.0, "high")]
    monkeypatch.setattr(harmonic_patterns, "find_significant_swings", lambda data, **kwargs: swings)
    original_mean = np.mean

    def failing_volume_mean(values, *args, **kwargs):
        if len(values) == 20 and all(v == 1000.0 for v in values):
            raise RuntimeError("synthetic volume calculation failure")
        return original_mean(values, *args, **kwargs)

    monkeypatch.setattr(np, "mean", failing_volume_mean)

    async def scenario():
        entry = initialized_pattern_registry["abcd"]
        legacy = await entry["function"](ohlcv())
        assert legacy[0]["key_levels"]["volume_score"] == 0.5
        with pytest.raises(RuntimeError, match="volume calculation failure"):
            await entry["strict_function"](ohlcv())
        assert not strict_errors_enabled()

    asyncio.run(scenario())


def test_degenerate_harmonic_ratio_is_still_a_legitimate_rejected_candidate():
    swings = [(0, 100.0, "low"), (60, 100.0, "high"),
              (120, 90.0, "low"), (180, 110.0, "high")]
    config = {
        "name": "abcd", "number_of_points": 4,
        "ratios": [{"name": "degenerate", "calculate": lambda p: 1 / (p[1] - p[0]),
                    "ideal": 1.0, "tolerance": 0.05, "weight": 1.0}],
    }
    with strict_detector_errors():
        assert harmonic_patterns.validate_pattern(swings, config, ohlcv()) == []


def test_strict_policy_does_not_leak_to_concurrent_legacy_callers():
    async def scenario():
        entered, legacy_finished = asyncio.Event(), asyncio.Event()

        async def strict_call():
            with strict_detector_errors():
                entered.set()
                await legacy_finished.wait()
                assert strict_errors_enabled()
            assert not strict_errors_enabled()

        async def legacy_call():
            await entered.wait()
            assert not strict_errors_enabled()
            # An input failure retains the existing legacy fallback in this task.
            assert await initialized_pattern_registry["engulfing"]["function"]({}) is None
            legacy_finished.set()

        await asyncio.gather(strict_call(), legacy_call())
        assert not strict_errors_enabled()

    asyncio.run(scenario())


def test_nested_strict_scope_and_cancellation_restore_the_previous_policy():
    async def scenario():
        entered = asyncio.Event()

        async def cancelled_call():
            try:
                with strict_detector_errors():
                    with strict_detector_errors():
                        assert strict_errors_enabled()
                    assert strict_errors_enabled()
                    entered.set()
                    await asyncio.Future()
            finally:
                assert not strict_errors_enabled()

        task = asyncio.create_task(cancelled_call())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not strict_errors_enabled()

    asyncio.run(scenario())
