import asyncio

import numpy as np
import pytest

from core.use_cases.market_analysis.detect_patterns_engine.shared_features import (
    MAX_FEATURES, shared_feature, shared_features,
)


def test_features_are_scoped_bounded_and_mutation_safe():
    calls = []
    @shared_feature
    def feature(values):
        calls.append(1)
        return [values.copy()]
    values = np.arange(5.)
    with shared_features() as state:
        feature(values)[0][0] = 99
        assert feature(values)[0][0] == 0
        assert len(calls) == 1 and state['hits'] == 1
        values[0] = 2
        assert feature(values)[0][0] == 2
        assert len(calls) == 2
        for i in range(MAX_FEATURES + 10):
            feature(np.array([i]))
        assert len(state['cache']) == MAX_FEATURES
    assert not state['cache']
    feature(values)
    feature(values)
    assert state['hits'] == 1  # No process-wide retention outside the scope.


def test_nested_failure_restores_outer_scope():
    @shared_feature
    def feature():
        return 1
    with shared_features() as outer:
        feature()
        with pytest.raises(RuntimeError):
            with shared_features() as inner:
                feature()
                raise RuntimeError('cancelled computation')
        feature()
        assert outer['hits'] == 1
        assert not inner['cache']


def test_concurrent_instruments_do_not_share_scopes():
    @shared_feature
    def feature(value):
        return value
    async def scan(value):
        with shared_features() as state:
            feature(value)
            await asyncio.sleep(0)
            feature(value)
            return state['hits'], state['computed']
    async def run():
        assert await asyncio.gather(scan(1), scan(1)) == [(1, 1), (1, 1)]
    asyncio.run(run())
