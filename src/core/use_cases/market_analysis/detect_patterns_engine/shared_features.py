"""Bounded, task-local feature reuse for one immutable detector input window.

Legacy callers stay uncached. Scanner scopes never retain an instrument's data
past that scan or share mutable feature results between detectors.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

import numpy as np
from scipy.signal import argrelextrema as _argrelextrema

_scope = ContextVar("detector_feature_scope", default=None)
MAX_FEATURES = 64


def _key(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple((k, _key(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        # OHLCV columns are flat lists of immutable numbers/strings. Avoid a
        # Python recursive call per candle on every detector invocation.
        flat = tuple(value)
        try:
            hash(flat)
            return ("list", flat)
        except TypeError:
            return ("list", tuple(_key(v) for v in value))
    if isinstance(value, tuple):
        return tuple(_key(v) for v in value)
    return value


@contextmanager
def shared_features():
    state = {"cache": {}, "hits": 0, "computed": 0}
    token = _scope.set(state)
    try:
        yield state
    finally:
        _scope.reset(token)
        state["cache"].clear()


def shared_feature(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = _scope.get()
        if state is None:
            return function(*args, **kwargs)
        key = (function, _key(args), _key(kwargs))
        if key in state["cache"]:
            state["hits"] += 1
            return deepcopy(state["cache"][key])
        result = function(*args, **kwargs)
        state["computed"] += 1
        if len(state["cache"]) < MAX_FEATURES:
            state["cache"][key] = deepcopy(result)
        return result
    return wrapped


# Keep SciPy's exact comparison, edge and order semantics.
argrelextrema = shared_feature(_argrelextrema)
