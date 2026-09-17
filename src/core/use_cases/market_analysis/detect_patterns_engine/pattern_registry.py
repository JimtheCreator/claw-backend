# src/core/use_cases/market_analysis/detect_patterns/pattern_registry.py
"""
Central pattern registry and decorator for registering pattern detection functions.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

pattern_registry = {}
_strict_errors = ContextVar("pattern_detector_strict_errors", default=False)


def strict_errors_enabled():
    """Whether this detector invocation must expose internal failures."""
    return _strict_errors.get()


@contextmanager
def strict_detector_errors():
    """Opt into failure propagation without changing concurrent legacy callers.

    Registered scanner entry points use this scope. Legacy chart/forecast callers
    keep their historical fallback behavior. Always reset the task-local policy,
    including when detection raises or the task is cancelled.
    """
    token = _strict_errors.set(True)
    try:
        yield
    finally:
        _strict_errors.reset(token)


def register_pattern(name, category, types=None):
    def decorator(func):
        @wraps(func)
        async def strict_function(*args, **kwargs):
            with strict_detector_errors():
                return await func(*args, **kwargs)

        pattern_registry[name] = {
            "function": func,
            "strict_function": strict_function,
            "category": category,
            "types": types if types is not None else [name]
        }
        return func
    return decorator

def get_patterns_by_category(category):
    return {name: info for name, info in pattern_registry.items() if info["category"] == category}

def get_pattern_function(name):
    return pattern_registry.get(name, {}).get("function")

def get_patterns_by_type(pattern_type):
    return {name: info for name, info in pattern_registry.items() if pattern_type in (info.get("types") or [])}
