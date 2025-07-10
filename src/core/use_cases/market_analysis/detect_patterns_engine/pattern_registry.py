# src/core/use_cases/market_analysis/detect_patterns/pattern_registry.py
"""
Central pattern registry and decorator for registering pattern detection functions.
"""

pattern_registry = {}

def register_pattern(name, category, types=None):
    def decorator(func):
        pattern_registry[name] = {
            "function": func,
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