# src/core/use_cases/market_analysis/detect_patterns_engine/__init__.py
"""
Pattern detection engine initialization.
This module imports all pattern detection modules to ensure they are registered.
"""

# Import all pattern modules to ensure registration
from . import candlestick_patterns
from . import chart_patterns  
from . import harmonic_patterns

# Import the pattern registry
from .pattern_registry import pattern_registry

# Create the initialized pattern registry that the rest of the code expects
initialized_pattern_registry = pattern_registry

# Import PatternDetector from the main detect_patterns module
from ..detect_patterns import PatternDetector

# Export the main classes and registry
__all__ = [
    'PatternDetector',
    'initialized_pattern_registry',
    'pattern_registry'
]
