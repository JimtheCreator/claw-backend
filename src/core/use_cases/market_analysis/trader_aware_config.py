"""
Configuration for Trader-Aware Analysis System

This module provides configuration settings for the trader-aware analysis system,
including trend detection, zone detection, pattern scanning, and scoring parameters.
"""

import os
from typing import Dict, Any, List

# Default configuration for trader-aware analysis
DEFAULT_CONFIG = {
    # Trend Detection Settings
    "trend_detection": {
        "atr_period": 14,
        "swing_lookback": 20,
        "trend_strength_threshold": 0.6,
        "min_swing_distance": 0.02,  # 2% minimum distance between swings
        "trendline_tolerance": 0.01,  # 1% tolerance for trendline fitting
    },
    
    # Zone Detection Settings
    "zone_detection": {
        "support_resistance": {
            "lookback_period": 200,
            "price_proximity_factor": 0.01,  # 1% proximity for clustering
            "min_touches": 2,
            "strength_threshold": 0.5,
        },
        "demand_supply": {
            "lookback_period": 200,
            "pivot_order": 5,
            "zone_proximity_factor": 0.01,
            "min_touches_for_strong_zone": 2,
            "volume_weight": 0.3,
        }
    },
    
    # Pattern Scanning Settings
    "pattern_scanning": {
        "zone_proximity_threshold": 0.02,  # 2% proximity to zones
        "trend_alignment_weight": 0.4,
        "zone_relevance_weight": 0.3,
        "pattern_confidence_threshold": 0.6,
        "max_patterns_per_zone": 3,
        "scan_window_sizes": [20, 50, 100],  # Different window sizes for pattern detection
    },
    
    # Candlestick Confirmation Settings
    "candle_confirmation": {
        "confirmation_patterns": [
            "doji", "hammer", "shooting_star", "engulfing", "harami",
            "morning_star", "evening_star", "three_white_soldiers", "three_black_crows"
        ],
        "confirmation_weight": 0.2,
        "volume_confirmation_weight": 0.1,
        "min_confirmation_strength": 0.5,
    },
    
    # Scoring System Settings
    "scoring": {
        "weights": {
            "trend_alignment": 0.25,
            "zone_relevance": 0.25,
            "pattern_clarity": 0.20,
            "candle_confirmation": 0.15,
            "key_level_precision": 0.15,
        },
        "bonus_factors": {
            "volume_spike": 0.1,
            "multiple_timeframe": 0.05,
            "news_catalyst": 0.05,
        },
        "penalty_factors": {
            "overbought_oversold": -0.1,
            "low_volume": -0.05,
            "conflicting_signals": -0.15,
        },
        "min_score_threshold": 0.6,
        "max_setups_returned": 10,
    },
    
    # Performance Settings
    "performance": {
        "max_candles": 1000,
        "batch_size": 100,
        "parallel_processing": True,
        "cache_results": True,
        "cache_ttl": 300,  # 5 minutes
    },
    
    # Output Settings
    "output": {
        "include_raw_data": False,
        "include_chart_data": True,
        "include_trading_signals": True,
        "include_risk_metrics": True,
        "format_for_frontend": True,
    }
}

def get_config(override_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration for trader-aware analysis with environment overrides.
    
    Args:
        override_config: Optional configuration overrides
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Apply environment variable overrides
    env_overrides = _get_env_overrides()
    _deep_merge(config, env_overrides)
    
    # Apply user-provided overrides
    if override_config:
        _deep_merge(config, override_config)
    
    return config

def _get_env_overrides() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.
    
    Returns:
        Dictionary of environment-based overrides
    """
    overrides = {}
    
    # Trend detection overrides
    if os.getenv("TRA_TREND_ATR_PERIOD"):
        overrides.setdefault("trend_detection", {})["atr_period"] = int(os.getenv("TRA_TREND_ATR_PERIOD"))
    
    if os.getenv("TRA_TREND_STRENGTH_THRESHOLD"):
        overrides.setdefault("trend_detection", {})["trend_strength_threshold"] = float(os.getenv("TRA_TREND_STRENGTH_THRESHOLD"))
    
    # Zone detection overrides
    if os.getenv("TRA_ZONE_LOOKBACK_PERIOD"):
        overrides.setdefault("zone_detection", {}).setdefault("support_resistance", {})["lookback_period"] = int(os.getenv("TRA_ZONE_LOOKBACK_PERIOD"))
        overrides.setdefault("zone_detection", {}).setdefault("demand_supply", {})["lookback_period"] = int(os.getenv("TRA_ZONE_LOOKBACK_PERIOD"))
    
    if os.getenv("TRA_ZONE_PROXIMITY_FACTOR"):
        proximity = float(os.getenv("TRA_ZONE_PROXIMITY_FACTOR"))
        overrides.setdefault("zone_detection", {}).setdefault("support_resistance", {})["price_proximity_factor"] = proximity
        overrides.setdefault("zone_detection", {}).setdefault("demand_supply", {})["zone_proximity_factor"] = proximity
    
    # Pattern scanning overrides
    if os.getenv("TRA_PATTERN_CONFIDENCE_THRESHOLD"):
        overrides.setdefault("pattern_scanning", {})["pattern_confidence_threshold"] = float(os.getenv("TRA_PATTERN_CONFIDENCE_THRESHOLD"))
    
    if os.getenv("TRA_ZONE_PROXIMITY_THRESHOLD"):
        overrides.setdefault("pattern_scanning", {})["zone_proximity_threshold"] = float(os.getenv("TRA_ZONE_PROXIMITY_THRESHOLD"))
    
    # Scoring overrides
    if os.getenv("TRA_MIN_SCORE_THRESHOLD"):
        overrides.setdefault("scoring", {})["min_score_threshold"] = float(os.getenv("TRA_MIN_SCORE_THRESHOLD"))
    
    if os.getenv("TRA_MAX_SETUPS_RETURNED"):
        overrides.setdefault("scoring", {})["max_setups_returned"] = int(os.getenv("TRA_MAX_SETUPS_RETURNED"))
    
    # Performance overrides
    if os.getenv("TRA_MAX_CANDLES"):
        overrides.setdefault("performance", {})["max_candles"] = int(os.getenv("TRA_MAX_CANDLES"))
    
    if os.getenv("TRA_PARALLEL_PROCESSING"):
        overrides.setdefault("performance", {})["parallel_processing"] = os.getenv("TRA_PARALLEL_PROCESSING").lower() == "true"
    
    return overrides

def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> None:
    """
    Deep merge override dictionary into base dictionary.
    
    Args:
        base_dict: Base dictionary to merge into
        override_dict: Dictionary with overrides
    """
    for key, value in override_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_merge(base_dict[key], value)
        else:
            base_dict[key] = value

def get_trend_config() -> Dict[str, Any]:
    """Get trend detection configuration."""
    return get_config()["trend_detection"]

def get_zone_config() -> Dict[str, Any]:
    """Get zone detection configuration."""
    return get_config()["zone_detection"]

def get_pattern_config() -> Dict[str, Any]:
    """Get pattern scanning configuration."""
    return get_config()["pattern_scanning"]

def get_candle_config() -> Dict[str, Any]:
    """Get candlestick confirmation configuration."""
    return get_config()["candle_confirmation"]

def get_scoring_config() -> Dict[str, Any]:
    """Get scoring system configuration."""
    return get_config()["scoring"]

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return get_config()["performance"]

def get_output_config() -> Dict[str, Any]:
    """Get output configuration."""
    return get_config()["output"]

# Preset configurations for different use cases
PRESET_CONFIGS = {
    "conservative": {
        "scoring": {
            "min_score_threshold": 0.7,
            "weights": {
                "trend_alignment": 0.30,
                "zone_relevance": 0.30,
                "pattern_clarity": 0.25,
                "candle_confirmation": 0.10,
                "key_level_precision": 0.05,
            }
        },
        "pattern_scanning": {
            "pattern_confidence_threshold": 0.7,
            "zone_proximity_threshold": 0.015,
        }
    },
    
    "aggressive": {
        "scoring": {
            "min_score_threshold": 0.5,
            "weights": {
                "trend_alignment": 0.20,
                "zone_relevance": 0.20,
                "pattern_clarity": 0.15,
                "candle_confirmation": 0.20,
                "key_level_precision": 0.25,
            }
        },
        "pattern_scanning": {
            "pattern_confidence_threshold": 0.5,
            "zone_proximity_threshold": 0.03,
        }
    },
    
    "balanced": {
        "scoring": {
            "min_score_threshold": 0.6,
            "weights": {
                "trend_alignment": 0.25,
                "zone_relevance": 0.25,
                "pattern_clarity": 0.20,
                "candle_confirmation": 0.15,
                "key_level_precision": 0.15,
            }
        },
        "pattern_scanning": {
            "pattern_confidence_threshold": 0.6,
            "zone_proximity_threshold": 0.02,
        }
    },
    
    "high_frequency": {
        "performance": {
            "max_candles": 500,
            "batch_size": 50,
            "parallel_processing": True,
        },
        "scoring": {
            "min_score_threshold": 0.5,
            "max_setups_returned": 5,
        },
        "pattern_scanning": {
            "scan_window_sizes": [10, 20, 30],
        }
    }
}

def get_preset_config(preset_name: str, override_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get a preset configuration with optional overrides.
    
    Args:
        preset_name: Name of the preset ("conservative", "aggressive", "balanced", "high_frequency")
        override_config: Optional additional overrides
        
    Returns:
        Configuration dictionary
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESET_CONFIGS.keys())}")
    
    config = get_config()
    preset_overrides = PRESET_CONFIGS[preset_name]
    _deep_merge(config, preset_overrides)
    
    if override_config:
        _deep_merge(config, override_config)
    
    return config 