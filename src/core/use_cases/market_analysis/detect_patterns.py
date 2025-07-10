# src/core/use_cases/market_analysis/detect_patterns.py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Any, Optional, Callable
from common.logger import logger
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

# Import all pattern modules to ensure registration
from .detect_patterns_engine import candlestick_patterns, chart_patterns, harmonic_patterns
from .detect_patterns_engine.pattern_registry import get_patterns_by_category, get_pattern_function, register_pattern

class PatternDetectionContext:
    def __init__(self, min_swings=3):
        self.min_swings = min_swings
        self.pattern_key_levels = {}

class PatternDetector:
    def __init__(self):
        self.min_swings=3
        self.context = PatternDetectionContext(min_swings=self.min_swings)
    
    async def detect(self, pattern_name: str, ohlcv: dict) -> Tuple[bool, float, str]:
        func = get_pattern_function(pattern_name)
        if not func:
            raise ValueError(f"Unsupported pattern: {pattern_name}")
        return await func(ohlcv, self.context)

    async def detect_by_category(self, category: str, ohlcv: dict) -> Dict[str, Tuple[bool, float, str]]:
        results = {}
        for name, info in get_patterns_by_category(category).items():
            found, confidence, ptype = await info["function"](ohlcv, self.context)
            results[name] = (found, confidence, ptype)
        return results

    async def detect_multiple(self, pattern_names: list, ohlcv: dict) -> Dict[str, Tuple[bool, float, str]]:
        results = {}
        for name in pattern_names:
            func = get_pattern_function(name)
            if func:
                found, confidence, ptype = await func(ohlcv, self.context)
                results[name] = (found, confidence, ptype)
        return results


    async def find_key_levels(self, ohlcv: dict, pattern_type: Optional[str] = None) -> Dict[str, float]:
        """Find key price levels from detected patterns and swing points, pattern-specific."""
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        opens = np.array(ohlcv['open']) if 'open' in ohlcv else None


        # Single-candle patterns (1 candle required)
        if pattern_type is not None and any(doji_type in pattern_type for doji_type in ["doji", "standard_doji", "gravestone_doji", "dragonfly_doji"]):
            return {
                "latest_close": float(closes[-1]),
                "avg_high_5": float(np.mean(highs[-5:])),
                "avg_low_5": float(np.mean(lows[-5:])),
                "doji_high": float(highs[-1]),
                "doji_low": float(lows[-1]),
                "doji_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                "doji_close": float(closes[-1])
            }

        # Spinning top pattern
        if pattern_type is not None and "spinning_top" in pattern_type:
            return {
                "latest_close": float(closes[-1]),
                "avg_high_5": float(np.mean(highs[-5:])),
                "avg_low_5": float(np.mean(lows[-5:])),
                "spinning_top_high": float(highs[-1]),
                "spinning_top_low": float(lows[-1]),
                "spinning_top_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                "spinning_top_close": float(closes[-1])
            }

        # Marubozu patterns
        if pattern_type is not None and "marubozu" in pattern_type:
            body_size = float(abs(closes[-1] - opens[-1])) if opens is not None and len(opens) > 0 and len(closes) > 0 else 0.0
            return {
                "latest_close": float(closes[-1]),
                "avg_high_5": float(np.mean(highs[-5:])),
                "avg_low_5": float(np.mean(lows[-5:])),
                "marubozu_high": float(highs[-1]),
                "marubozu_low": float(lows[-1]),
                "marubozu_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                "marubozu_close": float(closes[-1]),
                "body_size": body_size
            }

        # Two-candle patterns (2 candles required)
        if pattern_type is not None and "engulfing" in pattern_type:
            if opens is not None and len(opens) >= 2 and len(closes) >= 2:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-2]),
                    "prev_close": float(closes[-2]),
                    "curr_open": float(opens[-1]),
                    "curr_close": float(closes[-1]),
                    "engulfed_range": float(abs(closes[-1] - opens[-2]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "prev_close": float(closes[-1]),
                    "curr_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "curr_close": float(closes[-1]),
                    "engulfed_range": 0.0
                }

        # Dark cloud cover and piercing pattern
        if pattern_type is not None and ("dark_cloud_cover" in pattern_type or "piercing_pattern" in pattern_type):
            if opens is not None and len(opens) >= 2 and len(closes) >= 2:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-2]),
                    "prev_close": float(closes[-2]),
                    "curr_open": float(opens[-1]),
                    "curr_close": float(closes[-1]),
                    "pattern_midpoint": float((opens[-2] + closes[-2]) / 2)
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "prev_close": float(closes[-1]),
                    "curr_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "curr_close": float(closes[-1]),
                    "pattern_midpoint": float(closes[-1])
                }

        # Kicker patterns
        if pattern_type is not None and "kicker" in pattern_type:
            if opens is not None and len(opens) >= 2 and len(closes) >= 2:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-2]),
                    "prev_close": float(closes[-2]),
                    "curr_open": float(opens[-1]),
                    "curr_close": float(closes[-1]),
                    "gap_size": float(abs(opens[-1] - closes[-2]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "prev_close": float(closes[-1]),
                    "curr_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "curr_close": float(closes[-1]),
                    "gap_size": 0.0
                }

        # Harami patterns
        if pattern_type is not None and "harami" in pattern_type:
            if opens is not None and len(opens) >= 2 and len(closes) >= 2:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-2]),
                    "prev_close": float(closes[-2]),
                    "curr_open": float(opens[-1]),
                    "curr_close": float(closes[-1]),
                    "harami_containment": float(abs(opens[-2] - closes[-2]) - abs(opens[-1] - closes[-1]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "prev_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "prev_close": float(closes[-1]),
                    "curr_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                    "curr_close": float(closes[-1]),
                    "harami_containment": 0.0
                }

        # Tweezers patterns
        if pattern_type is not None and ("tweezers_top" in pattern_type or "tweezers_bottom" in pattern_type):
            if len(highs) >= 2 and len(lows) >= 2:
                return {
                    "latest_close": float(closes[-1]),
                    "tweezers_high": float(highs[-1]),
                    "tweezers_low": float(lows[-1]),
                    "prev_high": float(highs[-2]),
                    "prev_low": float(lows[-2]),
                    "tweezers_level": float(highs[-1]) if "top" in pattern_type else float(lows[-1])
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "tweezers_high": float(highs[-1]),
                    "tweezers_low": float(lows[-1]),
                    "prev_high": float(highs[-1]),
                    "prev_low": float(lows[-1]),
                    "tweezers_level": float(highs[-1]) if "top" in pattern_type else float(lows[-1])
                }

        # Three-candle patterns (3 candles required)
        if pattern_type is not None and any(three_candle in pattern_type for three_candle in [
            "three_outside_up", "three_outside_down", "three_inside_up", "three_inside_down",
            "three_white_soldiers", "three_black_crows", "three_line_strike"
        ]):
            if len(closes) >= 3:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-3]),
                    "second_candle_close": float(closes[-2]),
                    "third_candle_close": float(closes[-1]),
                    "pattern_range": float(max(closes[-3:]) - min(closes[-3:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-1]),
                    "second_candle_close": float(closes[-1]),
                    "third_candle_close": float(closes[-1]),
                    "pattern_range": 0.0
                }

        # Evening star and morning star
        if pattern_type is not None and ("evening_star" in pattern_type or "morning_star" in pattern_type):
            if len(closes) >= 3:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-3]),
                    "second_candle_close": float(closes[-2]),
                    "third_candle_close": float(closes[-1]),
                    "star_gap_up": float(closes[-2] - closes[-3]) if "morning" in pattern_type else 0.0,
                    "star_gap_down": float(closes[-3] - closes[-2]) if "evening" in pattern_type else 0.0
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-1]),
                    "second_candle_close": float(closes[-1]),
                    "third_candle_close": float(closes[-1]),
                    "star_gap_up": 0.0,
                    "star_gap_down": 0.0
                }

        # Five-candle patterns (5 candles required)
        if pattern_type is not None and any(five_candle in pattern_type for five_candle in [
            "hammer", "hanging_man", "inverted_hammer", "shooting_star", "bullish_shooting_star", "bearish_shooting_star"
        ]):
            lower_shadow = float(closes[-1] - lows[-1]) if len(closes) > 0 and len(lows) > 0 else 0.0
            upper_shadow = float(highs[-1] - closes[-1]) if len(highs) > 0 and len(closes) > 0 else 0.0
            return {
                "latest_close": float(closes[-1]),
                "avg_high_5": float(np.mean(highs[-5:])),
                "avg_low_5": float(np.mean(lows[-5:])),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "pattern_open": float(opens[-1]) if opens is not None else float(closes[-1]),
                "pattern_close": float(closes[-1]),
                "lower_shadow": lower_shadow,
                "upper_shadow": upper_shadow
            }

        # Abandoned baby patterns
        if pattern_type is not None and "abandoned_baby" in pattern_type:
            if len(closes) >= 3:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-3]),
                    "second_candle_close": float(closes[-2]),
                    "third_candle_close": float(closes[-1]),
                    "baby_gap": float(abs(closes[-2] - closes[-3]) + abs(closes[-1] - closes[-2]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "first_candle_close": float(closes[-1]),
                    "second_candle_close": float(closes[-1]),
                    "third_candle_close": float(closes[-1]),
                    "baby_gap": 0.0
                }

        # Six-candle patterns (6 candles required)
        if pattern_type is not None and any(six_candle in pattern_type for six_candle in [
            "hikkake", "bullish_hikkake", "bearish_hikkake", "mat_hold", "bullish_mat_hold", "bearish_mat_hold"
        ]):
            if len(closes) >= 6:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_start": float(closes[-6]),
                    "pattern_end": float(closes[-1]),
                    "pattern_range": float(max(closes[-6:]) - min(closes[-6:])),
                    "avg_high_6": float(np.mean(highs[-6:])),
                    "avg_low_6": float(np.mean(lows[-6:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_start": float(closes[0]),
                    "pattern_end": float(closes[-1]),
                    "pattern_range": float(max(closes) - min(closes)),
                    "avg_high_6": float(np.mean(highs)),
                    "avg_low_6": float(np.mean(lows))
                }

        # Seven-candle patterns (7 candles required)
        if pattern_type is not None and ("rising_three_methods" in pattern_type or "falling_three_methods" in pattern_type):
            if len(closes) >= 7:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_start": float(closes[-7]),
                    "pattern_end": float(closes[-1]),
                    "three_methods_range": float(max(closes[-7:]) - min(closes[-7:])),
                    "avg_high_7": float(np.mean(highs[-7:])),
                    "avg_low_7": float(np.mean(lows[-7:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_start": float(closes[0]),
                    "pattern_end": float(closes[-1]),
                    "three_methods_range": float(max(closes) - min(closes)),
                    "avg_high_7": float(np.mean(highs)),
                    "avg_low_7": float(np.mean(lows))
                }

        # Large patterns (20+ candles required)
        if pattern_type is not None and any(large_pattern in pattern_type for large_pattern in [
            "cup_and_handle", "cup_with_handle", "inverse_cup_and_handle"
        ]):
            if len(closes) >= 20:
                return {
                    "latest_close": float(closes[-1]),
                    "cup_depth": float(min(closes[-20:])),
                    "cup_rim": float(max(closes[-20:])),
                    "handle_start": float(closes[-10]),
                    "handle_end": float(closes[-1]),
                    "pattern_range": float(max(closes[-20:]) - min(closes[-20:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "cup_depth": float(min(closes)),
                    "cup_rim": float(max(closes)),
                    "handle_start": float(closes[len(closes)//2]),
                    "handle_end": float(closes[-1]),
                    "pattern_range": float(max(closes) - min(closes))
                }

        # Medium patterns (12+ candles required)
        if pattern_type is not None and any(medium_pattern in pattern_type for medium_pattern in [
            "rectangle", "ascending_triangle", "descending_triangle", "symmetrical_triangle",
            "ascending_channel", "descending_channel", "horizontal_channel",
            "wedge_falling", "wedge_rising", "broadening_wedge"
        ]):
            if len(closes) >= 12:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_top": float(max(highs[-12:])),
                    "pattern_bottom": float(min(lows[-12:])),
                    "pattern_height": float(max(highs[-12:]) - min(lows[-12:])),
                    "pattern_start": float(closes[-12]),
                    "pattern_end": float(closes[-1])
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "pattern_top": float(max(highs)),
                    "pattern_bottom": float(min(lows)),
                    "pattern_height": float(max(highs) - min(lows)),
                    "pattern_start": float(closes[0]),
                    "pattern_end": float(closes[-1])
                }

        # Flag/Pennant patterns (10+ candles required)
        if pattern_type is not None and any(flag_pattern in pattern_type for flag_pattern in [
            "flag_bullish", "flag_bearish", "pennant", "bullish_pennant", "bearish_pennant"
        ]):
            if len(closes) >= 10:
                return {
                    "latest_close": float(closes[-1]),
                    "flag_pole_start": float(closes[-10]),
                    "flag_pole_end": float(closes[-7]),
                    "flag_start": float(closes[-7]),
                    "flag_end": float(closes[-1]),
                    "pole_height": float(closes[-7] - closes[-10]),
                    "flag_range": float(max(closes[-7:]) - min(closes[-7:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "flag_pole_start": float(closes[0]),
                    "flag_pole_end": float(closes[len(closes)//2]),
                    "flag_start": float(closes[len(closes)//2]),
                    "flag_end": float(closes[-1]),
                    "pole_height": float(closes[len(closes)//2] - closes[0]),
                    "flag_range": float(max(closes[len(closes)//2:]) - min(closes[len(closes)//2:]))
                }

        # Head and shoulders patterns (15+ candles required)
        if pattern_type is not None and any(hs_pattern in pattern_type for hs_pattern in [
            "head_and_shoulders", "bearish_head_and_shoulders", "inverse_head_and_shoulders"
        ]):
            if len(closes) >= 15:
                return {
                    "latest_close": float(closes[-1]),
                    "left_shoulder": float(max(highs[-15:-10])),
                    "head": float(max(highs[-10:-5])),
                    "right_shoulder": float(max(highs[-5:])),
                    "neckline": float(min(lows[-15:])),
                    "pattern_range": float(max(highs[-15:]) - min(lows[-15:]))
                }
            else:
                return {
                    "latest_close": float(closes[-1]),
                    "left_shoulder": float(max(highs[:len(highs)//3])),
                    "head": float(max(highs[len(highs)//3:2*len(highs)//3])),
                    "right_shoulder": float(max(highs[2*len(highs)//3:])),
                    "neckline": float(min(lows)),
                    "pattern_range": float(max(highs) - min(lows))
                }

        # Double top/bottom patterns (10+ candles required)
        if pattern_type is not None and any(double_pattern in pattern_type for double_pattern in [
            "double_top", "double_bottom"
        ]):
            if len(closes) >= 10:
                if "double_top" in pattern_type:
                    return {
                        "latest_close": float(closes[-1]),
                        "first_peak": float(max(highs[-10:-5])),
                        "second_peak": float(max(highs[-5:])),
                        "valley": float(min(lows[-10:])),
                        "pattern_range": float(max(highs[-10:]) - min(lows[-10:]))
                    }
                else:  # double_bottom
                    return {
                        "latest_close": float(closes[-1]),
                        "first_trough": float(min(lows[-10:-5])),
                        "second_trough": float(min(lows[-5:])),
                        "peak": float(max(highs[-10:])),
                        "pattern_range": float(max(highs[-10:]) - min(lows[-10:]))
                    }
            else:
                if "double_top" in pattern_type:
                    return {
                        "latest_close": float(closes[-1]),
                        "first_peak": float(max(highs[:len(highs)//2])),
                        "second_peak": float(max(highs[len(highs)//2:])),
                        "valley": float(min(lows)),
                        "pattern_range": float(max(highs) - min(lows))
                    }
                else:  # double_bottom
                    return {
                        "latest_close": float(closes[-1]),
                        "first_trough": float(min(lows[:len(lows)//2])),
                        "second_trough": float(min(lows[len(lows)//2:])),
                        "peak": float(max(highs)),
                        "pattern_range": float(max(highs) - min(lows))
                    }

        # Triple top/bottom patterns (15+ candles required)
        if pattern_type is not None and any(triple_pattern in pattern_type for triple_pattern in [
            "triple_top", "triple_bottom"
        ]):
            if len(closes) >= 15:
                if "triple_top" in pattern_type:
                    return {
                        "latest_close": float(closes[-1]),
                        "first_peak": float(max(highs[-15:-10])),
                        "second_peak": float(max(highs[-10:-5])),
                        "third_peak": float(max(highs[-5:])),
                        "valley1": float(min(lows[-15:-5])),
                        "valley2": float(min(lows[-5:])),
                        "pattern_range": float(max(highs[-15:]) - min(lows[-15:]))
                    }
                else:  # triple_bottom
                    return {
                        "latest_close": float(closes[-1]),
                        "first_trough": float(min(lows[-15:-10])),
                        "second_trough": float(min(lows[-10:-5])),
                        "third_trough": float(min(lows[-5:])),
                        "peak1": float(max(highs[-15:-5])),
                        "peak2": float(max(highs[-5:])),
                        "pattern_range": float(max(highs[-15:]) - min(lows[-15:]))
                    }
            else:
                if "triple_top" in pattern_type:
                    return {
                        "latest_close": float(closes[-1]),
                        "first_peak": float(max(highs[:len(highs)//3])),
                        "second_peak": float(max(highs[len(highs)//3:2*len(highs)//3])),
                        "third_peak": float(max(highs[2*len(highs)//3:])),
                        "valley1": float(min(lows[:len(lows)//2])),
                        "valley2": float(min(lows[len(lows)//2:])),
                        "pattern_range": float(max(highs) - min(lows))
                    }
                else:  # triple_bottom
                    return {
                        "latest_close": float(closes[-1]),
                        "first_trough": float(min(lows[:len(lows)//3])),
                        "second_trough": float(min(lows[len(lows)//3:2*len(lows)//3])),
                        "third_trough": float(min(lows[2*len(lows)//3:])),
                        "peak1": float(max(highs[:len(highs)//2])),
                        "peak2": float(max(highs[len(highs)//2:])),
                        "pattern_range": float(max(highs) - min(lows))
                    }

        # Special patterns
        if pattern_type is not None and any(special_pattern in pattern_type for special_pattern in [
            "diamond_top", "bump_and_run", "catapult", "scallop", "tower_top", "horn_top", "pipe_bottom", "zigzag"
        ]):
            return {
                "latest_close": float(closes[-1]),
                "pattern_top": float(max(highs)),
                "pattern_bottom": float(min(lows)),
                "pattern_height": float(max(highs) - min(lows)),
                "pattern_start": float(closes[0]),
                "pattern_end": float(closes[-1]),
                "pattern_range": float(max(closes) - min(closes))
            }

        # Default fallback for any unrecognized pattern
        return {
            "latest_close": float(closes[-1]),
            "pattern_top": float(max(highs)),
            "pattern_bottom": float(min(lows)),
            "pattern_height": float(max(highs) - min(lows)),
            "pattern_start": float(closes[0]),
            "pattern_end": float(closes[-1])
        }