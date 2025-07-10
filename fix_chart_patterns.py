#!/usr/bin/env python3
"""
Script to refactor all chart pattern functions to standardized dictionary format.
Removes context dependencies and ensures consistent array definitions.
"""

import re
import os

def fix_chart_patterns():
    file_path = "src/core/use_cases/market_analysis/detect_patterns_engine/chart_patterns.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove context references and fix array definitions
    content = re.sub(
        r'context\._pattern_key_levels\s*=\s*\{([^}]+)\}',
        r'# Key levels stored in return dictionary',
        content
    )
    
    # Fix 2: Replace tuple returns with dictionary returns
    # Pattern: return True, round(min(confidence, 0.95), 2), pattern_type
    content = re.sub(
        r'return True, round\(min\(confidence, 0\.95\), 2\), pattern_type',
        r'return {\n            "pattern_name": pattern_type,\n            "confidence": round(min(confidence, 0.95), 2),\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "support": bottom_band,\n                "resistance": top_band,\n                "pattern_height": top_band - bottom_band\n            }\n        }',
        content
    )
    
    # Fix 3: Replace other tuple returns
    content = re.sub(
        r'return True, confidence, pattern_type',
        r'return {\n            "pattern_name": pattern_type,\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "support": min(lows),\n                "resistance": max(highs),\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Fix 4: Replace return None with return None (keep as is)
    # This is already correct
    
    # Fix 5: Add missing array definitions where needed
    # Look for functions that use highs, lows, closes, opens without defining them
    content = re.sub(
        r'def detect_([a-zA-Z_]+)\(ohlcv: dict\):',
        r'def detect_\1(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])\n        lows = np.array(ohlcv[\'low\'])\n        closes = np.array(ohlcv[\'close\'])\n        opens = np.array(ohlcv[\'open\'])',
        content
    )
    
    # Fix 6: Handle specific pattern functions that need custom key levels
    # Rectangle pattern
    content = re.sub(
        r'# Store rectangle-specific key levels for plotting\n\s*context\._pattern_key_levels\s*=\s*\{([^}]+)\}\n\s*return True, round\(min\(confidence, 0\.95\), 2\), pattern_type',
        r'return {\n            "pattern_name": pattern_type,\n            "confidence": round(min(confidence, 0.95), 2),\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "rectangle_top": top_band,\n                "rectangle_bottom": bottom_band,\n                "rectangle_height": top_band - bottom_band,\n                "top_touches": top_touches,\n                "bottom_touches": bot_touches,\n                "touch_quality": (avg_top_quality + avg_bot_quality) / 2\n            }\n        }',
        content
    )
    
    # Fix 7: Handle pennant pattern specifically
    content = re.sub(
        r'return True, confidence, pattern_type',
        r'return {\n            "pattern_name": pattern_type,\n            "confidence": confidence,\n            "start_index": consolidation_start,\n            "end_index": consolidation_end,\n            "key_levels": {\n                "flagpole_start": best_flagpole_start,\n                "flagpole_end": best_flagpole_end,\n                "flagpole_change": best_flagpole_change,\n                "consolidation_start": consolidation_start,\n                "consolidation_end": consolidation_end,\n                "high_slope": high_slope,\n                "low_slope": low_slope\n            }\n        }',
        content
    )
    
    # Fix 8: Handle triangle pattern
    content = re.sub(
        r'return True, confidence, triangle_type',
        r'return {\n            "pattern_name": triangle_type,\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_trendline_slope": upper_slope,\n                "lower_trendline_slope": lower_slope,\n                "convergence_point": convergence_point,\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Fix 9: Handle head and shoulders pattern
    content = re.sub(
        r'return True, confidence, pattern_type',
        r'return {\n            "pattern_name": pattern_type,\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "left_shoulder": left_shoulder,\n                "head": head,\n                "right_shoulder": right_shoulder,\n                "neckline": neckline,\n                "pattern_height": head - neckline\n            }\n        }',
        content
    )
    
    # Fix 10: Handle double top/bottom patterns
    content = re.sub(
        r'return True, confidence, "double_top"',
        r'return {\n            "pattern_name": "double_top",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "first_peak": first_peak,\n                "second_peak": second_peak,\n                "neckline": neckline,\n                "pattern_height": max(first_peak, second_peak) - neckline\n            }\n        }',
        content
    )
    
    content = re.sub(
        r'return True, confidence, "double_bottom"',
        r'return {\n            "pattern_name": "double_bottom",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "first_trough": first_trough,\n                "second_trough": second_trough,\n                "neckline": neckline,\n                "pattern_height": neckline - min(first_trough, second_trough)\n            }\n        }',
        content
    )
    
    # Fix 11: Handle triple top/bottom patterns
    content = re.sub(
        r'return True, confidence, "triple_top"',
        r'return {\n            "pattern_name": "triple_top",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "first_peak": first_peak,\n                "second_peak": second_peak,\n                "third_peak": third_peak,\n                "neckline": neckline,\n                "pattern_height": max(first_peak, second_peak, third_peak) - neckline\n            }\n        }',
        content
    )
    
    content = re.sub(
        r'return True, confidence, "triple_bottom"',
        r'return {\n            "pattern_name": "triple_bottom",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "first_trough": first_trough,\n                "second_trough": second_trough,\n                "third_trough": third_trough,\n                "neckline": neckline,\n                "pattern_height": neckline - min(first_trough, second_trough, third_trough)\n            }\n        }',
        content
    )
    
    # Fix 12: Handle wedge patterns
    content = re.sub(
        r'return True, confidence, "wedge_rising"',
        r'return {\n            "pattern_name": "wedge_rising",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_trendline_slope": upper_slope,\n                "lower_trendline_slope": lower_slope,\n                "convergence_point": convergence_point,\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    content = re.sub(
        r'return True, confidence, "wedge_falling"',
        r'return {\n            "pattern_name": "wedge_falling",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_trendline_slope": upper_slope,\n                "lower_trendline_slope": lower_slope,\n                "convergence_point": convergence_point,\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Fix 13: Handle flag patterns
    content = re.sub(
        r'return True, confidence, "flag_bullish"',
        r'return {\n            "pattern_name": "flag_bullish",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "flagpole_start": flagpole_start,\n                "flagpole_end": flagpole_end,\n                "flag_start": flag_start,\n                "flag_end": flag_end,\n                "flag_support": flag_support,\n                "flag_resistance": flag_resistance\n            }\n        }',
        content
    )
    
    content = re.sub(
        r'return True, confidence, "flag_bearish"',
        r'return {\n            "pattern_name": "flag_bearish",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "flagpole_start": flagpole_start,\n                "flagpole_end": flagpole_end,\n                "flag_start": flag_start,\n                "flag_end": flag_end,\n                "flag_support": flag_support,\n                "flag_resistance": flag_resistance\n            }\n        }',
        content
    )
    
    # Fix 14: Handle channel pattern
    content = re.sub(
        r'return True, confidence, channel_type',
        r'return {\n            "pattern_name": channel_type,\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_channel": upper_channel,\n                "lower_channel": lower_channel,\n                "channel_slope": channel_slope,\n                "channel_height": upper_channel - lower_channel\n            }\n        }',
        content
    )
    
    # Fix 15: Handle other patterns with generic key levels
    patterns_to_fix = [
        "island_reversal", "cup_and_handle", "diamond_top", "bump_and_run",
        "cup_with_handle", "inverse_cup_and_handle", "horn_top", 
        "broadening_wedge", "pipe_bottom", "catapult", "scallop", "tower_top"
    ]
    
    for pattern in patterns_to_fix:
        content = re.sub(
            rf'return True, confidence, "{pattern}"',
            rf'return {{\n            "pattern_name": "{pattern}",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {{\n                "support": min(lows),\n                "resistance": max(highs),\n                "pattern_height": max(highs) - min(lows)\n            }}\n        }}',
            content
        )
    
    # Fix 16: Handle zigzag pattern
    content = re.sub(
        r'return True, confidence, "zigzag"',
        r'return {\n            "pattern_name": "zigzag",\n            "confidence": confidence,\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "zigzag_points": zigzag_points,\n                "pattern_length": len(zigzag_points),\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Chart patterns refactoring completed!")

if __name__ == "__main__":
    fix_chart_patterns() 