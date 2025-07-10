#!/usr/bin/env python3
"""
Script to fix remaining issues in chart patterns refactoring.
"""

import re

def fix_remaining_issues():
    file_path = "src/core/use_cases/market_analysis/detect_patterns_engine/chart_patterns.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace context.deviation_pct with a default value
    content = re.sub(
        r'context\.deviation_pct',
        '2.0',  # Default deviation percentage
        content
    )
    
    # Fix 2: Fix the triangle pattern return statement
    content = re.sub(
        r'return True, round\(confidence, 2\), triangle_type',
        r'return {\n            "pattern_name": triangle_type,\n            "confidence": round(confidence, 2),\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_trendline_slope": peak_slope,\n                "lower_trendline_slope": trough_slope,\n                "convergence_point": None,  # Would need calculation\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Fix 3: Fix the channel pattern return statement
    content = re.sub(
        r'return True, round\(confidence, 2\), channel_type',
        r'return {\n            "pattern_name": channel_type,\n            "confidence": round(confidence, 2),\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "upper_channel": upper_channel,\n                "lower_channel": lower_channel,\n                "channel_slope": channel_slope,\n                "channel_height": upper_channel - lower_channel\n            }\n        }',
        content
    )
    
    # Fix 4: Add missing array definitions for functions that need them
    # Look for functions that use highs, lows, closes, opens without defining them at the start
    content = re.sub(
        r'def detect_zigzag\(ohlcv: dict\):',
        r'def detect_zigzag(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])\n        lows = np.array(ohlcv[\'low\'])\n        closes = np.array(ohlcv[\'close\'])\n        opens = np.array(ohlcv[\'open\'])',
        content
    )
    
    content = re.sub(
        r'def detect_triangle\(ohlcv: dict\):',
        r'def detect_triangle(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])\n        lows = np.array(ohlcv[\'low\'])\n        closes = np.array(ohlcv[\'close\'])\n        opens = np.array(ohlcv[\'open\'])',
        content
    )
    
    content = re.sub(
        r'def detect_head_and_shoulders\(ohlcv: dict\):',
        r'def detect_head_and_shoulders(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])\n        lows = np.array(ohlcv[\'low\'])\n        closes = np.array(ohlcv[\'close\'])\n        opens = np.array(ohlcv[\'open\'])',
        content
    )
    
    # Fix 5: Remove the incomplete key_levels definition in triangle pattern
    content = re.sub(
        r'# Key levels stored in return dictionary,\n\s*\'support_line\': \{([^}]+)\}\n\s*\}',
        r'# Key levels stored in return dictionary',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Remaining chart pattern issues fixed!")

if __name__ == "__main__":
    fix_remaining_issues() 