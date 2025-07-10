#!/usr/bin/env python3
"""
Script to add missing 'closes' array definitions to pattern functions.
"""

import re

def fix_missing_closes():
    file_path = "src/core/use_cases/market_analysis/detect_patterns_engine/chart_patterns.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add missing closes definitions to functions that use closes but don't define it
    
    # Fix 1: detect_head_and_shoulders
    content = re.sub(
        r'def detect_head_and_shoulders\(ohlcv: dict\):',
        r'def detect_head_and_shoulders(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 2: detect_wedge_rising
    content = re.sub(
        r'def detect_wedge_rising\(ohlcv: dict\):',
        r'def detect_wedge_rising(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 3: detect_wedge_falling
    content = re.sub(
        r'def detect_wedge_falling\(ohlcv: dict\):',
        r'def detect_wedge_falling(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 4: detect_flag_bullish
    content = re.sub(
        r'def detect_flag_bullish\(ohlcv: dict\):',
        r'def detect_flag_bullish(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 5: detect_flag_bearish
    content = re.sub(
        r'def detect_flag_bearish\(ohlcv: dict\):',
        r'def detect_flag_bearish(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 6: detect_cup_and_handle
    content = re.sub(
        r'def detect_cup_and_handle\(ohlcv: dict\):',
        r'def detect_cup_and_handle(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 7: detect_diamond_top
    content = re.sub(
        r'def detect_diamond_top\(ohlcv: dict\):',
        r'def detect_diamond_top(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 8: detect_horn_top
    content = re.sub(
        r'def detect_horn_top\(ohlcv: dict\):',
        r'def detect_horn_top(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 9: detect_broadening_wedge
    content = re.sub(
        r'def detect_broadening_wedge\(ohlcv: dict\):',
        r'def detect_broadening_wedge(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 10: detect_pipe_bottom
    content = re.sub(
        r'def detect_pipe_bottom\(ohlcv: dict\):',
        r'def detect_pipe_bottom(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 11: detect_catapult
    content = re.sub(
        r'def detect_catapult\(ohlcv: dict\):',
        r'def detect_catapult(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 12: detect_scallop
    content = re.sub(
        r'def detect_scallop\(ohlcv: dict\):',
        r'def detect_scallop(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 13: detect_tower_top
    content = re.sub(
        r'def detect_tower_top\(ohlcv: dict\):',
        r'def detect_tower_top(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Missing 'closes' definitions added!")

if __name__ == "__main__":
    fix_missing_closes() 