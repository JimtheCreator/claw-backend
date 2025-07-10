#!/usr/bin/env python3
"""
Script to check and fix array definition mismatches in chart_patterns.py
"""

import re

def fix_array_mismatches():
    file_path = "src/core/use_cases/market_analysis/detect_patterns_engine/chart_patterns.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: detect_head_and_shoulders - closes is incorrectly set to opens
    content = re.sub(
        r'closes = np\.array\(ohlcv\[\'open\'\]\)',
        r'closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 2: detect_channel - missing closes definition
    content = re.sub(
        r'def detect_channel\(ohlcv: dict\):',
        r'def detect_channel(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 3: detect_island_reversal - return tuple instead of dict
    content = re.sub(
        r'return is_island, round\(confidence, 2\), island_type',
        r'return {\n            "pattern_name": island_type,\n            "confidence": round(confidence, 2),\n            "start_index": 0,\n            "end_index": len(closes) - 1,\n            "key_levels": {\n                "support": min(lows),\n                "resistance": max(highs),\n                "pattern_height": max(highs) - min(lows)\n            }\n        }',
        content
    )
    
    # Fix 4: detect_diamond_top - missing closes definition
    content = re.sub(
        r'def detect_diamond_top\(ohlcv: dict\):',
        r'def detect_diamond_top(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 5: detect_horn_top - missing closes and lows definitions
    content = re.sub(
        r'def detect_horn_top\(ohlcv: dict\):',
        r'def detect_horn_top(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])\n        lows = np.array(ohlcv[\'low\'])',
        content
    )
    
    # Fix 6: detect_pipe_bottom - missing closes definition
    content = re.sub(
        r'def detect_pipe_bottom\(ohlcv: dict\):',
        r'def detect_pipe_bottom(ohlcv: dict):\n        closes = np.array(ohlcv[\'close\'])',
        content
    )
    
    # Fix 7: detect_catapult - missing highs definition
    content = re.sub(
        r'def detect_catapult\(ohlcv: dict\):',
        r'def detect_catapult(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])',
        content
    )
    
    # Fix 8: detect_scallop - missing highs definition
    content = re.sub(
        r'def detect_scallop\(ohlcv: dict\):',
        r'def detect_scallop(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])',
        content
    )
    
    # Fix 9: detect_tower_top - missing highs definition
    content = re.sub(
        r'def detect_tower_top\(ohlcv: dict\):',
        r'def detect_tower_top(ohlcv: dict):\n        highs = np.array(ohlcv[\'high\'])',
        content
    )
    
    # Fix 10: detect_triple_bottom - duplicate lows definition
    content = re.sub(
        r'highs = np\.array\(ohlcv\[\'high\'\]\)\n        lows = np\.array\(ohlcv\[\'low\'\]\)',
        r'highs = np.array(ohlcv[\'high\'])',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Array definition mismatches fixed!")

if __name__ == "__main__":
    fix_array_mismatches() 