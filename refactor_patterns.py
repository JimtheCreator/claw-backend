#!/usr/bin/env python3
"""
Pattern Refactoring Script
This script helps refactor pattern detection functions from the old tuple format to the new standardized dictionary format.
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

def refactor_pattern_function(file_path: str, pattern_name: str, category: str = "candlestick") -> bool:
    """
    Refactor a specific pattern function in a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the pattern function - handle both _detect_ and detect_ prefixes, and different signatures
        pattern_regex = rf'@register_pattern\("{pattern_name}",[^)]+\)\s*\nasync def (detect_|_detect_){pattern_name}\(ohlcv: dict, context\)( -> Tuple\[bool, float, str\])?:'
        
        if not re.search(pattern_regex, content):
            print(f"Pattern {pattern_name} not found in {file_path}")
            return False
        
        # Extract the function body
        function_start = re.search(pattern_regex, content)
        if not function_start:
            return False
        
        # Find the end of the function (look for the next function or end of file)
        start_pos = function_start.start()
        remaining_content = content[start_pos:]
        
        # Find the next function or end of file
        next_function = re.search(r'\n@register_pattern', remaining_content[1:])
        if next_function:
            end_pos = start_pos + next_function.start() + 1
        else:
            end_pos = len(content)
        
        old_function = content[start_pos:end_pos]
        
        # Create new function signature
        prefix = "detect_" if "detect_" in old_function else "_detect_"
        new_signature = f'''@register_pattern("{pattern_name}", "{category}", types=["{pattern_name}"])
async def {prefix}{pattern_name}(ohlcv: dict) -> Optional[Dict[str, Any]]:'''
        
        # Replace function signature
        new_function = re.sub(
            rf'async def (detect_|_detect_){pattern_name}\(ohlcv: dict, context\)( -> Tuple\[bool, float, str\])?:',
            f'async def {prefix}{pattern_name}(ohlcv: dict) -> Optional[Dict[str, Any]]:',
            old_function
        )
        
        # Replace return False, 0.0, "" with return None
        new_function = re.sub(r'return False, 0\.0, ""\s*#.*', 'return None', new_function)
        new_function = re.sub(r'return False, 0\.0, ""', 'return None', new_function)
        
        # Replace return True, round(confidence, 2), pattern_type with new format
        new_function = re.sub(
            r'return True, round\(confidence, 2\), pattern_type',
            '''        # Prepare key levels
        key_levels = {
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }

        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": len(opens) - 1,  # Relative to the segment
            "end_index": len(opens) - 1,    # Relative to the segment
            "key_levels": key_levels
        }''',
            new_function
        )
        
        # Add highs and lows arrays if not present
        if 'highs = np.array(ohlcv[\'high\'])' not in new_function:
            new_function = new_function.replace(
                'closes = np.array(ohlcv[\'close\'])',
                'closes = np.array(ohlcv[\'close\'])\n        highs = np.array(ohlcv[\'high\'])\n        lows = np.array(ohlcv[\'low\'])'
            )
        
        # Replace the old function with the new one
        new_content = content[:start_pos] + new_function + content[end_pos:]
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Successfully refactored {pattern_name} in {file_path}")
        return True
        
    except Exception as e:
        print(f"Error refactoring {pattern_name}: {str(e)}")
        return False

def main():
    """
    Main function to refactor all patterns.
    """
    # Remaining candlestick patterns that weren't refactored
    remaining_candlestick_patterns = [
        "tweezers_top",
        "tweezers_bottom",
        "abandoned_baby",
        "rising_three_methods",
        "falling_three_methods",
        "hikkake",
        "mat_hold",
        "spinning_top",
        "marubozu",
        "harami",
        "three_black_crows"
    ]
    
    # Harmonic patterns that need special handling
    harmonic_patterns = [
        "shark"
    ]
    
    candlestick_file = "src/core/use_cases/market_analysis/detect_patterns_engine/candlestick_patterns.py"
    harmonic_file = "src/core/use_cases/market_analysis/detect_patterns_engine/harmonic_patterns.py"
    
    print("Starting pattern refactoring...")
    
    # Refactor remaining candlestick patterns
    print("\n=== Refactoring remaining candlestick patterns ===")
    for pattern in remaining_candlestick_patterns:
        print(f"Refactoring {pattern}...")
        success = refactor_pattern_function(candlestick_file, pattern, "candlestick")
        if success:
            print(f"✓ {pattern} refactored successfully")
        else:
            print(f"✗ Failed to refactor {pattern}")
    
    # Refactor harmonic patterns
    print("\n=== Refactoring harmonic patterns ===")
    for pattern in harmonic_patterns:
        print(f"Refactoring {pattern}...")
        success = refactor_pattern_function(harmonic_file, pattern, "harmonic")
        if success:
            print(f"✓ {pattern} refactored successfully")
        else:
            print(f"✗ Failed to refactor {pattern}")
    
    print("\nPattern refactoring completed!")

if __name__ == "__main__":
    main() 