#!/usr/bin/env python3
"""
Script to fix remaining candlestick patterns that still use tuple returns
"""

import re

def fix_candlestick_patterns():
    file_path = "src/core/use_cases/market_analysis/detect_patterns_engine/candlestick_patterns.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix bearish abandoned baby
    content = re.sub(
        r'if is_prior_uptrend and first_is_bullish and gap1_bearish and third_is_bearish and gap2_bearish:\s*confidence = 0\.85\s*if \(c1-o1\) > avg_body_size and \(o3-c3\) > avg_body_size: confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), "bearish_abandoned_baby"',
        '''if is_prior_uptrend and first_is_bullish and gap1_bearish and third_is_bearish and gap2_bearish:
            confidence = 0.85
            if (c1-o1) > avg_body_size and (o3-c3) > avg_body_size: confidence += 0.1
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "first_candle_open": float(opens[-3]),
                "first_candle_close": float(closes[-3]),
                "doji_open": float(opens[-2]),
                "doji_close": float(closes[-2]),
                "third_candle_open": float(opens[-1]),
                "third_candle_close": float(closes[-1]),
                "gap1_size": float(h2 - l1),
                "gap2_size": float(h2 - l3)
            }

            return {
                "pattern_name": "bearish_abandoned_baby",
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 3,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix rising three methods
    content = re.sub(
        r'if is_uptrend and is_c1_long_bullish and are_middle_small_and_in_range and middle_trend_down and is_c5_long_bullish_breakout:\s*confidence = 0\.75\s*if \(c5-o5\) > 1\.2 \* \(c1-o1\) and \(c1-o1\)>0: confidence \+= 0\.1\s*# Check if middle candles don\'t dip below open of C1\s*if np\.min\(\[l2,l3,l4\]\) > o1 : confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if is_uptrend and is_c1_long_bullish and are_middle_small_and_in_range and middle_trend_down and is_c5_long_bullish_breakout:
            confidence = 0.75
            if (c5-o5) > 1.2 * (c1-o1) and (c1-o1)>0: confidence += 0.1
            # Check if middle candles don't dip below open of C1
            if np.min([l2,l3,l4]) > o1 : confidence += 0.1
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "first_candle_open": float(opens[-5]),
                "first_candle_close": float(closes[-5]),
                "fifth_candle_open": float(opens[-1]),
                "fifth_candle_close": float(closes[-1]),
                "breakout_level": float(h1),
                "consolidation_range": float(h1 - l1)
            }

            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 5,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix falling three methods
    content = re.sub(
        r'if is_downtrend and is_c1_long_bearish and are_middle_small_and_in_range and middle_trend_up and is_c5_long_bearish_breakdown:\s*confidence = 0\.75\s*if \(o5-c5\) > 1\.2 \* \(o1-c1\) and \(o1-c1\)>0: confidence \+= 0\.1\s*if np\.max\(\[h2,h3,h4\]\) < o1 : confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if is_downtrend and is_c1_long_bearish and are_middle_small_and_in_range and middle_trend_up and is_c5_long_bearish_breakdown:
            confidence = 0.75
            if (o5-c5) > 1.2 * (o1-c1) and (o1-c1)>0: confidence += 0.1
            if np.max([h2,h3,h4]) < o1 : confidence += 0.1
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "first_candle_open": float(opens[-5]),
                "first_candle_close": float(closes[-5]),
                "fifth_candle_open": float(opens[-1]),
                "fifth_candle_close": float(closes[-1]),
                "breakdown_level": float(l1),
                "consolidation_range": float(h1 - l1)
            }

            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 5,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix hikkake patterns
    content = re.sub(
        r'if c_cur > o_cur and \(c_cur - o_cur\) > 0\.5 \* abs\(o_m2 - c_m2\) if abs\(o_m2-c_m2\)>0 else True :\s*confidence = 0\.7\s*if h_m1 < h_m2: confidence \+= 0\.1 # C-1 high also below inside bar high\s*return True, round\(min\(confidence,1\.0\),2\), "bullish_hikkake"',
        '''if c_cur > o_cur and (c_cur - o_cur) > 0.5 * abs(o_m2 - c_m2) if abs(o_m2-c_m2)>0 else True :
                confidence = 0.7
                if h_m1 < h_m2: confidence += 0.1 # C-1 high also below inside bar high
                
                # Prepare key levels
                key_levels = {
                    "latest_close": float(closes[-1]),
                    "pattern_high": float(highs[-1]),
                    "pattern_low": float(lows[-1]),
                    "inside_bar_high": float(h_m2),
                    "inside_bar_low": float(l_m2),
                    "false_break_low": float(l_m1),
                    "confirmation_close": float(c_cur)
                }

                return {
                    "pattern_name": "bullish_hikkake",
                    "confidence": round(min(confidence,1.0),2),
                    "start_index": len(opens) - 3,  # Relative to the segment
                    "end_index": len(opens) - 1,    # Relative to the segment
                    "key_levels": key_levels
                }''',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'if c_cur < o_cur and \(o_cur - c_cur\) > 0\.5 \* abs\(o_m2 - c_m2\) if abs\(o_m2-c_m2\)>0 else True:\s*confidence = 0\.7\s*if l_m1 > l_m2: confidence \+= 0\.1 # C-1 low also above inside bar low\s*return True, round\(min\(confidence,1\.0\),2\), "bearish_hikkake"',
        '''if c_cur < o_cur and (o_cur - c_cur) > 0.5 * abs(o_m2 - c_m2) if abs(o_m2-c_m2)>0 else True:
                confidence = 0.7
                if l_m1 > l_m2: confidence += 0.1 # C-1 low also above inside bar low
                
                # Prepare key levels
                key_levels = {
                    "latest_close": float(closes[-1]),
                    "pattern_high": float(highs[-1]),
                    "pattern_low": float(lows[-1]),
                    "inside_bar_high": float(h_m2),
                    "inside_bar_low": float(l_m2),
                    "false_break_high": float(h_m1),
                    "confirmation_close": float(c_cur)
                }

                return {
                    "pattern_name": "bearish_hikkake",
                    "confidence": round(min(confidence,1.0),2),
                    "start_index": len(opens) - 3,  # Relative to the segment
                    "end_index": len(opens) - 1,    # Relative to the segment
                    "key_levels": key_levels
                }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix mat hold patterns
    content = re.sub(
        r'if len\(closes\) >= 7 and closes\[-7\] < closes\[-6\] < o_init :\s*confidence = 0\.8\s*if c_break > h_init \* 1\.01 : confidence \+= 0\.1 # Stronger breakout\s*return True, round\(min\(confidence,1\.0\),2\), "bullish_mat_hold"',
        '''if len(closes) >= 7 and closes[-7] < closes[-6] < o_init :
                    confidence = 0.8
                    if c_break > h_init * 1.01 : confidence += 0.1 # Stronger breakout
                    
                    # Prepare key levels
                    key_levels = {
                        "latest_close": float(closes[-1]),
                        "pattern_high": float(highs[-1]),
                        "pattern_low": float(lows[-1]),
                        "initial_candle_open": float(o_init),
                        "initial_candle_close": float(c_init),
                        "breakout_candle_open": float(o_break),
                        "breakout_candle_close": float(c_break),
                        "gap_size": float(cons_lows[0] - c_init)
                    }

                    return {
                        "pattern_name": "bullish_mat_hold",
                        "confidence": round(min(confidence,1.0),2),
                        "start_index": len(opens) - 5,  # Relative to the segment
                        "end_index": len(opens) - 1,    # Relative to the segment
                        "key_levels": key_levels
                    }''',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'if len\(closes\) >= 7 and closes\[-7\] > closes\[-6\] > o_init :  # Fixed indentation here\s*confidence = 0\.8\s*if c_break < l_init \* 0\.99 : confidence \+= 0\.1\s*return True, round\(min\(confidence,1\.0\),2\), "bearish_mat_hold"',
        '''if len(closes) >= 7 and closes[-7] > closes[-6] > o_init :  # Fixed indentation here
                    confidence = 0.8
                    if c_break < l_init * 0.99 : confidence += 0.1
                    
                    # Prepare key levels
                    key_levels = {
                        "latest_close": float(closes[-1]),
                        "pattern_high": float(highs[-1]),
                        "pattern_low": float(lows[-1]),
                        "initial_candle_open": float(o_init),
                        "initial_candle_close": float(c_init),
                        "breakout_candle_open": float(o_break),
                        "breakout_candle_close": float(c_break),
                        "gap_size": float(c_init - cons_highs[0])
                    }

                    return {
                        "pattern_name": "bearish_mat_hold",
                        "confidence": round(min(confidence,1.0),2),
                        "start_index": len(opens) - 5,  # Relative to the segment
                        "end_index": len(opens) - 1,    # Relative to the segment
                        "key_levels": key_levels
                    }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix spinning top
    content = re.sub(
        r'if small_body and long_upper_shadow and long_lower_shadow and similar_shadows:\s*confidence = 0\.6\s*if body < avg_body_size \* 0\.4: confidence \+= 0\.1\s*if upper_shadow > 1\.5 \* body and lower_shadow > 1\.5 \* body and body > 0: confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if small_body and long_upper_shadow and long_lower_shadow and similar_shadows:
            confidence = 0.6
            if body < avg_body_size * 0.4: confidence += 0.1
            if upper_shadow > 1.5 * body and lower_shadow > 1.5 * body and body > 0: confidence += 0.1
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "pattern_open": float(opens[-1]),
                "pattern_close": float(closes[-1]),
                "body_size": float(body),
                "upper_shadow": float(upper_shadow),
                "lower_shadow": float(lower_shadow),
                "body_ratio": float(body / candle_range) if candle_range > 0 else 0.0
            }

            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 1,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix marubozu
    content = re.sub(
        r'if is_bullish_marubozu or is_bearish_marubozu:\s*confidence = 0\.7\s*# Compare body to average body size\s*avg_body = np\.mean\(np\.abs\(opens - closes\)\) if len\(opens\) > 1 else body\s*if body > avg_body \* 1\.5: confidence \+= 0\.2 # Stronger if body is large\s*if body / candle_range > 0\.98 : confidence \+= 0\.1 # Very little shadow\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if is_bullish_marubozu or is_bearish_marubozu:
            confidence = 0.7
            # Compare body to average body size
            avg_body = np.mean(np.abs(opens - closes)) if len(opens) > 1 else body
            if body > avg_body * 1.5: confidence += 0.2 # Stronger if body is large
            if body / candle_range > 0.98 : confidence += 0.1 # Very little shadow
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "pattern_open": float(opens[-1]),
                "pattern_close": float(closes[-1]),
                "body_size": float(body),
                "body_ratio": float(body / candle_range) if candle_range > 0 else 0.0,
                "shadow_threshold": float(body * 0.05)
            }

            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 1,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix harami patterns
    content = re.sub(
        r'if pattern_type:\s*if body1 > avg_body_size \* 1\.2 : confidence \+= 0\.1 # Larger C1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if pattern_type:
                if body1 > avg_body_size * 1.2 : confidence += 0.1 # Larger C1
                
                # Prepare key levels
                key_levels = {
                    "latest_close": float(closes[-1]),
                    "pattern_high": float(highs[-1]),
                    "pattern_low": float(lows[-1]),
                    "first_candle_open": float(opens[-2]),
                    "first_candle_close": float(closes[-2]),
                    "second_candle_open": float(opens[-1]),
                    "second_candle_close": float(closes[-1]),
                    "first_body_size": float(body1),
                    "second_body_size": float(body2)
                }

                return {
                    "pattern_name": pattern_type,
                    "confidence": round(min(confidence, 1.0), 2),
                    "start_index": len(opens) - 2,  # Relative to the segment
                    "end_index": len(opens) - 1,    # Relative to the segment
                    "key_levels": key_levels
                }''',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'if pattern_type:\s*if body1 > avg_body_size \* 1\.2 : confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if pattern_type:
                if body1 > avg_body_size * 1.2 : confidence += 0.1
                
                # Prepare key levels
                key_levels = {
                    "latest_close": float(closes[-1]),
                    "pattern_high": float(highs[-1]),
                    "pattern_low": float(lows[-1]),
                    "first_candle_open": float(opens[-2]),
                    "first_candle_close": float(closes[-2]),
                    "second_candle_open": float(opens[-1]),
                    "second_candle_close": float(closes[-1]),
                    "first_body_size": float(body1),
                    "second_body_size": float(body2)
                }

                return {
                    "pattern_name": pattern_type,
                    "confidence": round(min(confidence, 1.0), 2),
                    "start_index": len(opens) - 2,  # Relative to the segment
                    "end_index": len(opens) - 1,    # Relative to the segment
                    "key_levels": key_levels
                }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix three black crows
    content = re.sub(
        r'if is_prior_uptrend and progressive_lows and open_in_prior_body and are_long_bodies and all_close_near_lows:\s*confidence = 0\.8\s*# More confidence if bodies are of similar size or growing\s*if body3 >= body2 >= body1 \* 0\.8: confidence \+= 0\.1\s*return True, round\(min\(confidence, 1\.0\), 2\), pattern_type',
        '''if is_prior_uptrend and progressive_lows and open_in_prior_body and are_long_bodies and all_close_near_lows:
            confidence = 0.8
            # More confidence if bodies are of similar size or growing
            if body3 >= body2 >= body1 * 0.8: confidence += 0.1
            
            # Prepare key levels
            key_levels = {
                "latest_close": float(closes[-1]),
                "pattern_high": float(highs[-1]),
                "pattern_low": float(lows[-1]),
                "first_candle_open": float(opens[-3]),
                "first_candle_close": float(closes[-3]),
                "second_candle_open": float(opens[-2]),
                "second_candle_close": float(closes[-2]),
                "third_candle_open": float(opens[-1]),
                "third_candle_close": float(closes[-1]),
                "progressive_lows": [float(c1_c), float(c2_c), float(c3_c)]
            }

            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": len(opens) - 3,  # Relative to the segment
                "end_index": len(opens) - 1,    # Relative to the segment
                "key_levels": key_levels
            }''',
        content,
        flags=re.DOTALL
    )
    
    # Fix the missing lows variable issue in detect_piercing_pattern
    content = re.sub(
        r'lows = np\.array\(ohlcv\[\'low\'\]\)\s*lows = np\.array\(ohlcv\[\'low\'\]\)',
        'lows = np.array(ohlcv[\'low\'])',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all remaining candlestick pattern functions!")

if __name__ == "__main__":
    fix_candlestick_patterns() 