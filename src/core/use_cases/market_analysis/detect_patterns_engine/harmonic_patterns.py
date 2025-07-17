import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional, Any
from .pattern_registry import register_pattern
from common.logger import logger

class PatternValidator:
    """Enhanced pattern validation with market context"""
    
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range for dynamic thresholds"""
        if len(highs) < period + 1:
            return np.mean(highs - lows)
        
        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        return np.mean(tr_values[-period:])
    
    @staticmethod
    def detect_market_regime(closes: np.ndarray, period: int = 50) -> str:
        """Detect if market is trending or ranging"""
        if len(closes) < period:
            return "ranging"
        
        recent_closes = closes[-period:]
        slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        
        y_pred = np.polyval([slope, recent_closes[0]], range(len(recent_closes)))
        ss_res = np.sum((recent_closes - y_pred) ** 2)
        ss_tot = np.sum((recent_closes - np.mean(recent_closes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return "trending" if r_squared > 0.7 else "ranging"
    
    @staticmethod
    def validate_volume_confirmation(volumes: np.ndarray, pattern_indices: List[int]) -> float:
        """Validate pattern with volume analysis"""
        if len(volumes) == 0 or len(pattern_indices) < 2:
            return 0.5
        
        try:
            avg_volume = np.mean(volumes[-20:])
            pattern_volume = np.mean([volumes[i] for i in pattern_indices if i < len(volumes)])
            if pattern_volume > avg_volume * 1.2:
                return 0.8
            elif pattern_volume > avg_volume * 0.8:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5
    
    @staticmethod
    def check_pattern_maturity(ohlcv: dict, d_point_index: int, pattern_type: str) -> Tuple[bool, float]:
        """Check if pattern is mature and showing reversal signs"""
        if d_point_index >= len(ohlcv['close']) - 1:
            return False, 0.0
        
        closes = np.array(ohlcv['close'])
        d_price = closes[d_point_index]
        confirmation_period = min(3, len(closes) - d_point_index - 1)
        if confirmation_period < 1:
            return False, 0.0
        
        post_d_closes = closes[d_point_index + 1:d_point_index + 1 + confirmation_period]
        if pattern_type == 'bullish':
            reversal_strength = np.mean(post_d_closes > d_price)
        else:
            reversal_strength = np.mean(post_d_closes < d_price)
        
        is_mature = reversal_strength > 0.5
        return is_mature, reversal_strength

def find_significant_swings(ohlcv: dict, atr_multiplier: float = 3.0) -> List[Tuple[int, float, str]]:
    """
    Finds significant swing points (peaks and troughs) using ATR-based threshold.
    Increased default atr_multiplier for harmonics to detect only major swings.
    """
    highs = np.array(ohlcv['high'])
    lows = np.array(ohlcv['low'])
    closes = np.array(ohlcv['close'])
    
    if len(highs) < 15:
        atr = float(np.mean(highs - lows))
    else:
        atr = float(PatternValidator.calculate_atr(highs, lows, closes))
    
    threshold = atr * atr_multiplier
    swings = []
    last_swing_idx = 0
    last_swing_price = highs[0] if highs[0] > lows[0] else lows[0]
    trend = 1 if highs[0] > lows[0] else -1  # 1 for up, -1 for down

    for i in range(1, len(highs)):
        if trend == 1:  # Uptrend
            if highs[i] > last_swing_price:
                last_swing_price = highs[i]
                last_swing_idx = i
            elif last_swing_price - lows[i] > threshold:
                swings.append((last_swing_idx, last_swing_price, 'high'))
                trend = -1
                last_swing_price = lows[i]
                last_swing_idx = i
        else:  # Downtrend
            if lows[i] < last_swing_price:
                last_swing_price = lows[i]
                last_swing_idx = i
            elif highs[i] - last_swing_price > threshold:
                swings.append((last_swing_idx, last_swing_price, 'low'))
                trend = 1
                last_swing_price = highs[i]
                last_swing_idx = i

    if swings and swings[-1][0] != last_swing_idx:
        swing_type = 'high' if trend == 1 else 'low'
        swings.append((last_swing_idx, last_swing_price, swing_type))

    return swings

def calculate_ratio_confidence(actual_ratio: float, ideal_ratio: float, tolerance: float = 0.05) -> float:
    """Calculate confidence based on ratio deviation"""
    if abs(actual_ratio - ideal_ratio) > tolerance:
        return 0.0
    deviation = abs(actual_ratio - ideal_ratio) / ideal_ratio
    normalized_deviation = deviation / tolerance
    confidence = (1 - normalized_deviation) ** 2
    return max(0.0, min(1.0, confidence))

def check_ratio_range(actual_ratio: float, min_ratio: float, max_ratio: float) -> Tuple[bool, float]:
    """Check if ratio is within range and calculate confidence"""
    if not (min_ratio <= actual_ratio <= max_ratio):
        return False, 0.0
    range_size = max_ratio - min_ratio
    distance_from_center = abs(actual_ratio - (min_ratio + max_ratio) / 2)
    normalized_distance = distance_from_center / (range_size / 2)
    confidence = 1.0 - (normalized_distance * 0.3)
    return True, max(0.0, min(1.0, confidence))

def calculate_enhanced_confidence(base_confidence: float, market_regime: str, volume_score: float, maturity_score: float) -> float:
    """Calculate final confidence score with market context"""
    regime_multiplier = 1.2 if market_regime == "ranging" else 0.9
    volume_weight = 0.2
    maturity_weight = 0.3
    final_confidence = (
        base_confidence * 0.5 +
        volume_score * volume_weight +
        maturity_score * maturity_weight
    ) * regime_multiplier
    return min(1.0, max(0.0, final_confidence))

def get_pattern_point_labels(pattern_name: str, number_of_points: int) -> List[str]:
    """Get correct point labels for each pattern type"""
    if pattern_name in ["gartley", "bat", "butterfly", "crab", "cypher", "shark"]:
        return ["X", "A", "B", "C", "D"]
    elif pattern_name == "abcd":
        return ["A", "B", "C", "D"]
    elif pattern_name == "three_drives":
        return ["X", "A1", "B1", "A2", "B2", "D"]
    else:
        # Generic fallback
        return [f"P{i}" for i in range(number_of_points)]

pattern_configs = {
    "gartley": {
        "name": "gartley",
        "number_of_points": 5,
        "ratios": [
            {"name": "AB_XA", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 0.618, "tolerance": 0.05, "weight": 0.3},
            {"name": "BC_AB", "calculate": lambda p: abs(p[3] - p[2]) / abs(p[2] - p[1]) if abs(p[2] - p[1]) != 0 else 0, "ideal": 0.382, "tolerance": 0.05, "weight": 0.2},
            {"name": "AD_XA", "calculate": lambda p: abs(p[4] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 0.786, "tolerance": 0.05, "weight": 0.5},
        ]
    },
    "bat": {
        "name": "bat",
        "number_of_points": 5,
        "ratios": [
            {"name": "AB_XA", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 0.382, "max": 0.500, "weight": 0.3},
            {"name": "BC_AB", "calculate": lambda p: abs(p[3] - p[2]) / abs(p[2] - p[1]) if abs(p[2] - p[1]) != 0 else 0, "min": 0.382, "max": 0.886, "weight": 0.2},
            {"name": "XD_XA", "calculate": lambda p: abs(p[4] - p[0]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 0.886, "tolerance": 0.05, "weight": 0.5},
        ]
    },
    "butterfly": {
        "name": "butterfly",
        "number_of_points": 5,
        "ratios": [
            {"name": "AB_XA", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 0.786, "tolerance": 0.05, "weight": 0.4},
            {"name": "XD_XA", "calculate": lambda p: abs(p[4] - p[0]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 1.272, "max": 1.618, "weight": 0.6},
        ]
    },
    "crab": {
        "name": "crab",
        "number_of_points": 5,
        "ratios": [
            {"name": "AB_XA", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 0.382, "max": 0.618, "weight": 0.3},
            {"name": "BC_AB", "calculate": lambda p: abs(p[3] - p[2]) / abs(p[2] - p[1]) if abs(p[2] - p[1]) != 0 else 0, "min": 0.382, "max": 0.886, "weight": 0.3},
            {"name": "XD_XA", "calculate": lambda p: abs(p[4] - p[0]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 1.618, "tolerance": 0.03, "weight": 0.4},
        ]
    },
    "cypher": {
        "name": "cypher",
        "number_of_points": 5,
        "ratios": [
            {"name": "AB_XA", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 0.382, "max": 0.618, "weight": 0.3},
            {"name": "XC_XA", "calculate": lambda p: abs(p[3] - p[0]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 1.272, "max": 1.414, "weight": 0.3},
            {"name": "CD_XC", "calculate": lambda p: abs(p[4] - p[3]) / abs(p[3] - p[0]) if abs(p[3] - p[0]) != 0 else 0, "ideal": 0.786, "tolerance": 0.05, "weight": 0.4},
        ]
    },
    "shark": {
        "name": "shark",
        "number_of_points": 5,
        "ratios": [
            {"name": "XC_XA", "calculate": lambda p: abs(p[3] - p[0]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 1.13, "max": 1.618, "weight": 0.5},
            {"name": "CD_XC", "calculate": lambda p: abs(p[4] - p[3]) / abs(p[3] - p[0]) if abs(p[3] - p[0]) != 0 else 0, "min": 0.886, "max": 1.13, "weight": 0.5},
        ]
    },
    "abcd": {
        "name": "abcd",
        "number_of_points": 4,
        "ratios": [
            {"name": "AB_CD", "calculate": lambda p: abs(p[3] - p[2]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 1.0, "tolerance": 0.05, "weight": 0.5},
            {"name": "BC_AB", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 0.382, "max": 0.786, "weight": 0.5},
        ]
    },
    "three_drives": {
        "name": "three_drives",
        "number_of_points": 6,
        "ratios": [
            {"name": "retrace1", "calculate": lambda p: abs(p[2] - p[1]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "min": 0.618, "max": 0.786, "weight": 0.25},
            {"name": "retrace2", "calculate": lambda p: abs(p[4] - p[3]) / abs(p[3] - p[2]) if abs(p[3] - p[2]) != 0 else 0, "min": 0.618, "max": 0.786, "weight": 0.25},
            {"name": "drive_ratio1", "calculate": lambda p: abs(p[3] - p[2]) / abs(p[1] - p[0]) if abs(p[1] - p[0]) != 0 else 0, "ideal": 1.0, "tolerance": 0.1, "weight": 0.25},
            {"name": "drive_ratio2", "calculate": lambda p: abs(p[5] - p[4]) / abs(p[3] - p[2]) if abs(p[3] - p[2]) != 0 else 0, "ideal": 1.0, "tolerance": 0.1, "weight": 0.25},
        ],
        "direction_check": lambda p, pattern_type: (pattern_type == 'bullish' and p[1] < p[3] < p[5]) or (pattern_type == 'bearish' and p[1] > p[3] > p[5])
    }
}

def validate_pattern(swings: List[Tuple[int, float, str]], pattern_config: Dict[str, Any], ohlcv: dict) -> List[Dict[str, Any]]:
    """Validate patterns based on configuration"""
    number_of_points = pattern_config["number_of_points"]
    if len(swings) < number_of_points:
        return []
    
    detected_patterns = []
    timestamps = ohlcv.get('timestamp', None)
    
    MIN_PATTERN_LENGTH = 150  # Minimum number of candles a harmonic pattern must span (greater than chart patterns)
    MIN_POINT_GAP = 30  # Minimum gap between consecutive pattern points
    
    for i in range(len(swings) - number_of_points + 1):
        points = swings[i:i + number_of_points]
        p_types = [p[2] for p in points]
        p_prices = [p[1] for p in points]
        p_indices = [p[0] for p in points]
        
        # Enforce minimum pattern length
        if p_indices[-1] - p_indices[0] < MIN_PATTERN_LENGTH:
            continue
        # Enforce minimum gap between consecutive points
        if any(p_indices[j+1] - p_indices[j] < MIN_POINT_GAP for j in range(len(p_indices)-1)):
            continue
        
        if p_types[0] == 'low':
            pattern_type = 'bullish'
        else:
            pattern_type = 'bearish'
        
        expected_sequence = ['low', 'high'] * (number_of_points // 2 + 1) if pattern_type == 'bullish' else ['high', 'low'] * (number_of_points // 2 + 1)
        expected_sequence = expected_sequence[:number_of_points]
        
        if p_types == expected_sequence:
            if "direction_check" in pattern_config and not pattern_config["direction_check"](p_prices, pattern_type):
                continue
            
            ratios_conf = {}
            for ratio_config in pattern_config["ratios"]:
                try:
                    actual_ratio = ratio_config["calculate"](p_prices)
                    if "ideal" in ratio_config:
                        conf = calculate_ratio_confidence(actual_ratio, ratio_config["ideal"], ratio_config["tolerance"])
                    else:
                        is_valid, conf = check_ratio_range(actual_ratio, ratio_config["min"], ratio_config["max"])
                        if not is_valid:
                            conf = 0.0
                    ratios_conf[ratio_config["name"]] = conf
                except ZeroDivisionError:
                    ratios_conf[ratio_config["name"]] = 0.0
            
            if all(conf > 0 for conf in ratios_conf.values()):
                weighted_conf = sum(ratios_conf[name] * ratio_config["weight"] for name, ratio_config in zip(ratios_conf, pattern_config["ratios"]))
                closes = np.array(ohlcv['close'])
                market_regime = PatternValidator.detect_market_regime(closes)
                volumes = np.array(ohlcv.get('volume', []))
                volume_score = PatternValidator.validate_volume_confirmation(volumes, p_indices)
                d_index = p_indices[-1]
                is_mature, maturity_score = PatternValidator.check_pattern_maturity(ohlcv, d_index, pattern_type)
                overall_confidence = calculate_enhanced_confidence(weighted_conf, market_regime, volume_score, maturity_score)
                targets = get_pattern_targets_and_stops(pattern_config["name"], p_prices, pattern_type)
                
                # Get correct point labels for this pattern
                point_labels = get_pattern_point_labels(pattern_config["name"], number_of_points)
                
                # Add index and timestamp for each point
                points_dict = {}
                for j, (idx, price, label) in enumerate(zip(p_indices, p_prices, point_labels)):
                    point_info = {"index": idx, "price": price}
                    if timestamps is not None and idx < len(timestamps):
                        point_info["timestamp"] = timestamps[idx]
                    points_dict[label] = point_info
                
                # Add start_time and end_time
                start_time = timestamps[p_indices[0]] if timestamps is not None and p_indices[0] < len(timestamps) else None
                end_time = timestamps[p_indices[-1]] if timestamps is not None and p_indices[-1] < len(timestamps) else None
                
                key_levels = {
                    "points": points_dict,
                    "targets": targets,
                    "ratios": {name: ratio_config["calculate"](p_prices) for name, ratio_config in zip(ratios_conf, pattern_config["ratios"])},
                    "market_regime": market_regime,
                    "volume_score": volume_score,
                    "maturity_score": maturity_score,
                    "is_mature": is_mature,
                    "latest_close": float(closes[-1]),
                    "pattern_high": float(max(p_prices)),
                    "pattern_low": float(min(p_prices))
                }
                
                detected_patterns.append({
                    "pattern_name": f"{pattern_config['name']}_{pattern_type}",
                    "confidence": round(overall_confidence, 2),
                    "start_index": p_indices[0],
                    "end_index": p_indices[-1],
                    "start_time": start_time,
                    "end_time": end_time,
                    "key_levels": key_levels
                })
    
    return detected_patterns

def get_pattern_targets_and_stops(pattern_name: str, points: List[float], pattern_direction: str) -> Dict[str, float]:
    """Calculate target and stop levels for patterns"""
    if pattern_name in ["gartley", "bat", "butterfly", "crab", "cypher", "shark"]:
        X, A, B, C, D = points
        if pattern_direction == 'bullish':
            stop_loss = D * 0.98
            target_1 = D + (A - D) * 0.382
            target_2 = D + (A - D) * 0.618
            target_3 = A
        else:
            stop_loss = D * 1.02
            target_1 = D - (D - A) * 0.382
            target_2 = D - (D - A) * 0.618
            target_3 = A
        return {'stop_loss': stop_loss, 'target_1': target_1, 'target_2': target_2, 'target_3': target_3}
    elif pattern_name == "abcd":
        A, B, C, D = points
        if pattern_direction == 'bullish':
            stop_loss = D * 0.98
            target_1 = D + (B - C) * 0.382
            target_2 = D + (B - C) * 0.618
            target_3 = B
        else:
            stop_loss = D * 1.02
            target_1 = D - (C - B) * 0.382
            target_2 = D - (C - B) * 0.618
            target_3 = B
        return {'stop_loss': stop_loss, 'target_1': target_1, 'target_2': target_2, 'target_3': target_3}
    elif pattern_name == "three_drives":
        X, A1, B1, A2, B2, D = points
        if pattern_direction == 'bullish':
            stop_loss = D * 0.98
            target_1 = D + (B2 - A2) * 0.382
            target_2 = D + (B2 - A2) * 0.618
            target_3 = B2
        else:
            stop_loss = D * 1.02
            target_1 = D - (A2 - B2) * 0.382
            target_2 = D - (A2 - B2) * 0.618
            target_3 = B2
        return {'stop_loss': stop_loss, 'target_1': target_1, 'target_2': target_2, 'target_3': target_3}
    return {}

@register_pattern("gartley", "harmonic", types=["gartley_bullish", "gartley_bearish"])
async def detect_gartley(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["gartley"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("bat", "harmonic", types=["bat_bullish", "bat_bearish"])
async def detect_bat(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["bat"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("butterfly", "harmonic", types=["butterfly_bullish", "butterfly_bearish"])
async def detect_butterfly(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["butterfly"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("crab", "harmonic", types=["crab_bullish", "crab_bearish"])
async def detect_crab(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["crab"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("cypher", "harmonic", types=["cypher_bullish", "cypher_bearish"])
async def detect_cypher(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["cypher"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("shark", "harmonic", types=["shark_bullish", "shark_bearish"])
async def detect_shark(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["shark"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("abcd", "harmonic", types=["abcd_bullish", "abcd_bearish"])
async def detect_abcd(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["abcd"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None

@register_pattern("three_drives", "harmonic", types=["three_drives_bullish", "three_drives_bearish"])
async def detect_three_drives(ohlcv: dict) -> Optional[List[Dict[str, Any]]]:
    config = pattern_configs["three_drives"]
    swings = find_significant_swings(ohlcv, atr_multiplier=1.5)
    detected_patterns = validate_pattern(swings, config, ohlcv)
    return detected_patterns if detected_patterns else None