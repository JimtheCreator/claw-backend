import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy import stats
from typing import Dict, List, Any, Optional
from common.logger import logger

class TrendlineEngine:
    """
    Enhanced trendline detection with FIXED angle validation using proper price/time scaling
    """
    def __init__(self, interval: str = "1h", min_window: Optional[int] = None, max_window: Optional[int] = None):
        self.interval = interval
        interval_defaults = {
            "1m": (30, 300), "5m": (50, 300), "15m": (80, 400), "30m": (100, 400),
            "1h": (120, 600), "2h": (180, 800), "4h": (200, 900), "6h": (250, 1000),
            "1d": (300, 1200), "3d": (400, 1400), "1w": (500, 1500), "1M": (600, 1800)
        }
        self.min_window, self.max_window = interval_defaults.get(interval, (50, 300))
        if min_window is not None:
            self.min_window = min_window
        if max_window is not None:
            self.max_window = max_window
        
        # FIXED: Timeframe-specific angle limits (more realistic)
        self.angle_limits = {
            "1m": 85, "5m": 80, "15m": 75, "30m": 70, 
            "1h": 65, "2h": 60, "4h": 55, "6h": 50,
            "1d": 45, "3d": 40, "1w": 35, "1M": 30
        }
        self.max_angle_degrees = self.angle_limits.get(interval, 65)
        
        self.min_r_squared = 0.7
        self.min_time_distance = 5
        self.recency_decay = 0.95

    def _calculate_price_time_scaling(self, df: pd.DataFrame, time_span: int) -> float:
        """
        CRITICAL FIX: Calculate proper price/time scaling factor
        This ensures angle calculation reflects visual chart appearance
        """
        price_range = df['high'].max() - df['low'].min()
        
        # Get time interval in minutes for scaling
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30, 
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
        }
        
        time_unit = interval_minutes.get(self.interval, 60)
        time_range_minutes = time_span * time_unit
        
        # Price per minute vs time per bar scaling
        # This makes angles visually meaningful
        price_per_minute = price_range / time_range_minutes
        scaling_factor = price_per_minute * time_unit
        
        logger.debug(f"Price range: {price_range:.4f}, Time span: {time_span} bars, "
                    f"Scaling factor: {scaling_factor:.6f}")
        
        return scaling_factor

    def _calculate_angle_degrees(self, slope: float, scaling_factor: float) -> float:
        """
        FIXED: Proper angle calculation with price/time scaling
        """
        # Apply scaling to convert slope to visual angle
        scaled_slope = slope / scaling_factor
        angle_radians = np.arctan(scaled_slope)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

    def _validate_trendline_angle(self, slope: float, scaling_factor: float, 
                                 time_span: int, line_type: str) -> tuple[bool, float, str]:
        """
        CRITICAL: Enhanced angle validation with context-aware limits
        """
        angle = self._calculate_angle_degrees(slope, scaling_factor)
        
        # Adjust limits based on line type and time span
        base_limit = self.max_angle_degrees
        
        # Support lines can be steeper in uptrends
        if line_type == "support" and slope > 0:
            adjusted_limit = base_limit * 1.2  # 20% more tolerance for rising support
        # Resistance lines can be steeper in downtrends  
        elif line_type == "resistance" and slope < 0:
            adjusted_limit = base_limit * 1.2  # 20% more tolerance for falling resistance
        else:
            adjusted_limit = base_limit
        
        # Shorter timeframes get more tolerance for steep angles
        if time_span < 20:
            adjusted_limit *= 1.3
        elif time_span < 50:
            adjusted_limit *= 1.1
        
        is_valid = abs(angle) <= adjusted_limit
        
        reason = ""
        if not is_valid:
            if abs(angle) > adjusted_limit:
                reason = f"Angle too steep: {angle:.1f}° > {adjusted_limit:.1f}°"
        
        logger.debug(f"{line_type} line angle: {angle:.2f}°, limit: {adjusted_limit:.1f}°, valid: {is_valid}")
        
        return is_valid, angle, reason

    def _fit_trendlines(self, df: pd.DataFrame, swings: List[Dict[str, Any]], 
                       line_type: str, avg_atr: float) -> List[Dict[str, Any]]:
        lines = []
        n = len(swings)
        if n < 2:
            return lines

        price_std = np.std(df['close'].to_numpy(dtype=float))
        avg_volume = df['volume'].mean()
        current_idx = len(df) - 1
        
        # FIXED: Calculate scaling factor once for the entire dataset
        total_time_span = len(df) - 1
        scaling_factor = self._calculate_price_time_scaling(df, total_time_span)
        
        # Dynamic minimum touches based on timeframe
        timeframe_touches = {"1m": 2, "5m": 2, "15m": 3, "30m": 3, "1h": 3, "4h": 4, "1d": 4}
        min_touches = timeframe_touches.get(self.interval, 3)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                idx1, price1 = swings[i]["idx"], swings[i]["price"]
                idx2, price2 = swings[j]["idx"], swings[j]["price"]
                vol1, vol2 = swings[i]["volume"], swings[j]["volume"]
                
                if idx2 == idx1: 
                    continue

                m = (price2 - price1) / (idx2 - idx1)
                b = price1 - m * idx1
                time_span = idx2 - idx1
                
                # CRITICAL FIX: Enhanced angle validation
                is_valid_angle, angle, angle_reason = self._validate_trendline_angle(
                    m, scaling_factor, time_span, line_type
                )
                
                if not is_valid_angle:
                    logger.debug(f"Rejecting {line_type} line: {angle_reason}")
                    continue
                
                # Enhanced tolerance using ATR
                atr_tolerance = avg_atr * 0.5
                std_tolerance = price_std * 0.05
                tolerance = max(atr_tolerance, std_tolerance)

                # === Penetration Check ===
                penetrated = False
                for k in range(idx1, idx2 + 1):
                    expected = m * k + b
                    if line_type == "support" and df['close'].iloc[k] < expected - tolerance:
                        penetrated = True; break
                    if line_type == "resistance" and df['close'].iloc[k] > expected + tolerance:
                        penetrated = True; break
                if penetrated:
                    continue

                # === Enhanced Touch Counting with Volume Weighting ===
                touches, touch_indices, deviations, touch_volumes = 0, [], [], []
                touch_points_x, touch_points_y = [], []
                
                for k in range(idx1, idx2 + 1):
                    expected = m * k + b
                    actual = df['low'].iloc[k] if line_type == "support" else df['high'].iloc[k]
                    if abs(actual - expected) < tolerance:
                        touches += 1
                        touch_indices.append(k)
                        deviations.append(abs(actual - expected))
                        touch_volumes.append(df['volume'].iloc[k])
                        touch_points_x.append(k)
                        touch_points_y.append(actual)
                
                if touches < min_touches:
                    continue

                # Calculate R-squared for line fit quality
                r_squared = self._calculate_r_squared(touch_points_x, touch_points_y, m, b)
                if r_squared < self.min_r_squared:
                    continue

                # === Extended Touches with Recency Weighting ===
                extended_touches = 0
                extended_weighted_touches = 0
                
                for k in range(idx2 + 1, len(df)):
                    expected = m * k + b
                    actual = df['low'].iloc[k] if line_type == "support" else df['high'].iloc[k]
                    if abs(actual - expected) < tolerance:
                        extended_touches += 1
                        # Apply recency weighting
                        recency_weight = self.recency_decay ** (current_idx - k)
                        extended_weighted_touches += recency_weight

                # === Enhanced Scoring System with Angle Bonus ===
                length = idx2 - idx1
                avg_deviation = np.mean(deviations) if deviations else 0
                
                # Volume weighting for touches
                volume_weight = np.mean(touch_volumes) / avg_volume if avg_volume > 0 else 1.0
                volume_score = min(volume_weight * 5, 10)  # Cap at 10 points
                
                # Recency bonus for recent formation
                recency_bonus = self.recency_decay ** (current_idx - idx2) * 10
                
                # R-squared bonus
                r_squared_bonus = (r_squared - self.min_r_squared) * 20
                
                # FIXED: Angle scoring - prefer moderate angles
                abs_angle = abs(angle)
                if abs_angle <= 30:  # Sweet spot for stable trendlines
                    angle_bonus = 5
                elif abs_angle <= 45:
                    angle_bonus = 2
                else:
                    angle_bonus = 0
                
                score = (
                    (touches * 10) + 
                    (extended_weighted_touches * 15) + 
                    (length * 0.1) + 
                    volume_score + 
                    recency_bonus + 
                    r_squared_bonus +
                    angle_bonus -
                    (avg_deviation / price_std * 50)
                )

                lines.append({
                    "type": line_type, "start_idx": idx1, "end_idx": idx2,
                    "start_price": price1, "end_price": price2, "slope": m, "intercept": b,
                    "touches": touches, "extended_touches": extended_touches, "score": score,
                    "avg_deviation": avg_deviation, "tolerance_value": tolerance,
                    "r_squared": r_squared, "angle_degrees": angle,  # Now properly calculated
                    "volume_weight": volume_weight, "recency_bonus": recency_bonus,
                    "extended_weighted_touches": extended_weighted_touches,
                    "time_span": time_span, "scaling_factor": scaling_factor,  # Added for debugging
                    "breakout_detected": False, "breakout_confidence": 0.0
                })
        
        lines.sort(key=lambda l: l['score'], reverse=True)
        
        # Enhanced decluttering with angle consideration
        unique_lines = []
        max_lines = 3
        slope_similarity_threshold = 0.15  # Slightly more lenient
        price_similarity_threshold = avg_atr * 2
        angle_similarity_threshold = 10  # degrees

        for line in lines:
            if len(unique_lines) >= max_lines:
                break
            
            is_similar = False
            for unique_line in unique_lines:
                slope_diff = abs((line['slope'] - unique_line['slope']) / (unique_line['slope'] + 1e-9))
                angle_diff = abs(line['angle_degrees'] - unique_line['angle_degrees'])
                
                ref_idx = line['start_idx']
                price_at_ref_unique = unique_line['slope'] * ref_idx + unique_line['intercept']
                price_at_ref_current = line['start_price']
                price_diff = abs(price_at_ref_unique - price_at_ref_current)

                # IMPROVED: Consider both slope and angle similarity
                if (slope_diff < slope_similarity_threshold and 
                    price_diff < price_similarity_threshold and
                    angle_diff < angle_similarity_threshold):
                    is_similar = True
                    logger.debug(f"Discarding similar {line_type} line. "
                               f"Angle diff: {angle_diff:.1f}°, Score: {line['score']:.2f} vs {unique_line['score']:.2f}")
                    break
            
            if not is_similar:
                unique_lines.append(line)

        return unique_lines

    # Keep all other methods unchanged...
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for dynamic tolerance"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period, min_periods=1).mean()
        return atr

    def _find_swings(self, df: pd.DataFrame) -> tuple[List[Dict], List[Dict]]:
        order = max(3, int(len(df) * 0.03))
        highs = df['high'].to_numpy(dtype=float)
        lows = df['low'].to_numpy(dtype=float)
        volumes = df['volume'].to_numpy(dtype=float)
        
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        
        # Filter swing points with minimum time distance
        high_idx = self._filter_by_time_distance(high_idx)
        low_idx = self._filter_by_time_distance(low_idx)
        
        swing_highs = [{"idx": int(i), "price": float(highs[i]), "volume": float(volumes[i])} for i in high_idx]
        swing_lows = [{"idx": int(i), "price": float(lows[i]), "volume": float(volumes[i])} for i in low_idx]
        
        return swing_highs, swing_lows

    def _filter_by_time_distance(self, indices: np.ndarray) -> np.ndarray:
        """Filter swing points to maintain minimum time distance"""
        if len(indices) <= 1:
            return indices
        
        filtered = [indices[0]]
        for idx in indices[1:]:
            if idx - filtered[-1] >= self.min_time_distance:
                filtered.append(idx)
        
        return np.array(filtered)

    def _calculate_r_squared(self, x_points: List[int], y_points: List[float], slope: float, intercept: float) -> float:
        """Calculate R-squared for line fit quality"""
        if len(x_points) < 2:
            return 0.0
        
        y_pred = [slope * x + intercept for x in x_points]
        y_mean = np.mean(y_points)
        
        ss_res = sum((y_actual - y_pred_val) ** 2 for y_actual, y_pred_val in zip(y_points, y_pred))
        ss_tot = sum((y_actual - y_mean) ** 2 for y_actual in y_points)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)

    def _check_breakouts(self, df: pd.DataFrame, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect definitive breakouts of trendlines"""
        for line in lines:
            breakout_detected = False
            breakout_confidence = 0.0
            
            # Check last 10 bars for breakout
            check_period = min(10, len(df) - line['end_idx'] - 1)
            if check_period <= 0:
                line['breakout_detected'] = breakout_detected
                line['breakout_confidence'] = breakout_confidence
                continue
            
            breakout_bars = 0
            total_bars = 0
            avg_atr = df['atr'].mean()
            
            for k in range(len(df) - check_period, len(df)):
                expected = line['slope'] * k + line['intercept']
                close_price = df['close'].iloc[k]
                
                # Define breakout threshold (more conservative)
                breakout_threshold = avg_atr * 1.5
                
                if line['type'] == "support":
                    if close_price < expected - breakout_threshold:
                        breakout_bars += 1
                elif line['type'] == "resistance":
                    if close_price > expected + breakout_threshold:
                        breakout_bars += 1
                
                total_bars += 1
            
            if total_bars > 0 and breakout_bars >= 2:  # At least 2 bars breaking
                breakout_detected = True
                breakout_confidence = breakout_bars / total_bars
                logger.info(f"Breakout detected for {line['type']} line. Confidence: {breakout_confidence:.2f}")
            
            line['breakout_detected'] = breakout_detected
            line['breakout_confidence'] = breakout_confidence
        
        return lines

    async def detect(self, ohlcv: Dict[str, List]) -> Dict[str, Any]:
        try:
            logger.info(f"[TrendlineEngine] Starting enhanced detection for interval: {self.interval}")
            df = pd.DataFrame(ohlcv)
            if len(df) < self.min_window:
                logger.error(f"[TrendlineEngine] Not enough data for trendline detection (min {self.min_window} bars)")
                raise ValueError(f"Not enough data for trendline detection (min {self.min_window} bars)")
            
            df = df.tail(self.max_window).copy()
            df.reset_index(drop=True, inplace=True)
            
            # Calculate ATR for dynamic tolerance
            df['atr'] = self._calculate_atr(df)
            avg_atr = df['atr'].mean()
            
            logger.info(f"[TrendlineEngine] Dataframe prepared with {len(df)} rows. Average ATR: {avg_atr:.4f}")
            
            swing_highs, swing_lows = self._find_swings(df)
            logger.info(f"[TrendlineEngine] Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows.")

            support_lines = self._fit_trendlines(df, swing_lows, line_type="support", avg_atr=avg_atr)
            resistance_lines = self._fit_trendlines(df, swing_highs, line_type="resistance", avg_atr=avg_atr)
            
            # Check for breakouts
            support_lines = self._check_breakouts(df, support_lines)
            resistance_lines = self._check_breakouts(df, resistance_lines)
            
            logger.info(f"[TrendlineEngine] Found {len(support_lines)} high-significance support lines.")
            logger.info(f"[TrendlineEngine] Found {len(resistance_lines)} high-significance resistance lines.")
            
            trendlines = support_lines + resistance_lines
            for line in trendlines:
                line['start_timestamp'] = str(df['timestamp'].iloc[line['start_idx']])
                # Extend line to the last candle for charting
                line['end_idx'] = len(df) - 1
                line['end_price'] = line['slope'] * line['end_idx'] + line['intercept']
                line['end_timestamp'] = str(df['timestamp'].iloc[line['end_idx']])
            
            logger.info(f"[TrendlineEngine] Enhanced trendlines: {len(trendlines)} total")
            return {
                "trendlines": trendlines,
                "meta": {
                    "interval": self.interval,
                    "window": len(df),
                    "avg_atr": avg_atr,
                    "max_angle_limit": self.max_angle_degrees,
                    "timestamp_range": (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))
                }
            }
        except Exception as e:
            logger.error(f"[TrendlineEngine] Error during detection: {e}", exc_info=True)
            raise