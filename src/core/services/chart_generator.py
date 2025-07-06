import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import numpy as np

class ChartGenerator:
    """
    Generates and saves a financial chart with analysis overlays using Plotly.
    """
    def __init__(self, analysis_data: Dict[str, Any], ohlcv_data: Dict[str, Any]):
        self.analysis = analysis_data
        self.ohlcv = pd.DataFrame(ohlcv_data)
        self.ohlcv['timestamp'] = pd.to_datetime(self.ohlcv['timestamp'])

    def create_chart_image(self) -> bytes:
        """
        Creates a candlestick chart with support/resistance levels, pattern overlays, and demand/supply zones, 
        and returns it as PNG bytes.
        """
        fig = go.Figure(data=[go.Candlestick(
            x=self.ohlcv['timestamp'],
            open=self.ohlcv['open'],
            high=self.ohlcv['high'],
            low=self.ohlcv['low'],
            close=self.ohlcv['close']
        )])

        # Add support and resistance lines from analysis data
        support_levels = self.analysis.get("market_context", {}).get("support_levels", [])
        resistance_levels = self.analysis.get("market_context", {}).get("resistance_levels", [])

        for level in support_levels:
            fig.add_hline(y=level, line_dash="dash", line_color="green", annotation_text="Support", annotation_position="bottom right")

        for level in resistance_levels:
            fig.add_hline(y=level, line_dash="dash", line_color="red", annotation_text="Resistance", annotation_position="top right")

        # Add pattern overlays for all patterns
        patterns = self.analysis.get("patterns", [])
        for pattern in patterns:
            self._add_pattern_overlay(fig, pattern)

        # Add demand and supply zones
        demand_zones = self.analysis.get("market_context", {}).get("demand_zones", [])
        supply_zones = self.analysis.get("market_context", {}).get("supply_zones", [])

        for zone in demand_zones:
            fig.add_hrect(
                y0=zone["bottom"],
                y1=zone["top"],
                fillcolor="green",
                opacity=0.2,
                line_width=0
            )

        for zone in supply_zones:
            fig.add_hrect(
                y0=zone["bottom"],
                y1=zone["top"],
                fillcolor="red",
                opacity=0.2,
                line_width=0
            )

        # Zoom chart to the relevant pattern date range
        if patterns:
            min_start_time = pd.to_datetime(min(p['timestamp_start'] for p in patterns))
            max_end_time = pd.to_datetime(max(p['timestamp_end'] for p in patterns))
            time_range = max_end_time - min_start_time
            padding = time_range * 0.1  # 10% padding
            fig.update_layout(
                xaxis_range=[min_start_time - padding, max_end_time + padding]
            )

        # Update layout with user-specified adjustments
        fig.update_layout(
            title="Market Analysis",
            xaxis_title="",
            yaxis_title="",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            yaxis=dict(side="right")
        )

        # Export to PNG image bytes
        image_bytes = fig.to_image(format="png", scale=2)
        return image_bytes

    def _add_pattern_overlay(self, fig, pattern):
        """
        Add pattern overlay to the chart based on pattern type.
        Routes to specific pattern drawing methods.
        """
        pattern_name = pattern.get('pattern', '').lower()
        pattern_type = pattern.get('exact_pattern_type', '').lower()
        
        # Get pattern data for the specific pattern timeframe
        start_idx = pattern.get('start_idx', 0)
        end_idx = pattern.get('end_idx', len(self.ohlcv) - 1)
        pattern_data = self.ohlcv.iloc[start_idx:end_idx + 1]
        
        # Get key levels from pattern
        key_levels = pattern.get('key_levels', {})
        
        # Get color based on pattern type
        color = self._get_pattern_color(pattern_name)
        
        # Route to specific pattern drawing methods
        if "double_top" in pattern_name:
            self._draw_double_top(fig, pattern_data, key_levels, color, pattern_type)
        elif "double_bottom" in pattern_name:
            self._draw_double_bottom(fig, pattern_data, key_levels, color, pattern_type)
        elif "triangle" in pattern_name:
            self._draw_triangle_improved(fig, pattern_data, key_levels, color, pattern_type)
        elif "channel" in pattern_name:
            self._draw_channel(fig, pattern_data, key_levels, color, pattern_type)
        elif "wedge" in pattern_name:
            self._draw_wedge(fig, pattern_data, key_levels, color, pattern_type)
        elif "head_and_shoulder" in pattern_name:
            self._draw_head_and_shoulders(fig, pattern_data, key_levels, color, pattern_type)
        elif "multiple_peaks" in pattern_name:
            self._draw_multiple_peaks(fig, pattern_data, key_levels, color, pattern_type)
        elif "cup" in pattern_name and "handle" in pattern_name:
            self._draw_cup_and_handle(fig, pattern_data, key_levels, color, pattern_type)
        elif "flag" in pattern_name or "pennant" in pattern_name:
            self._draw_flag_pennant(fig, pattern_data, key_levels, color, pattern_type)
        elif "rectangle" in pattern_name:
            self._draw_rectangle(fig, pattern_data, key_levels, color, pattern_type)
        elif "diamond" in pattern_name:
            self._draw_diamond(fig, pattern_data, key_levels, color, pattern_type)
        elif any(candlestick in pattern_name for candlestick in ["doji", "hammer", "shooting_star", "engulfing", "harami", "evening_star", "morning_star"]):
            self._draw_candlestick_pattern(fig, pattern_data, color, pattern_type)
        else:
            # Default pattern drawing for unrecognized patterns
            self._draw_default_pattern(fig, pattern_data, color, pattern_type)

    def _get_pattern_color(self, pattern_type):
        """Get color based on pattern type."""
        if "bullish" in pattern_type:
            return "lime"
        elif "bearish" in pattern_type:
            return "red"
        elif "top" in pattern_type:
            return "orange"
        elif "bottom" in pattern_type:
            return "green"
        elif "wedge" in pattern_type or "triangle" in pattern_type:
            return "purple"
        elif "channel" in pattern_type:
            return "cyan"
        elif "flag" in pattern_type or "pennant" in pattern_type:
            return "magenta"
        elif "doji" in pattern_type:
            return "yellow"
        elif "cup" in pattern_type:
            return "blue"
        else:
            return "white"
        
    def _draw_double_top(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw double top pattern with proper shape overlay."""
        if len(pattern_data) < 3:
            return
        
        # Find the two highest peaks
        highs = pattern_data['high'].values
        high_indices = np.argsort(highs)[-2:]  # Get indices of 2 highest points
        high_indices = np.sort(high_indices)  # Sort by time
        
        peak1_idx = high_indices[0]
        peak2_idx = high_indices[1]
        
        # Get peak data
        peak1_time = pattern_data.iloc[peak1_idx]['timestamp']
        peak2_time = pattern_data.iloc[peak2_idx]['timestamp']
        peak1_price = pattern_data.iloc[peak1_idx]['high']
        peak2_price = pattern_data.iloc[peak2_idx]['high']
        
        # Find valley between peaks
        valley_data = pattern_data.iloc[peak1_idx:peak2_idx+1]
        valley_idx = valley_data['low'].idxmin()
        valley_time = pattern_data.loc[valley_idx, 'timestamp']
        valley_price = pattern_data.loc[valley_idx, 'low']
        
        # Draw the M-shaped pattern
        fig.add_trace(go.Scatter(
            x=[peak1_time, valley_time, peak2_time],
            y=[peak1_price, valley_price, peak2_price],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=12, color=color, symbol="triangle-up"),
            name=f"Double Top",
            showlegend=False
        ))
        
        # Add resistance line connecting peaks
        fig.add_trace(go.Scatter(
            x=[peak1_time, peak2_time],
            y=[peak1_price, peak2_price],
            mode="lines",
            line=dict(color=color, width=3, dash="dash"),
            showlegend=False
        ))
        
        # Add neckline (support at valley level)
        start_time = pattern_data.iloc[0]['timestamp']
        end_time = pattern_data.iloc[-1]['timestamp']
        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[valley_price, valley_price],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False
        ))
        
        # Add pattern annotation
        self._add_pattern_annotation(fig, pattern_data, color, "Double Top")

    def _draw_double_bottom(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw double bottom pattern with proper shape overlay."""
        if len(pattern_data) < 3:
            return
        
        # Find the two lowest valleys
        lows = pattern_data['low'].values
        low_indices = np.argsort(lows)[:2]  # Get indices of 2 lowest points
        low_indices = np.sort(low_indices)  # Sort by time
        
        valley1_idx = low_indices[0]
        valley2_idx = low_indices[1]
        
        # Get valley data
        valley1_time = pattern_data.iloc[valley1_idx]['timestamp']
        valley2_time = pattern_data.iloc[valley2_idx]['timestamp']
        valley1_price = pattern_data.iloc[valley1_idx]['low']
        valley2_price = pattern_data.iloc[valley2_idx]['low']
        
        # Find peak between valleys
        peak_data = pattern_data.iloc[valley1_idx:valley2_idx+1]
        peak_idx = peak_data['high'].idxmax()
        peak_time = pattern_data.loc[peak_idx, 'timestamp']
        peak_price = pattern_data.loc[peak_idx, 'high']
        
        # Draw the W-shaped pattern
        fig.add_trace(go.Scatter(
            x=[valley1_time, peak_time, valley2_time],
            y=[valley1_price, peak_price, valley2_price],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=12, color=color, symbol="triangle-down"),
            name=f"Double Bottom",
            showlegend=False
        ))
        
        # Add support line connecting valleys
        fig.add_trace(go.Scatter(
            x=[valley1_time, valley2_time],
            y=[valley1_price, valley2_price],
            mode="lines",
            line=dict(color=color, width=3, dash="dash"),
            showlegend=False
        ))
        
        # Add neckline (resistance at peak level)
        start_time = pattern_data.iloc[0]['timestamp']
        end_time = pattern_data.iloc[-1]['timestamp']
        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[peak_price, peak_price],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False
        ))
        
        # Add pattern annotation
        self._add_pattern_annotation(fig, pattern_data, color, "Double Bottom")

    def _draw_triangle_improved(self, fig, pattern_data, key_levels, color, pattern_type):
        """Improved triangle pattern drawing with actual price points."""
        if len(pattern_data) < 4:
            return
        
        times = pattern_data['timestamp'].tolist()
        
        # Calculate trend lines from actual data
        if "ascending" in pattern_type:
            # Ascending triangle: flat resistance, rising support
            resistance_level = pattern_data['high'].max()
            
            # Find support trend line from lows
            lows = pattern_data['low'].values
            x_vals = np.arange(len(lows))
            slope, intercept = np.polyfit(x_vals, lows, 1)
            
            support_start = lows[0]
            support_end = slope * (len(lows) - 1) + intercept
            
            # Draw triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[resistance_level, resistance_level],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[support_start, support_end],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            # Fill triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1], times[0]],
                y=[support_start, resistance_level, resistance_level],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
        
        elif "descending" in pattern_type:
            # Descending triangle: declining resistance, flat support
            support_level = pattern_data['low'].min()
            
            # Find resistance trend line from highs
            highs = pattern_data['high'].values
            x_vals = np.arange(len(highs))
            slope, intercept = np.polyfit(x_vals, highs, 1)
            
            resistance_start = highs[0]
            resistance_end = slope * (len(highs) - 1) + intercept
            
            # Draw triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[resistance_start, resistance_end],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[support_level, support_level],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            # Fill triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1], times[0]],
                y=[support_level, support_level, resistance_start],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
        
        else:
            # Symmetrical triangle: both lines converge
            highs = pattern_data['high'].values
            lows = pattern_data['low'].values
            x_vals = np.arange(len(pattern_data))
            
            # Fit trend lines
            high_slope, high_intercept = np.polyfit(x_vals, highs, 1)
            low_slope, low_intercept = np.polyfit(x_vals, lows, 1)
            
            resistance_start = highs[0]
            resistance_end = high_slope * (len(highs) - 1) + high_intercept
            support_start = lows[0]
            support_end = low_slope * (len(lows) - 1) + low_intercept
            
            # Draw triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[resistance_start, resistance_end],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[support_start, support_end],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False
            ))
            
            # Fill triangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1], times[0]],
                y=[support_start, support_end, resistance_start],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
        
        # Add pattern annotation
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_channel(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw channel patterns with parallel lines."""
        times = pattern_data['timestamp'].tolist()
        
        if len(times) >= 2:
            # Use actual data if key_levels not provided
            if "upper_trendline" in key_levels and "lower_trendline" in key_levels:
                upper_level = key_levels["upper_trendline"]
                lower_level = key_levels["lower_trendline"]
            else:
                # Calculate from data
                upper_level = pattern_data['high'].max()
                lower_level = pattern_data['low'].min()
            
            # Draw parallel lines
            x_vals = [times[0], times[-1]]
            
            # Upper channel line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=[upper_level, upper_level],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{pattern_type} Upper",
                showlegend=False
            ))
            
            # Lower channel line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=[lower_level, lower_level],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{pattern_type} Lower",
                showlegend=False
            ))
            
            # Add shaded area between lines
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=[upper_level, upper_level, lower_level, lower_level],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    

    def _draw_wedge(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw wedge patterns."""
        if "upper_trendline" in key_levels and "lower_trendline" in key_levels:
            start_time = pattern_data['timestamp'].iloc[0]
            end_time = pattern_data['timestamp'].iloc[-1]
            
            upper_start = key_levels["upper_trendline"]
            lower_start = key_levels["lower_trendline"]
            
            if "rising" in pattern_type:
                # Rising wedge: both lines slope up but converge
                upper_y = [upper_start, upper_start * 1.02]
                lower_y = [lower_start, upper_start * 0.98]
            elif "falling" in pattern_type:
                # Falling wedge: both lines slope down but converge
                upper_y = [upper_start, lower_start * 1.02]
                lower_y = [lower_start, lower_start * 0.98]
            else:
                # Broadening wedge: lines diverge
                mid_point = (upper_start + lower_start) / 2
                upper_y = [mid_point * 1.01, upper_start]
                lower_y = [mid_point * 0.99, lower_start]
            
            x_vals = [start_time, end_time]
            
            # Draw wedge lines
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=upper_y,
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=lower_y,
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False
            ))
            
            # Fill wedge
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=upper_y + lower_y[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                line=dict(width=0),
                showlegend=False
            ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_head_and_shoulders(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw head and shoulders pattern."""
        if len(pattern_data) >= 3:
            # Create the head and shoulders shape
            times = pattern_data['timestamp'].tolist()
            
            if "head" in key_levels:
                head_level = key_levels["head"]
                shoulder_level = key_levels.get("left_shoulder", key_levels.get("right_shoulder", head_level * 0.95))
                neckline = key_levels.get("neckline", head_level * 0.9)
                
                # Create H&S shape
                n_points = len(times)
                if n_points >= 5:
                    # Left shoulder, head, right shoulder pattern
                    y_vals = [
                        shoulder_level,  # Left shoulder
                        neckline,       # Valley
                        head_level,     # Head
                        neckline,       # Valley
                        shoulder_level  # Right shoulder
                    ]
                    x_vals = [times[i] for i in [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]]
                else:
                    # Simplified version
                    y_vals = [shoulder_level, head_level, shoulder_level]
                    x_vals = [times[0], times[n_points//2], times[-1]]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(size=8, color=color),
                    showlegend=False
                ))
                
                # Add neckline
                fig.add_trace(go.Scatter(
                    x=[times[0], times[-1]],
                    y=[neckline, neckline],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False
                ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_multiple_peaks(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw double/triple top/bottom patterns."""
        times = pattern_data['timestamp'].tolist()
        
        if "double" in pattern_type:
            # Two peaks/valleys with connecting lines
            if len(times) >= 2:
                if "top" in pattern_type:
                    # Double top - find the two highest points
                    high_indices = pattern_data['high'].nlargest(2).index
                    peak1_idx = min(high_indices)
                    peak2_idx = max(high_indices)
                    
                    peak1_time = pattern_data.loc[peak1_idx, 'timestamp']
                    peak2_time = pattern_data.loc[peak2_idx, 'timestamp']
                    peak1_price = pattern_data.loc[peak1_idx, 'high']
                    peak2_price = pattern_data.loc[peak2_idx, 'high']
                    
                    # Find valley between peaks
                    valley_data = pattern_data.loc[peak1_idx:peak2_idx]
                    valley_idx = valley_data['low'].idxmin()
                    valley_time = pattern_data.loc[valley_idx, 'timestamp']
                    valley_price = pattern_data.loc[valley_idx, 'low']
                    
                    # Draw the pattern
                    fig.add_trace(go.Scatter(
                        x=[peak1_time, valley_time, peak2_time],
                        y=[peak1_price, valley_price, peak2_price],
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(size=12, color=color, symbol="triangle-up"),
                        showlegend=False
                    ))
                    
                    # Add horizontal line connecting the peaks
                    fig.add_trace(go.Scatter(
                        x=[peak1_time, peak2_time],
                        y=[peak1_price, peak2_price],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        showlegend=False
                    ))
                    
                elif "bottom" in pattern_type:
                    # Double bottom - find the two lowest points
                    low_indices = pattern_data['low'].nsmallest(2).index
                    valley1_idx = min(low_indices)
                    valley2_idx = max(low_indices)
                    
                    valley1_time = pattern_data.loc[valley1_idx, 'timestamp']
                    valley2_time = pattern_data.loc[valley2_idx, 'timestamp']
                    valley1_price = pattern_data.loc[valley1_idx, 'low']
                    valley2_price = pattern_data.loc[valley2_idx, 'low']
                    
                    # Find peak between valleys
                    peak_data = pattern_data.loc[valley1_idx:valley2_idx]
                    peak_idx = peak_data['high'].idxmax()
                    peak_time = pattern_data.loc[peak_idx, 'timestamp']
                    peak_price = pattern_data.loc[peak_idx, 'high']
                    
                    # Draw the pattern
                    fig.add_trace(go.Scatter(
                        x=[valley1_time, peak_time, valley2_time],
                        y=[valley1_price, peak_price, valley2_price],
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(size=12, color=color, symbol="triangle-down"),
                        showlegend=False
                    ))
                    
                    # Add horizontal line connecting the valleys
                    fig.add_trace(go.Scatter(
                        x=[valley1_time, valley2_time],
                        y=[valley1_price, valley2_price],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        showlegend=False
                    ))
        
        elif "triple" in pattern_type:
            # Three peaks/valleys
            if len(times) >= 3:
                if "top" in pattern_type:
                    # Triple top
                    high_indices = pattern_data['high'].nlargest(3).index.sort_values()
                    peak_times = [pattern_data.loc[idx, 'timestamp'] for idx in high_indices]
                    peak_prices = [pattern_data.loc[idx, 'high'] for idx in high_indices]
                    
                    # Draw connecting lines
                    fig.add_trace(go.Scatter(
                        x=peak_times,
                        y=peak_prices,
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(size=12, color=color, symbol="triangle-up"),
                        showlegend=False
                    ))
                    
                elif "bottom" in pattern_type:
                    # Triple bottom
                    low_indices = pattern_data['low'].nsmallest(3).index.sort_values()
                    valley_times = [pattern_data.loc[idx, 'timestamp'] for idx in low_indices]
                    valley_prices = [pattern_data.loc[idx, 'low'] for idx in low_indices]
                    
                    # Draw connecting lines
                    fig.add_trace(go.Scatter(
                        x=valley_times,
                        y=valley_prices,
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(size=12, color=color, symbol="triangle-down"),
                        showlegend=False
                    ))
        
        # Add shaded background
        fig.add_vrect(
            x0=times[0],
            x1=times[-1],
            fillcolor=color,
            opacity=0.1,
            line_width=0
        )
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_cup_and_handle(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw cup and handle pattern."""
        times = pattern_data['timestamp'].tolist()
        
        if len(times) >= 10:
            # Create cup shape (U-shaped curve)
            rim_level = key_levels.get("rim", pattern_data['high'].max())
            bottom_level = key_levels.get("bottom", pattern_data['low'].min())
            
            # Cup portion (first 70% of pattern)
            cup_end = int(len(times) * 0.7)
            cup_times = times[:cup_end]
            
            # Create U-shaped curve for cup
            cup_y = []
            for i, t in enumerate(cup_times):
                # Parabolic curve for cup
                progress = i / (len(cup_times) - 1)
                depth = 4 * progress * (1 - progress)  # Parabolic curve
                y_val = rim_level - (rim_level - bottom_level) * depth
                cup_y.append(y_val)
            
            fig.add_trace(go.Scatter(
                x=cup_times,
                y=cup_y,
                mode="lines",
                line=dict(color=color, width=4),
                name="Cup",
                showlegend=False
            ))
            
            # Handle portion (last 30% of pattern)
            if "handle" in pattern_type:
                handle_times = times[cup_end:]
                handle_level = rim_level * 0.95
                handle_y = [handle_level] * len(handle_times)
                
                fig.add_trace(go.Scatter(
                    x=handle_times,
                    y=handle_y,
                    mode="lines",
                    line=dict(color=color, width=3, dash="dot"),
                    name="Handle",
                    showlegend=False
                ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_flag_pennant(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw flag and pennant patterns."""
        times = pattern_data['timestamp'].tolist()
        
        if "flag" in pattern_type:
            # Flag: rectangular consolidation
            if len(times) >= 2:
                upper_level = key_levels.get("upper", pattern_data['high'].max())
                lower_level = key_levels.get("lower", pattern_data['low'].min())
                
                # Draw flag rectangle
                fig.add_trace(go.Scatter(
                    x=[times[0], times[-1], times[-1], times[0], times[0]],
                    y=[upper_level, upper_level, lower_level, lower_level, upper_level],
                    mode="lines",
                    line=dict(color=color, width=3),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.1,
                    showlegend=False
                ))
        
        elif "pennant" in pattern_type:
            # Pennant: triangular consolidation
            if len(times) >= 2:
                upper_start = key_levels.get("upper", pattern_data['high'].max())
                lower_start = key_levels.get("lower", pattern_data['low'].min())
                convergence_point = (upper_start + lower_start) / 2
                
                # Draw pennant triangle
                fig.add_trace(go.Scatter(
                    x=[times[0], times[-1], times[0]],
                    y=[upper_start, convergence_point, lower_start],
                    mode="lines",
                    line=dict(color=color, width=3),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.1,
                    showlegend=False
                ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_rectangle(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw rectangle pattern."""
        times = pattern_data['timestamp'].tolist()
        
        if len(times) >= 2:
            upper_level = key_levels.get("upper", pattern_data['high'].max())
            lower_level = key_levels.get("lower", pattern_data['low'].min())
            
            # Draw rectangle
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1], times[-1], times[0], times[0]],
                y=[upper_level, upper_level, lower_level, lower_level, upper_level],
                mode="lines",
                line=dict(color=color, width=3),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                showlegend=False
            ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_diamond(self, fig, pattern_data, key_levels, color, pattern_type):
        """Draw diamond pattern."""
        times = pattern_data['timestamp'].tolist()
        
        if len(times) >= 4:
            center_time = times[len(times)//2]
            high_level = key_levels.get("high", pattern_data['high'].max())
            low_level = key_levels.get("low", pattern_data['low'].min())
            mid_level = (high_level + low_level) / 2
            
            # Draw diamond shape
            fig.add_trace(go.Scatter(
                x=[times[0], center_time, times[-1], center_time, times[0]],
                y=[mid_level, high_level, mid_level, low_level, mid_level],
                mode="lines",
                line=dict(color=color, width=3),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                showlegend=False
            ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_candlestick_pattern(self, fig, pattern_data, color, pattern_type):
        """Draw candlestick pattern markers."""
        if len(pattern_data) > 0:
            # Highlight the relevant candles
            for idx, row in pattern_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['close']],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=color,
                        symbol="diamond",
                        line=dict(width=2, color="white")
                    ),
                    showlegend=False
                ))
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _draw_default_pattern(self, fig, pattern_data, color, pattern_type):
        """Draw default pattern overlay."""
        if len(pattern_data) > 0:
            start_time = pattern_data['timestamp'].iloc[0]
            end_time = pattern_data['timestamp'].iloc[-1]
            
            # Add shaded region
            fig.add_vrect(
                x0=start_time,
                x1=end_time,
                fillcolor=color,
                opacity=0.2,
                line_width=0
            )
        
        self._add_pattern_annotation(fig, pattern_data, color, pattern_type)

    def _add_pattern_annotation(self, fig, pattern_data, color, pattern_type):
        """Add pattern annotation."""
        if len(pattern_data) > 0:
            # Use the end time for annotation placement
            end_time = pattern_data['timestamp'].iloc[-1]
            
            # Get a good Y position for the annotation
            if "top" in pattern_type:
                y_pos = pattern_data['high'].max() * 1.01
            elif "bottom" in pattern_type:
                y_pos = pattern_data['low'].min() * 0.99
            else:
                y_pos = pattern_data['close'].iloc[-1]
            
            # Clean up pattern name for display
            display_name = pattern_type.replace("_", " ").title()
            
            fig.add_annotation(
                x=end_time,
                y=y_pos,
                text=display_name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(color="black", size=11, family="Arial Black"),
                bgcolor=color,
                bordercolor="white",
                borderwidth=2,
                opacity=0.9,
                xanchor="left",
                yanchor="middle"
            )