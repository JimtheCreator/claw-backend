import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class ChartGenerator:
    def __init__(self, ohlcv_data: pd.DataFrame, analysis_data: Dict, theme: str = "plotly_dark"):
        """Initialize ChartGenerator with OHLCV data and analysis."""
        if isinstance(ohlcv_data, dict):
            self.ohlcv = pd.DataFrame(ohlcv_data)
        else:
            self.ohlcv = ohlcv_data.copy()

        self.analysis = analysis_data
        self.theme = theme
        
        # --- ROBUST FIX for X-AXIS ---
        # Ensure 'timestamp' column exists and convert it to datetime
        if 'timestamp' in self.ohlcv.columns:
            self.ohlcv['timestamp'] = pd.to_datetime(self.ohlcv['timestamp'])
            self.ohlcv.set_index('timestamp', inplace=True)
        # Handle cases where the index is not datetime (e.g., from a file without a timestamp column)
        elif not isinstance(self.ohlcv.index, pd.DatetimeIndex):
             # Fallback: if no timestamp, we must use a linear scale.
             # This prevents the app from crashing if data is malformed.
             print("Warning: No 'timestamp' column found. Using numerical index for x-axis.")

        # Remove timezone if present to avoid plotting issues
        if isinstance(self.ohlcv.index, pd.DatetimeIndex) and self.ohlcv.index.tz is not None:
            self.ohlcv.index = self.ohlcv.index.tz_convert(None)
        
        self.support_resistance = self._extract_key_levels()

    def _extract_key_levels(self) -> Tuple[List[float], List[float]]:
        market_context = self.analysis.get('market_context', {})
        supports = market_context.get('support_levels', [])
        resistances = market_context.get('resistance_levels', [])
        return supports, resistances
    
    def _draw_demand_supply_zones(self, fig: go.Figure):
        """Draws demand and supply zones as shaded rectangles."""
        market_context = self.analysis.get('market_context', {})
        # Debug print to see if zones are being found
        # print("DEBUG: Found Demand Zones:", market_context.get('demand_zones', []))
        # print("DEBUG: Found Supply Zones:", market_context.get('supply_zones', []))

        if not isinstance(self.ohlcv.index, pd.DatetimeIndex) or len(self.ohlcv.index) < 2:
            return # Cannot draw zones without a proper date axis

        chart_start_date = self.ohlcv.index[0]
        chart_end_date = self.ohlcv.index[-1]

        # Draw Demand Zones (Green)
        for zone in market_context.get('demand_zones', []):
            fig.add_shape(type="rect", x0=chart_start_date, y0=zone.get('bottom'), x1=chart_end_date, y1=zone.get('top'),
                          line=dict(width=0), fillcolor="rgba(0, 255, 0, 0.15)", layer="below")

        # Draw Supply Zones (Red)
        for zone in market_context.get('supply_zones', []):
            fig.add_shape(type="rect", x0=chart_start_date, y0=zone.get('bottom'), x1=chart_end_date, y1=zone.get('top'),
                          line=dict(width=0), fillcolor="rgba(255, 0, 0, 0.15)", layer="below")
            
    
    def create_chart_image(self, width: int = 1200, height: int = 600) -> bytes:
        """Create a candlestick chart with all analysis overlays."""
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=self.ohlcv.index, open=self.ohlcv['open'], high=self.ohlcv['high'],
            low=self.ohlcv['low'], close=self.ohlcv['close'], name='Price'
        ))

        self._draw_demand_supply_zones(fig)

        # With this corrected version:
        # In create_chart_image method:
        if self.support_resistance:
            supports, resistances = self.support_resistance
            all_levels = supports + resistances
            x_end = self.ohlcv.index[-1]
            for level in all_levels:
                # Skip non-numeric levels
                if not isinstance(level, (int, float)):
                    continue
                fig.add_shape(type="line", x0=self.ohlcv.index[0], y0=level,
                            x1=x_end, y1=level, line=dict(color="green", dash="dash", width=1))
                fig.add_annotation(x=x_end, y=level, text=f"${level:.2f}", showarrow=False,
                                xanchor="left", font=dict(color="#ffd700", size=9), bgcolor="rgba(0,0,0,0.6)")
        
        self._draw_patterns(fig)
        
        fig.update_layout(
            title='', xaxis_title="", yaxis_title="",
            xaxis_rangeslider_visible=False, template=self.theme,
            yaxis=dict(side="right", showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(
                type='date' if isinstance(self.ohlcv.index, pd.DatetimeIndex) else 'linear',
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)'
            ),
            showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=80, t=50, b=40), width=width, height=height, hovermode='x unified'
        )
        
        return fig.to_image(format="png", width=width, height=height, scale=2)

    def _draw_patterns(self, fig: go.Figure) -> None:
        """NEW: Intelligently dispatches to the correct pattern drawing function."""
        pattern_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
        
        for i, pattern in enumerate(self.analysis.get('patterns', [])):
            try:
                pattern_data = self.ohlcv.iloc[pattern['start_idx'] : pattern['end_idx'] + 1]
                color = pattern_colors[i % len(pattern_colors)]
                p_type = pattern.get('pattern', '').lower()

                if 'double_top' in p_type or 'triple_top' in p_type:
                    self._draw_peak_valley_pattern(fig, pattern_data, pattern['key_levels'], color, 'top')
                elif 'double_bottom' in p_type or 'triple_bottom' in p_type:
                    self._draw_peak_valley_pattern(fig, pattern_data, pattern['key_levels'], color, 'bottom')
                # Add other pattern types here (e.g., head and shoulders)
            except Exception as e:
                print(f"Error drawing pattern {i} ({pattern.get('pattern')}): {e}")

    def _draw_peak_valley_pattern(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str, pattern_kind: str):
        """
        NEW: A single, smarter function to draw all top/bottom patterns and their connecting lines.
        """
        if pattern_kind == 'top':
            price_series, point_name, level_keys = 'high', 'Peak', ['first_peak', 'second_peak', 'third_peak']
        else: # bottom
            price_series, point_name, level_keys = 'low', 'Valley', ['first_peak', 'second_peak', 'third_peak'] # Note: Your JSON uses 'peak' for bottom patterns too

        points = []
        for key in level_keys:
            if key in key_levels:
                target_value = key_levels[key]
                # Find the timestamp of the point in the data slice
                time_of_point = (data[price_series] - target_value).abs().idxmin()
                price_at_point = data.loc[time_of_point, price_series]
                points.append((time_of_point, price_at_point))

        # --- FIX for MISSING LINES ---
        # Only draw the trace and annotations if we found points to connect
        if len(points) >= 2:
            # Draw connecting line
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points], y=[p[1] for p in points],
                mode='lines+markers', line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6), showlegend=False
            ))

            # Add annotations
            for i, (time, price) in enumerate(points):
                fig.add_annotation(
                    x=time, y=price, text=f"{point_name} {i+1}",
                    showarrow=True, arrowhead=2, font=dict(color=color, size=10),
                    bgcolor="rgba(0,0,0,0.6)"
                )
    def _draw_double_or_triple_top(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str, num_peaks: int) -> None:
        """Draws double or triple top patterns."""
        if len(data) < num_peaks: return

        peak_keys = {1: ['first_peak'], 2: ['first_peak', 'second_peak'], 3: ['first_peak', 'second_peak', 'third_peak']}
        peaks = []
        for i, peak_key in enumerate(peak_keys.get(num_peaks, [])):
            if peak_key in key_levels:
                target_value = key_levels[peak_key]
                # Find the index in the pattern's data slice that is closest to the peak value
                time_of_peak = (data['high'] - target_value).abs().idxmin()
                price_at_peak = data.loc[time_of_peak, 'high']
                peaks.append((time_of_peak, price_at_peak))

                # Add annotation for the peak
                fig.add_annotation(
                    x=time_of_peak, y=price_at_peak, text=f"Peak {i+1}",
                    showarrow=True, arrowhead=2, font=dict(color=color, size=10),
                    bgcolor="rgba(0,0,0,0.6)"
                )
        
        if len(peaks) == num_peaks:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in peaks], y=[p[1] for p in peaks],
                mode='lines+markers', line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6), showlegend=False
            ))


    def _draw_double_top(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str) -> None:
        """Draw double top pattern."""
        if len(data) < 3:
            return
        
        # Find the two highest points
        peaks = []
        for peak_key in ['first_peak', 'second_peak']:
            if peak_key in key_levels:
                target_value = key_levels[peak_key]
                idx = (data['high'] - target_value).abs().idxmin()
                timestamp = data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else data.index[idx]
                price = data.loc[idx, 'high']
                peaks.append((timestamp, price))
        
        if len(peaks) == 2:
            # Draw line connecting peaks
            fig.add_trace(go.Scatter(
                x=[peaks[0][0], peaks[1][0]],
                y=[peaks[0][1], peaks[1][1]],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name='Double Top',
                showlegend=False
            ))
            
            # Add annotations
            for i, (time, price) in enumerate(peaks):
                fig.add_annotation(
                    x=time, y=price,
                    text=f"Peak {i+1}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=color,
                    font=dict(color=color, size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
    
    def _draw_triple_top(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str) -> None:
        """Draw triple top pattern."""
        if len(data) < 5:
            return
        
        # Find the three highest points
        peaks = []
        for peak_key in ['first_peak', 'second_peak', 'third_peak']:
            if peak_key in key_levels:
                target_value = key_levels[peak_key]
                idx = (data['high'] - target_value).abs().idxmin()
                timestamp = data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else data.index[idx]
                price = data.loc[idx, 'high']
                peaks.append((timestamp, price))
        
        if len(peaks) == 3:
            # Draw line connecting peaks
            fig.add_trace(go.Scatter(
                x=[p[0] for p in peaks],
                y=[p[1] for p in peaks],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name='Triple Top',
                showlegend=False
            ))
            
            # Add annotations
            for i, (time, price) in enumerate(peaks):
                fig.add_annotation(
                    x=time, y=price,
                    text=f"Peak {i+1}",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color=color, size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
    
    def _draw_head_and_shoulders(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str, is_inverse: bool = False) -> None:
        """Draw head and shoulders pattern."""
        if len(data) < 5:
            return
        
        # Find shoulders and head
        points = []
        labels = ['Left Shoulder', 'Head', 'Right Shoulder']
        point_type = 'low' if is_inverse else 'high'
        
        for key in ['left_shoulder', 'head', 'right_shoulder']:
            if key in key_levels:
                target_value = key_levels[key]
                idx = (data[point_type] - target_value).abs().idxmin()
                timestamp = data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else data.index[idx]
                price = data.loc[idx, point_type]
                points.append((timestamp, price))
        
        if len(points) == 3:
            # Draw line connecting points
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name=f'{"Inverse " if is_inverse else ""}Head & Shoulders',
                showlegend=False
            ))
            
            # Add annotations
            for i, (time, price) in enumerate(points):
                fig.add_annotation(
                    x=time, y=price,
                    text=labels[i],
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color=color, size=10),
                    bgcolor="rgba(0,0,0,0.5)",
                    yanchor="bottom" if is_inverse else "top"
                )
    
    def _draw_generic_pattern(self, fig: go.Figure, data: pd.DataFrame, key_levels: Dict, color: str, pattern_type: str) -> None:
        """Draw a generic pattern using available key levels."""
        if len(data) < 2:
            return
        
        # Extract any numeric key levels and plot them
        points = []
        for key, value in key_levels.items():
            if isinstance(value, (int, float)):
                # Find closest point in data
                high_idx = (data['high'] - value).abs().idxmin()
                low_idx = (data['low'] - value).abs().idxmin()
                
                # Choose the closer match
                if abs(data.loc[high_idx, 'high'] - value) < abs(data.loc[low_idx, 'low'] - value):
                    timestamp = data.loc[high_idx, 'timestamp'] if 'timestamp' in data.columns else data.index[high_idx]
                    price = data.loc[high_idx, 'high']
                else:
                    timestamp = data.loc[low_idx, 'timestamp'] if 'timestamp' in data.columns else data.index[low_idx]
                    price = data.loc[low_idx, 'low']
                
                points.append((timestamp, price))
        
        if len(points) >= 2:
            # Sort by timestamp
            points.sort(key=lambda x: x[0])
            
            # Draw connecting line
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=4),
                name=pattern_type.replace('_', ' ').title(),
                showlegend=False
            ))