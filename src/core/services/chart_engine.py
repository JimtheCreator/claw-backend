import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import pandas as pd
from common.logger import logger

class ChartEngine:
    def __init__(self, ohlcv_data: Dict[str, list], analysis_data: Optional[Dict[str, Any]] = None, config: Optional[Dict] = None):
        """
        Initializes the ChartEngine.

        Args:
            ohlcv_data (Dict[str, list]): OHLCV data.
            analysis_data (Dict[str, Any]): Pattern analysis results.
            config (Dict, optional): Configuration for chart aesthetics.
        """
        self.ohlcv = ohlcv_data
        self.analysis = analysis_data
        
        # Convert the ohlcv data dictionary to a DataFrame
        self.ohlcv_df = pd.DataFrame(self.ohlcv)
        self.ohlcv_df['timestamp'] = pd.to_datetime(self.ohlcv_df['timestamp'])
        self.ohlcv_df = self.ohlcv_df.sort_values('timestamp')
        self.ohlcv_df = self.ohlcv_df.drop_duplicates(subset='timestamp')
        self.ohlcv_df = self.ohlcv_df.reset_index(drop=True)
        self.ohlcv_df['timestamp'] = self.ohlcv_df['timestamp'].apply(lambda x: x.to_pydatetime())

        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Main Plotly Figure object
        self.fig: Optional[go.Figure] = None

        # Map pattern names to their drawing methods
        self._label_positions = []
        self.pattern_drawing_map = {
            "double_top": self._draw_double_top,
            "double_bottom": self._draw_double_bottom,
            "triple_top": self._draw_triple_top,
            "triple_bottom": self._draw_triple_bottom,
            "ascending_channel": self._draw_channel,
            "descending_channel": self._draw_channel,
            "horizontal_channel": self._draw_channel,
            "symmetrical_triangle": self._draw_triangle,
            "ascending_triangle": self._draw_triangle,
            "descending_triangle": self._draw_triangle,
            "morning_star": self._draw_morning_star,
            "three_outside_up": self._draw_three_outside_up,
            "three_outside_down": self._draw_three_outside_down,
            "three_inside_down": self._draw_three_inside_down,
            "three_inside_up": self._draw_three_inside_up,
            "cup_and_handle": self._draw_cup_and_handle,
            # ... other patterns will be mapped here
        }

    def _get_default_config(self) -> Dict:
        """Provides default styling for a premium feel."""
        return {
            "template": "plotly_dark",
            "font": {"family": "Arial, sans-serif", "size": 12, "color": "#E0E0E0"},
            "colors": {
                "candlestick_increasing": "#14B89D",
                "candlestick_decreasing": "#F23645",
                "volume_increasing": "#14B89D",
                "volume_decreasing": "#F23645",
                "supply_zone_fill": "rgba(242, 54, 69, 0.2)",
                "demand_zone_fill": "rgba(20, 184, 157, 0.2)",
                "support_line": "#14B89D",
                "resistance_line": "#F23645",
                "pattern_fill_double_top": "rgba(242, 54, 69, 0.25)",
                "pattern_line_double_top": "rgba(242, 54, 69, 0.7)",
                "pattern_fill_double_bottom": "rgba(20, 184, 157, 0.25)",
                "pattern_line_double_bottom": "rgba(20, 184, 157, 0.7)",
                "target_line": "rgba(255, 0, 255, 0.7)",
                "annotation_bg": "rgba(242, 54, 69, 0.8)",
                "annotation_font": "#FFFFFF"
            }
        }

    def _create_figure(self):
        """Initializes the Plotly Figure with subplots."""
        self.fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.8, 0.2]
        )
        self.fig.update_layout(
            template=self.config['template'],
            font=self.config['font'],
            xaxis_rangeslider_visible=False,
            showlegend=False,  # Hide the legend
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Explicitly set x-axis type to date for both subplots
        self.fig.update_xaxes(showgrid=False, type="date", row=1, col=1)
        self.fig.update_xaxes(showgrid=False, type="date", row=2, col=1)
        # Move the price y-axis to the right and set autorange for tight fit
        min_price = float(self.ohlcv_df['low'].min())
        max_price = float(self.ohlcv_df['high'].max())
        price_padding = (max_price - min_price) * 0.03 if max_price > min_price else 1
        self.fig.update_yaxes(showgrid=False, side="right", row=1, col=1, autorange=True, range=[min_price - price_padding, max_price + price_padding])
        self.fig.update_yaxes(showgrid=False, row=2, col=1)

    def _add_candlestick_trace(self):
        """Adds the candlestick series to the main chart."""
        self.fig.add_trace(go.Candlestick(
            x=self.ohlcv_df['timestamp'].astype(str),  # Use string timestamps for test
            open=self.ohlcv_df['open'], high=self.ohlcv_df['high'],
            low=self.ohlcv_df['low'], close=self.ohlcv_df['close'],
            increasing_line_color=self.config['colors']['candlestick_increasing'],
            decreasing_line_color=self.config['colors']['candlestick_decreasing'],
            name='Candlesticks'
        ), row=1, col=1)

    def _add_volume_trace(self):
        """Adds the volume series to the secondary chart."""
        colors = [
            self.config['colors']['volume_increasing'] if self.ohlcv_df['close'][i] >= self.ohlcv_df['open'][i]
            else self.config['colors']['volume_decreasing']
            for i in range(len(self.ohlcv_df))
        ]
        self.fig.add_trace(go.Bar(
            x=self.ohlcv_df['timestamp'].astype(str),  # Use string timestamps for volume bars too
            y=self.ohlcv_df['volume'],
            name='Volume', marker_color=colors
        ), row=2, col=1)
        # Remove y-axis title and tick labels for the volume axis
        self.fig.update_yaxes(title_text=None, showticklabels=False, row=2, col=1)

    def _should_draw_significant_pattern(self, pattern: dict) -> bool:
        # Add null check for analysis
        if self.analysis is None:
            return
        """Return True if the pattern is significant enough to show to a trader."""
        # 1. Pattern must span at least 10% of visible chart
        total_candles = len(self.ohlcv_df)
        pattern_span = pattern.get('end_idx', 0) - pattern.get('start_idx', 0) + 1
        if pattern_span < max(5, int(0.1 * total_candles)):
            return False
        # 2. Pattern must be near a major support/resistance (within 1%)
        key_levels = pattern.get('key_levels', {})
        price_points = [v for k, v in key_levels.items() if isinstance(v, (int, float))]
        context = self.analysis.get('market_context', {})
        supports = context.get('support_levels', [])
        resistances = context.get('resistance_levels', [])
        all_levels = supports + resistances
        if not all_levels or not price_points:
            return False
        close_to_level = any(
            abs(p - lvl) / max(1, abs(lvl)) < 0.01 for p in price_points for lvl in all_levels
        )
        if not close_to_level:
            return False
        # 3. Trend alignment must be strong
        trend_alignment = pattern.get('trader_aware_scores', {}).get('trend_alignment', 0)
        if trend_alignment < 0.5:
            return False
        return True

    # Patch pattern drawing to use the significance filter and add context annotation
    def _draw_patterns(self):
        # Add null check for analysis
        if self.analysis is None:
            return
        # If analysis is a list of pattern dicts
        if isinstance(self.analysis, list):
            for pattern in self.analysis:
                pattern_name = pattern.get('pattern_name') or pattern.get('pattern_type') or pattern.get('pattern')
                if pattern_name in self.pattern_drawing_map:
                    self.pattern_drawing_map[pattern_name](pattern)
        # If analysis is a dict with a 'patterns' key (legacy support)
        elif isinstance(self.analysis, dict) and 'patterns' in self.analysis:
            for pattern in self.analysis['patterns']:
                pattern_name = pattern.get('pattern_name') or pattern.get('pattern_type') or pattern.get('pattern')
                if pattern_name in self.pattern_drawing_map:
                    self.pattern_drawing_map[pattern_name](pattern)

    def _find_peak_valley_coords(self, key_levels, candle_indexes):
        coords = {}
        valid_indexes = [idx for idx in candle_indexes if idx in self.ohlcv_df.index]
        if not valid_indexes:
            print(f"Warning: No valid candle data for pattern in index range.")
            return None

        # Defensive: Use key_levels if present, else fallback to OHLCV data
        if all(k in key_levels for k in ['first_peak', 'second_peak', 'valley']):
            coords['p1_y'] = key_levels['first_peak']
            coords['p2_y'] = key_levels['second_peak']
            coords['v_y'] = key_levels['valley']
            # For x-values, find the closest index in the window
            df = self.ohlcv_df.loc[valid_indexes]
            coords['p1_idx'] = (df['high'] - coords['p1_y']).abs().idxmin()
            coords['p1_x'] = self.ohlcv_df.loc[coords['p1_idx'], 'timestamp']
            coords['p2_idx'] = (df['high'] - coords['p2_y']).abs().idxmin()
            coords['p2_x'] = self.ohlcv_df.loc[coords['p2_idx'], 'timestamp']
            coords['v_idx'] = (df['low'] - coords['v_y']).abs().idxmin()
            coords['v_x'] = self.ohlcv_df.loc[coords['v_idx'], 'timestamp']
        else:
            df = self.ohlcv_df.loc[valid_indexes]
            # Find two highest highs for peaks
            high_vals = df['high'].nlargest(2)
            coords['p1_y'] = high_vals.iloc[0]
            coords['p2_y'] = high_vals.iloc[1] if len(high_vals) > 1 else high_vals.iloc[0]
            # Find lowest low for valley
            coords['v_y'] = df['low'].min()
            # For x-values, find the corresponding indices
            coords['p1_idx'] = (df['high'] - coords['p1_y']).abs().idxmin()
            coords['p1_x'] = self.ohlcv_df.loc[coords['p1_idx'], 'timestamp']
            coords['p2_idx'] = (df['high'] - coords['p2_y']).abs().idxmin()
            coords['p2_x'] = self.ohlcv_df.loc[coords['p2_idx'], 'timestamp']
            coords['v_idx'] = (df['low'] - coords['v_y']).abs().idxmin()
            coords['v_x'] = self.ohlcv_df.loc[coords['v_idx'], 'timestamp']
        return coords

    def _find_valley_peak_coords(self, key_levels, candle_indexes):
        """Finds the two valleys (bottoms) and the highest peak (neckline) between them for double bottom patterns."""
        valid_indexes = [idx for idx in candle_indexes if idx in self.ohlcv_df.index]
        if len(valid_indexes) < 3:
            return None
        df = self.ohlcv_df.loc[valid_indexes]
        # Find two lowest points (valleys)
        valley1_idx = df['low'].idxmin()
        valley1_time = self.ohlcv_df.loc[valley1_idx, 'timestamp']
        valley1_price = self.ohlcv_df.loc[valley1_idx, 'low']
        # Remove first valley, find second lowest
        df_wo_valley1 = df.drop(valley1_idx)
        if df_wo_valley1.empty:
            return None
        valley2_idx = df_wo_valley1['low'].idxmin()
        valley2_time = self.ohlcv_df.loc[valley2_idx, 'timestamp']
        valley2_price = self.ohlcv_df.loc[valley2_idx, 'low']
        # Neckline: highest high between the valleys
        start, end = sorted([valley1_idx, valley2_idx])
        neckline_df = self.ohlcv_df.loc[start:end]
        peak_idx = neckline_df['high'].idxmax()
        peak_time = self.ohlcv_df.loc[peak_idx, 'timestamp']
        peak_price = self.ohlcv_df.loc[peak_idx, 'high']
        return {
            'b1_idx': valley1_idx, 'b1_x': valley1_time, 'b1_y': valley1_price,
            'b2_idx': valley2_idx, 'b2_x': valley2_time, 'b2_y': valley2_price,
            'neck_idx': peak_idx, 'neck_x': peak_time, 'neck_y': peak_price
        }

    def _get_label_y(self, x_text, y_text, offset):
        # Stack label if it would collide with a previous one at the same x
        stack_count = 0
        for prev_x, prev_y in self._label_positions:
            if abs((prev_x - x_text).total_seconds()) < 60*60 and abs(prev_y - y_text) < offset * 0.5:
                stack_count += 1
        y_text += stack_count * offset * 2.0
        self._label_positions.append((x_text, y_text))
        return y_text

    def _get_pattern_arrow(self, pattern_name):
        # Upward for bullish, downward for bearish
        up_patterns = ['three_inside_up', 'three_outside_up', 'morning_star']
        down_patterns = ['three_inside_down', 'three_outside_down']
        if pattern_name in up_patterns:
            return ' ↗'
        elif pattern_name in down_patterns:
            return ' ↘'
        return ''

    def _draw_double_top(self, pattern: Dict):
        """Draws overlays for a Double Top pattern, like in the example image."""
        print(f"Drawing Double Top from index {pattern['start_idx']} to {pattern['end_idx']}")
        key_levels = pattern['key_levels']
        
        # Use the full range from the pattern for finding coordinates
        candle_indexes = range(pattern['start_idx'], pattern['end_idx'] + 1)
        
        # 1. Find the coordinates of the two peaks and the valley
        coords = self._find_peak_valley_coords(key_levels, candle_indexes)
        
        if coords is None:
            return

        # --- Use Scatter for filled polygon and solid lines ---
        # Polygon points: Top1 -> Valley -> Top2 -> horizontal to Top2_y -> horizontal to Top1_y -> close
        poly_x = [coords['p1_x'], coords['v_x'], coords['p2_x'], coords['p2_x'], coords['p1_x'], coords['p1_x']]
        poly_y = [coords['p1_y'], coords['v_y'], coords['p2_y'], coords['v_y'], coords['v_y'], coords['p1_y']]
        self.fig.add_trace(go.Scatter(
            x=poly_x,
            y=poly_y,
            fill='toself',
            mode='lines',
            line=dict(color=self.config['colors']['pattern_line_double_top'], width=0),
            fillcolor=self.config['colors']['pattern_fill_double_top'],
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)
        # Solid outline: Top1 -> Valley -> Top2
        self.fig.add_trace(go.Scatter(
            x=[coords['p1_x'], coords['v_x'], coords['p2_x']],
            y=[coords['p1_y'], coords['v_y'], coords['p2_y']],
            mode='lines',
            line=dict(color=self.config['colors']['pattern_line_double_top'], width=2),
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)

        # 3. Add annotations for the tops
        for i, p_key in enumerate(['p1', 'p2']):
            self.fig.add_annotation(
                x=coords[f'{p_key}_x'], y=coords[f'{p_key}_y'],
                text=f"Top {i+1}", showarrow=False,
                bgcolor=self.config['colors']['annotation_bg'],
                font=dict(color=self.config['colors']['annotation_font'], size=10),
                yanchor="bottom", yshift=5
            )

        # 4. Draw the projected target line (dotted)
        # For reliability: Use the next support level below the neckline (valley) as the target if available and within 5% below the neckline.
        # Otherwise, cap the target at 5% below the neckline.
        support_levels = sorted([
            lvl for lvl in self.analysis.get('market_context', {}).get('support_levels', [])
            if lvl < coords['v_y']
        ], reverse=True)
        max_drop = coords['v_y'] * 0.05  # 5% of neckline price
        if support_levels and (coords['v_y'] - support_levels[0]) <= max_drop:
            target_price = support_levels[0]
        else:
            pattern_height = coords['p1_y'] - coords['v_y']
            if pattern_height > max_drop:
                pattern_height = max_drop
            target_price = coords['v_y'] - pattern_height
        first_idx = min(coords['p1_idx'], coords['p2_idx'], coords['v_idx'])
        last_idx = max(coords['p1_idx'], coords['p2_idx'], coords['v_idx'])
        start_time = self.ohlcv_df.loc[first_idx, 'timestamp']
        end_time = self.ohlcv_df.loc[last_idx, 'timestamp']
        projection_time = end_time + (end_time - start_time) / 2
        # Dotted neckline extension
        self.fig.add_shape(type="line",
            x0=coords['p2_x'], y0=coords['v_y'], x1=end_time, y1=coords['v_y'],
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        # Dotted vertical drop
        self.fig.add_shape(type="line",
            x0=end_time, y0=coords['v_y'], x1=end_time, y1=target_price,
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        # Dotted horizontal target
        self.fig.add_shape(type="line",
            x0=end_time, y0=target_price, x1=projection_time, y1=target_price,
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        self.fig.add_annotation(
            x=projection_time, y=target_price, text="Target", showarrow=False,
            bgcolor=self.config['colors']['annotation_bg'], font=dict(color=self.config['colors']['annotation_font'], size=10),
            xanchor="left", xshift=5
        )


    def _draw_double_bottom(self, pattern: Dict):
        """Draws overlays for a Double Bottom pattern, matching educational chart style."""
        if pattern.get('confidence', 0) < 0.6:
            return
        key_levels = pattern['key_levels']
        candle_indexes = range(pattern['start_idx'], pattern['end_idx'] + 1)
        coords = self._find_valley_peak_coords(key_levels, candle_indexes)
        if coords is None:
            return
        # Polygon: Bottom1 → Neckline → Bottom2 → Bottom1
        poly_x = [coords['b1_x'], coords['neck_x'], coords['b2_x'], coords['b1_x']]
        poly_y = [coords['b1_y'], coords['neck_y'], coords['b2_y'], coords['b1_y']]
        self.fig.add_trace(go.Scatter(
            x=poly_x,
            y=poly_y,
            fill='toself',
            mode='lines',
            line=dict(color=self.config['colors']['pattern_line_double_bottom'], width=2),
            fillcolor=self.config['colors']['pattern_fill_double_bottom'],
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)
        # Outline: Bottom1 → Neckline → Bottom2
        self.fig.add_trace(go.Scatter(
            x=[coords['b1_x'], coords['neck_x'], coords['b2_x']],
            y=[coords['b1_y'], coords['neck_y'], coords['b2_y']],
            mode='lines',
            line=dict(color=self.config['colors']['pattern_line_double_bottom'], width=2),
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)
        # Dotted neckline
        self.fig.add_shape(type="line",
            x0=coords['b1_x'], y0=coords['neck_y'], x1=coords['b2_x'], y1=coords['neck_y'],
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        # Target projection (vertical and horizontal dotted lines)
        pattern_height = abs(coords['neck_y'] - min(coords['b1_y'], coords['b2_y']))
        target_price = coords['neck_y'] + pattern_height
        # Project from neckline breakout (right of Bottom2)
        target_time = coords['b2_x']
        self.fig.add_shape(type="line",
            x0=target_time, y0=coords['neck_y'], x1=target_time, y1=target_price,
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        self.fig.add_shape(type="line",
            x0=target_time, y0=target_price, x1=target_time + (target_time - coords['b1_x']) * 0.2, y1=target_price,
            line=dict(color=self.config['colors']['target_line'], width=2, dash="dot"), layer='above', row=1, col=1
        )
        # Labels
        self.fig.add_annotation(
            x=coords['b1_x'], y=coords['b1_y'], text="Bottom 1", showarrow=False,
            bgcolor=self.config['colors']['pattern_line_double_bottom'], font=dict(color="#fff", size=12), yanchor="top", yshift=10)
        self.fig.add_annotation(
            x=coords['b2_x'], y=coords['b2_y'], text="Bottom 2", showarrow=False,
            bgcolor=self.config['colors']['pattern_line_double_bottom'], font=dict(color="#fff", size=12), yanchor="top", yshift=10)
        self.fig.add_annotation(
            x=target_time + (target_time - coords['b1_x']) * 0.2, y=target_price, text="Target", showarrow=False,
            bgcolor=self.config['colors']['pattern_line_double_bottom'], font=dict(color="#fff", size=12), xanchor="left", yanchor="bottom", xshift=10, yshift=10)


    def _draw_triple_top(self, pattern: Dict):
        print(f"Drawing Triple Top from index {pattern['start_idx']} to {pattern['end_idx']}")


    def _draw_triple_bottom(self, pattern: Dict):
        print(f"Drawing Triple Bottom from index {pattern['start_idx']} to {pattern['end_idx']}")


    def _draw_channel(self, pattern: Dict):
        print(f"Drawing Channel from index {pattern['start_idx']} to {pattern['end_idx']}")
        
        
    def _draw_triangle(self, pattern: Dict):
        print(f"TRIANGLE DRAW: {pattern}")
        print(f"Drawing {pattern.get('pattern_type', 'triangle').replace('_', ' ').title()} from index {pattern['start_idx']} to {pattern['end_idx']}")
        key_levels = pattern['key_levels']
        pattern_type = pattern.get('pattern_type', 'symmetrical_triangle')
        candle_indexes = pattern['candle_indexes']
        if not candle_indexes:
            return

        color_map = {
            "symmetrical_triangle": "rgba(242,54,69,0.25)",   # Red
            "ascending_triangle": "rgba(0,102,255,0.25)",     # Blue
            "descending_triangle": "rgba(0,153,51,0.25)",     # Green
        }
        line_map = {
            "symmetrical_triangle": "rgba(242,54,69,0.7)",
            "ascending_triangle": "rgba(0,102,255,0.7)",
            "descending_triangle": "rgba(0,153,51,0.7)",
        }
        fillcolor = color_map.get(pattern_type, "rgba(255,255,255,0.15)")
        linecolor = line_map.get(pattern_type, "rgba(255,255,255,0.7)")

        left_idx = candle_indexes[0]
        right_idx = candle_indexes[-1]
        left_time = self.ohlcv_df.loc[left_idx, 'timestamp']
        right_time = self.ohlcv_df.loc[right_idx, 'timestamp']
        base_y = key_levels.get('pattern_bottom', None)
        top_y = key_levels.get('pattern_top', None)
        if base_y is None or top_y is None:
            return

        # --- SHAPE LOGIC ---
        if pattern_type == "ascending_triangle":
            start_idx = pattern.get('start_idx', candle_indexes[0])
            end_idx = pattern.get('end_idx', candle_indexes[-1])
            start_time = self.ohlcv_df.loc[start_idx, 'timestamp']
            end_time = self.ohlcv_df.loc[end_idx, 'timestamp']
            # Move apex leftward to shorten the nose
            apex_idx = max(start_idx + 2, end_idx - 2)
            apex_time = self.ohlcv_df.loc[apex_idx, 'timestamp']
            pattern_top = key_levels.get('pattern_top')
            pattern_bottom = key_levels.get('pattern_bottom')
            if pattern_top is None or pattern_bottom is None:
                return
            # Shortened nose: left base, apex, left top
            points_x = [start_time, apex_time, start_time]
            points_y = [pattern_bottom, pattern_top, pattern_top]
        elif pattern_type == "descending_triangle":
            start_idx = pattern.get('start_idx', candle_indexes[0])
            end_idx = pattern.get('end_idx', candle_indexes[-1])
            start_time = self.ohlcv_df.loc[start_idx, 'timestamp']
            end_time = self.ohlcv_df.loc[end_idx, 'timestamp']
            pattern_top = key_levels.get('pattern_top')
            pattern_bottom = key_levels.get('pattern_bottom')
            if pattern_top is None or pattern_bottom is None:
                return
            points_x = [start_time, end_time, start_time]
            points_y = [pattern_top, pattern_bottom, pattern_bottom]
        elif pattern_type == "symmetrical_triangle":
            start_idx = pattern.get('start_idx', candle_indexes[0])
            end_idx = pattern.get('end_idx', candle_indexes[-1])
            start_time = self.ohlcv_df.loc[start_idx, 'timestamp']
            end_time = self.ohlcv_df.loc[end_idx, 'timestamp']
            pattern_top = key_levels.get('pattern_top')
            pattern_bottom = key_levels.get('pattern_bottom')
            if pattern_top is None or pattern_bottom is None:
                return
            apex_y = (pattern_top + pattern_bottom) / 2
            points_x = [start_time, end_time, start_time]
            points_y = [pattern_top, apex_y, pattern_bottom]

        # Draw filled triangle
        self.fig.add_trace(go.Scatter(
            x=points_x + [points_x[0]],
            y=points_y + [points_y[0]],
            fill='toself',
            mode='lines',
            line=dict(color=linecolor, width=2),
            fillcolor=fillcolor,
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)

        # Draw triangle outline
        self.fig.add_trace(go.Scatter(
            x=points_x + [points_x[0]],
            y=points_y + [points_y[0]],
            mode='lines',
            line=dict(color=linecolor, width=2),
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)

        # Add label
        if pattern_type == "symmetrical_triangle":
            center_time = end_time
            center_y = apex_y
        elif pattern_type == "ascending_triangle":
            center_time = apex_time
            center_y = pattern_top
        elif pattern_type == "descending_triangle":
            center_time = end_time
            center_y = pattern_bottom
        else:
            center_time = points_x[0]
            center_y = points_y[0]
        label = pattern_type.replace('_', ' ').title()
        self.fig.add_annotation(
            x=center_time, y=center_y, text=label, showarrow=False,
            bgcolor=linecolor, font=dict(color="#fff", size=12), yanchor="bottom", yshift=10
        )

    def _draw_morning_star(self, pattern: Dict):
        """Highlights the 3 main candles of a Morning Star pattern with a gradient rectangle and adds the pattern name as text."""
        candle_indexes = pattern.get('candle_indexes')
        if candle_indexes is None:
            start_idx = pattern.get('start_index') or pattern.get('start_idx')
            end_idx = pattern.get('end_index') or pattern.get('end_idx')
            if start_idx is not None and end_idx is not None:
                candle_indexes = list(range(start_idx, end_idx + 1))
            else:
                return
        valid_indexes = [idx for idx in candle_indexes[:3] if idx in self.ohlcv_df.index]
        if not valid_indexes:
            return
        times = [self.ohlcv_df.loc[idx, 'timestamp'] for idx in valid_indexes]
        lows = [self.ohlcv_df.loc[idx, 'low'] for idx in valid_indexes]
        highs = [self.ohlcv_df.loc[idx, 'high'] for idx in valid_indexes]
        x0, x1 = min(times), max(times)
        y0, y1 = min(lows), max(highs)
        # Draw semi-transparent rectangle (yellow)
        self.fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="yellow", width=2),
            fillcolor="rgba(255,255,0,0.25)",
            row=1, col=1
        )
        # Add pattern name as text using Scatter, just above the rectangle
        center_idx = valid_indexes[len(valid_indexes) // 2]
        center_time = self.ohlcv_df.loc[center_idx, 'timestamp']
        price_range = float(self.ohlcv_df['high'].max()) - float(self.ohlcv_df['low'].min())
        offset = price_range * 0.01
        y_text = max(highs) + offset
        y_text = self._get_label_y(center_time, y_text, offset)
        label = pattern.get('pattern', '').replace('_', ' ').title() + self._get_pattern_arrow(pattern.get('pattern', ''))
        self.fig.add_trace(go.Scatter(
            x=[center_time],
            y=[y_text],
            text=[label],
            mode="text",
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="top center",
            showlegend=False
        ), row=1, col=1)

    def _draw_three_outside_up(self, pattern: Dict):
        """Highlights the 3 main candles of a Three Outside Up pattern with a gradient rectangle and adds the pattern name as text."""
        candle_indexes = pattern.get('candle_indexes')
        if candle_indexes is None:
            start_idx = pattern.get('start_index') or pattern.get('start_idx')
            end_idx = pattern.get('end_index') or pattern.get('end_idx')
            if start_idx is not None and end_idx is not None:
                candle_indexes = list(range(start_idx, end_idx + 1))
            else:
                return
        valid_indexes = [idx for idx in candle_indexes[:3] if idx in self.ohlcv_df.index]
        if not valid_indexes:
            return
        times = [self.ohlcv_df.loc[idx, 'timestamp'] for idx in valid_indexes]
        lows = [self.ohlcv_df.loc[idx, 'low'] for idx in valid_indexes]
        highs = [self.ohlcv_df.loc[idx, 'high'] for idx in valid_indexes]
        x0, x1 = min(times), max(times)
        y0, y1 = min(lows), max(highs)
        # Draw semi-transparent rectangle (blue)
        self.fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,0,255,0.25)",
            row=1, col=1
        )
        # Add pattern name as text using Scatter, just above the rectangle
        center_idx = valid_indexes[len(valid_indexes) // 2]
        center_time = self.ohlcv_df.loc[center_idx, 'timestamp']
        price_range = float(self.ohlcv_df['high'].max()) - float(self.ohlcv_df['low'].min())
        offset = price_range * 0.01
        y_text = max(highs) + offset
        y_text = self._get_label_y(center_time, y_text, offset)
        label = pattern.get('pattern', '').replace('_', ' ').title() + self._get_pattern_arrow(pattern.get('pattern', ''))
        self.fig.add_trace(go.Scatter(
            x=[center_time],
            y=[y_text],
            text=[label],
            mode="text",
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="top center",
            showlegend=False
        ), row=1, col=1)

    def _draw_three_outside_down(self, pattern: Dict):
        """Highlights the 3 main candles of a Three Outside Down pattern with a gradient rectangle and adds the pattern name as text."""
        candle_indexes = pattern.get('candle_indexes')
        if candle_indexes is None:
            start_idx = pattern.get('start_index') or pattern.get('start_idx')
            end_idx = pattern.get('end_index') or pattern.get('end_idx')
            if start_idx is not None and end_idx is not None:
                candle_indexes = list(range(start_idx, end_idx + 1))
            else:
                return
        valid_indexes = [idx for idx in candle_indexes[:3] if idx in self.ohlcv_df.index]
        if not valid_indexes:
            return
        times = [self.ohlcv_df.loc[idx, 'timestamp'] for idx in valid_indexes]
        lows = [self.ohlcv_df.loc[idx, 'low'] for idx in valid_indexes]
        highs = [self.ohlcv_df.loc[idx, 'high'] for idx in valid_indexes]
        x0, x1 = min(times), max(times)
        y0, y1 = min(lows), max(highs)
        # Draw semi-transparent rectangle (red)
        self.fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.25)",
            row=1, col=1
        )
        # Add pattern name as text using Scatter, just above the rectangle
        center_idx = valid_indexes[len(valid_indexes) // 2]
        center_time = self.ohlcv_df.loc[center_idx, 'timestamp']
        price_range = float(self.ohlcv_df['high'].max()) - float(self.ohlcv_df['low'].min())
        offset = price_range * 0.01
        y_text = max(highs) + offset
        y_text = self._get_label_y(center_time, y_text, offset)
        label = pattern.get('pattern', '').replace('_', ' ').title() + self._get_pattern_arrow(pattern.get('pattern', ''))
        self.fig.add_trace(go.Scatter(
            x=[center_time],
            y=[y_text],
            text=[label],
            mode="text",
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="top center",
            showlegend=False
        ), row=1, col=1)

    def _draw_three_inside_down(self, pattern: Dict):
        """Highlights the 3 main candles of a Three Inside Down pattern with a gradient rectangle and adds the pattern name as text."""
        candle_indexes = pattern.get('candle_indexes')
        if candle_indexes is None:
            start_idx = pattern.get('start_index') or pattern.get('start_idx')
            end_idx = pattern.get('end_index') or pattern.get('end_idx')
            if start_idx is not None and end_idx is not None:
                candle_indexes = list(range(start_idx, end_idx + 1))
            else:
                return
        valid_indexes = [idx for idx in candle_indexes[:3] if idx in self.ohlcv_df.index]
        if not valid_indexes:
            return
        times = [self.ohlcv_df.loc[idx, 'timestamp'] for idx in valid_indexes]
        lows = [self.ohlcv_df.loc[idx, 'low'] for idx in valid_indexes]
        highs = [self.ohlcv_df.loc[idx, 'high'] for idx in valid_indexes]
        x0, x1 = min(times), max(times)
        y0, y1 = min(lows), max(highs)
        # Draw semi-transparent rectangle (orange)
        self.fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,140,0,0.25)",
            row=1, col=1
        )
        # Add pattern name as text using Scatter, just above the rectangle
        center_idx = valid_indexes[len(valid_indexes) // 2]
        center_time = self.ohlcv_df.loc[center_idx, 'timestamp']
        price_range = float(self.ohlcv_df['high'].max()) - float(self.ohlcv_df['low'].min())
        offset = price_range * 0.01
        y_text = max(highs) + offset
        y_text = self._get_label_y(center_time, y_text, offset)
        label = pattern.get('pattern', '').replace('_', ' ').title() + self._get_pattern_arrow(pattern.get('pattern', ''))
        self.fig.add_trace(go.Scatter(
            x=[center_time],
            y=[y_text],
            text=[label],
            mode="text",
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="top center",
            showlegend=False
        ), row=1, col=1)

    def _draw_three_inside_up(self, pattern: Dict):
        """Highlights the 3 main candles of a Three Inside Up pattern with a gradient rectangle and adds the pattern name as text."""
        candle_indexes = pattern.get('candle_indexes')
        if candle_indexes is None:
            start_idx = pattern.get('start_index') or pattern.get('start_idx')
            end_idx = pattern.get('end_index') or pattern.get('end_idx')
            if start_idx is not None and end_idx is not None:
                candle_indexes = list(range(start_idx, end_idx + 1))
            else:
                return
        valid_indexes = [idx for idx in candle_indexes[:3] if idx in self.ohlcv_df.index]
        if not valid_indexes:
            return
        times = [self.ohlcv_df.loc[idx, 'timestamp'] for idx in valid_indexes]
        lows = [self.ohlcv_df.loc[idx, 'low'] for idx in valid_indexes]
        highs = [self.ohlcv_df.loc[idx, 'high'] for idx in valid_indexes]
        x0, x1 = min(times), max(times)
        y0, y1 = min(lows), max(highs)
        # Draw semi-transparent rectangle (green)
        self.fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="green", width=2),
            fillcolor="rgba(0,255,0,0.25)",
            row=1, col=1
        )
        # Add pattern name as text using Scatter, just above the rectangle
        center_idx = valid_indexes[len(valid_indexes) // 2]
        center_time = self.ohlcv_df.loc[center_idx, 'timestamp']
        price_range = float(self.ohlcv_df['high'].max()) - float(self.ohlcv_df['low'].min())
        offset = price_range * 0.01
        y_text = max(highs) + offset
        y_text = self._get_label_y(center_time, y_text, offset)
        label = pattern.get('pattern', '').replace('_', ' ').title() + self._get_pattern_arrow(pattern.get('pattern', ''))
        self.fig.add_trace(go.Scatter(
            x=[center_time],
            y=[y_text],
            text=[label],
            mode="text",
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="top center",
            showlegend=False
        ), row=1, col=1)

    def _draw_cup_and_handle(self, pattern: Dict):
        """Draws overlays for a Cup and Handle pattern using key_levels['points'] from the pattern output, with classic visuals and a single target."""
        key_levels = pattern.get('key_levels', {})
        points = key_levels.get('points', {})
        # Patch: support both start_idx/end_idx and start_index/end_index
        start_idx = pattern.get('start_idx')
        if start_idx is None:
            start_idx = pattern.get('start_index')
        end_idx = pattern.get('end_idx')
        if end_idx is None:
            end_idx = pattern.get('end_index')
        confidence = pattern.get('confidence', None)
        if not points or start_idx is None or end_idx is None:
            return
        df = self.ohlcv_df
        # Compute absolute indices for each key point
        cup_start_idx = start_idx + points['cup_start']['index']
        cup_bottom_idx = start_idx + points['cup_bottom']['index']
        cup_end_idx = start_idx + points['cup_end']['index']
        handle_start_idx = start_idx + points['handle_start']['index']
        handle_end_idx = start_idx + points['handle_end']['index']
        # Get timestamps and prices
        cup_start_time = df.loc[cup_start_idx, 'timestamp']
        cup_bottom_time = df.loc[cup_bottom_idx, 'timestamp']
        cup_end_time = df.loc[cup_end_idx, 'timestamp']
        handle_start_time = df.loc[handle_start_idx, 'timestamp']
        handle_end_time = df.loc[handle_end_idx, 'timestamp']
        cup_start_price = df.loc[cup_start_idx, 'close']
        cup_bottom_price = df.loc[cup_bottom_idx, 'close']
        cup_end_price = df.loc[cup_end_idx, 'close']
        handle_start_price = df.loc[handle_start_idx, 'close']
        handle_end_price = df.loc[handle_end_idx, 'close']
        # --- Draw cup arc as a green-to-transparent gradient ---
        import numpy as np
        cup_indices = [cup_start_idx, cup_bottom_idx, cup_end_idx]
        cup_times = [cup_start_time, cup_bottom_time, cup_end_time]
        cup_prices = [cup_start_price, cup_bottom_price, cup_end_price]
        x_vals = np.array([(t - cup_start_time).total_seconds() for t in cup_times])
        y_vals = np.array(cup_prices)
        if len(np.unique(x_vals)) >= 3:
            coeffs = np.polyfit(x_vals, y_vals, 2)
            interp_x = np.linspace(x_vals[0], x_vals[-1], 50)
            interp_y = np.polyval(coeffs, interp_x)
            interp_times = [cup_start_time + pd.Timedelta(seconds=float(s)) for s in interp_x]
            # Draw more lines for a smoother gradient effect
            for i, (alpha, width) in enumerate(zip(np.linspace(0.12, 1.0, 12), np.linspace(16, 2, 12))):
                self.fig.add_trace(go.Scatter(
                    x=interp_times,
                    y=interp_y,
                    mode='lines',
                    line=dict(color=f'rgba(0,255,0,{alpha:.2f})', width=width),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)
        # --- Draw handle as a downward-sloping regression line ---
        handle_indices = list(range(handle_start_idx, handle_end_idx + 1))
        if len(handle_indices) >= 2:
            handle_times = df.loc[handle_indices, 'timestamp']
            handle_prices = df.loc[handle_indices, 'close']
            x = np.arange(len(handle_prices))
            coeffs = np.polyfit(x, handle_prices, 1)
            regression = np.polyval(coeffs, x)
            self.fig.add_trace(go.Scatter(
                x=handle_times,
                y=regression,
                mode='lines',
                line=dict(color='rgba(0,120,255,1)', width=4, dash='dash'),
                name='Handle (Downward)',
                showlegend=False,
                hoverinfo='text',
                text=['Handle (Downward)']*len(handle_times)
            ), row=1, col=1)
        # --- Annotate key points ---
        for label, idx, t, y in [
            ("Cup Start", cup_start_idx, cup_start_time, cup_start_price),
            ("Cup Bottom", cup_bottom_idx, cup_bottom_time, cup_bottom_price),
            ("Cup End", cup_end_idx, cup_end_time, cup_end_price),
            ("Handle Start", handle_start_idx, handle_start_time, handle_start_price),
            ("Handle End", handle_end_idx, handle_end_time, handle_end_price)
        ]:
            self.fig.add_annotation(
                x=t, y=y, text=label, showarrow=True, arrowhead=2,
                bgcolor='rgba(0,0,0,0.7)', font=dict(color='white', size=10, family='Arial Black'),
                yanchor="bottom", xanchor="left"
            )
        # --- Draw single target from handle_end (breakout) ---
        cup_height = max(cup_start_price, cup_end_price) - cup_bottom_price
        target_price_breakout = handle_end_price + cup_height
        self.fig.add_shape(type="line",
            x0=handle_end_time, y0=handle_end_price,
            x1=handle_end_time, y1=target_price_breakout,
            line=dict(color='lime', width=4, dash='dot'), row=1, col=1
        )
        self.fig.add_annotation(
            x=handle_end_time, y=target_price_breakout,
            text=f"Target ({int(confidence*100)}%)" if confidence is not None else "Target",
            showarrow=False,
            bgcolor='lime', font=dict(color='black', size=14, family='Arial Black'),
            yanchor="bottom", xanchor="left"
        )

    def _draw_support_resistance(self):
        """Draws support and resistance levels as horizontal lines with labels."""
        # Only proceed if self.analysis is a dict and has 'market_context'
        if not isinstance(self.analysis, dict):
            return
        context = self.analysis.get('market_context', {})
        support_levels = context.get('support_levels', [])
        resistance_levels = context.get('resistance_levels', [])
        logger.info(f"Support levels: {support_levels}")
        logger.info(f"Resistance levels: {resistance_levels}")
        # Draw support lines
        for i, level in enumerate(support_levels):
            self.fig.add_shape(
                type="line",
                x0=self.ohlcv_df['timestamp'].min(), x1=self.ohlcv_df['timestamp'].max(),
                y0=level, y1=level,
                line=dict(color=self.config['colors']['support_line'], width=2, dash="dot"),
                xref="x1", yref="y1"
            )
            self.fig.add_annotation(
                x=self.ohlcv_df['timestamp'].max(), y=level,
                text=f"Support {i+1}", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(color=self.config['colors']['support_line'], size=10),
                xref="x1", yref="y1"
            )
        # Draw resistance lines
        for i, level in enumerate(resistance_levels):
            self.fig.add_shape(
                type="line",
                x0=self.ohlcv_df['timestamp'].min(), x1=self.ohlcv_df['timestamp'].max(),
                y0=level, y1=level,
                line=dict(color=self.config['colors']['resistance_line'], width=2, dash="dot"),
                xref="x1", yref="y1"
            )
            self.fig.add_annotation(
                x=self.ohlcv_df['timestamp'].max(), y=level,
                text=f"Resistance {i+1}", showarrow=False,
                xanchor="right", yanchor="top",
                font=dict(color=self.config['colors']['resistance_line'], size=10),
                xref="x1", yref="y1"
            )


    def create_chart(self, output_type: str = 'image') -> bytes | str:
        """Generates the chart with all overlays."""
        self._create_figure()
        # self._draw_market_context()   # Draw overlays first (REMOVED)
        # self._draw_support_resistance()  # Draw support/resistance lines
        self._draw_patterns()         # Draw pattern overlays next

        # Detailed debug logging for OHLCV DataFrame
        logger.info(f"OHLCV dtypes: {self.ohlcv_df.dtypes}")
        logger.info(f"OHLCV nulls: {self.ohlcv_df.isnull().sum()}")
        logger.info(f"OHLCV infs: {((self.ohlcv_df[['open','high','low','close','volume']] == float('inf')).sum())}")
        logger.info(f"OHLCV -len- open: {len(self.ohlcv_df['open'])}, high: {len(self.ohlcv_df['high'])}, low: {len(self.ohlcv_df['low'])}, close: {len(self.ohlcv_df['close'])}, timestamp: {len(self.ohlcv_df['timestamp'])}")
        logger.info(f"OHLCV sample: {self.ohlcv_df[['timestamp','open','high','low','close']].head(10)}")
        # Log timestamp diffs and duplicates
        logger.info(f"Timestamp diffs: {self.ohlcv_df['timestamp'].diff().dropna().unique()}")
        logger.info(f"Timestamp duplicates: {self.ohlcv_df['timestamp'].duplicated().sum()}")

        self._add_candlestick_trace() # Draw candlesticks on top
        self._add_volume_trace()

        # Log the first and last few rows of the OHLCV DataFrame for debugging
        logger.info(f"OHLCV head: {self.ohlcv_df[['timestamp', 'open', 'high', 'low', 'close']].head()}")
        logger.info(f"OHLCV tail: {self.ohlcv_df[['timestamp', 'open', 'high', 'low', 'close']].tail()}")

        if output_type == 'image':
            return self.fig.to_image(format="png", width=1600, height=900, scale=2)
        elif output_type == 'html':
            return self.fig.to_html()
        raise ValueError("Invalid output_type. Choose 'image' or 'html'.")