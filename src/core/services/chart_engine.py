import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import pandas as pd
from common.logger import logger

class ChartEngine:
    def __init__(self, ohlcv_data: Dict[str, list], analysis_data: Dict[str, Any], config: Optional[Dict] = None):
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
        # Ensure the timestamp column is in datetime format for the x-axis
        self.ohlcv_df['timestamp'] = pd.to_datetime(self.ohlcv_df['timestamp'])
        self.ohlcv_df = self.ohlcv_df.sort_values('timestamp')
        self.ohlcv_df = self.ohlcv_df.drop_duplicates(subset='timestamp')
        self.ohlcv_df = self.ohlcv_df.reset_index(drop=True)  # Ensure sequential index for Plotly
        # Convert to Python datetime objects for Plotly compatibility
        self.ohlcv_df['timestamp'] = self.ohlcv_df['timestamp'].apply(lambda x: x.to_pydatetime())

        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Main Plotly Figure object
        self.fig: Optional[go.Figure] = None

        # Map pattern names to their drawing methods
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
        min_price = float(self.ohlcv_df[['low']].min())
        max_price = float(self.ohlcv_df[['high']].max())
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

    def _draw_patterns(self):
        """Iterates through patterns and calls the correct drawing function."""
        if 'patterns' not in self.analysis: return
        for pattern in self.analysis['patterns']:
            pattern_name = pattern.get('pattern')
            if pattern_name in self.pattern_drawing_map:
                self.pattern_drawing_map[pattern_name](pattern)

    def _draw_market_context(self):
        """Draws supply/demand zones and support/resistance levels."""
        context = self.analysis.get('market_context', {})
        for zone_type, zones in [('supply', 'supply_zones'), ('demand', 'demand_zones')]:
            if zones in context:
                for zone in context[zones]:
                    color = self.config['colors'][f'{zone_type}_zone_fill']
                    self.fig.add_hrect(
                        y0=zone['bottom'], y1=zone['top'],
                        fillcolor=color, layer="below", line_width=0, # Changed layer to "below"
                        annotation_text=f"{zone_type.capitalize()} Zone ({zone.get('id', '')})",
                        annotation_position="top right"
                    )

    def _find_peak_valley_coords(self, key_levels, candle_indexes):
        """Helper to find x,y coordinates for pattern key points."""
        coords = {}
        
        valid_indexes = [idx for idx in candle_indexes if idx in self.ohlcv_df.index]
        if not valid_indexes:
             print(f"Warning: No valid candle data for pattern in index range.")
             return None

        first_peak_df = self.ohlcv_df.loc[valid_indexes]
        if first_peak_df.empty:
            return None # Cannot proceed if there's no data

        coords['p1_idx'] = first_peak_df['high'].idxmax()
        coords['p1_x'] = self.ohlcv_df.loc[coords['p1_idx'], 'timestamp']
        coords['p1_y'] = key_levels['first_peak']

        second_peak_df = first_peak_df.drop(coords['p1_idx'])
        if second_peak_df.empty:
            return None # Cannot find a second peak

        coords['p2_idx'] = second_peak_df['high'].idxmax()
        coords['p2_x'] = self.ohlcv_df.loc[coords['p2_idx'], 'timestamp']
        coords['p2_y'] = key_levels['second_peak']

        start, end = sorted([coords['p1_idx'], coords['p2_idx']])
        valley_df = self.ohlcv_df.loc[start:end]
        coords['v_idx'] = valley_df['low'].idxmin()
        coords['v_x'] = self.ohlcv_df.loc[coords['v_idx'], 'timestamp']
        coords['v_y'] = key_levels['valley']
        
        return coords


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
        print(f"Drawing Double Bottom from index {pattern['start_idx']} to {pattern['end_idx']}")


    def _draw_triple_top(self, pattern: Dict):
        print(f"Drawing Triple Top from index {pattern['start_idx']} to {pattern['end_idx']}")


    def _draw_triple_bottom(self, pattern: Dict):
        print(f"Drawing Triple Bottom from index {pattern['start_idx']} to {pattern['end_idx']}")


    def _draw_channel(self, pattern: Dict):
        print(f"Drawing Channel from index {pattern['start_idx']} to {pattern['end_idx']}")
        
        
    def _draw_triangle(self, pattern: Dict):
        print(f"Drawing Triangle from index {pattern['start_idx']} to {pattern['end_idx']}")


    def create_chart(self, output_type: str = 'image') -> bytes | str:
        """Generates the chart with all overlays."""
        self._create_figure()
        self._draw_market_context()   # Draw overlays first
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