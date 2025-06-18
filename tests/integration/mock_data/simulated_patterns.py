# This data represents the last 100 candles, with the pattern at the very end.
def get_mock_bearish_engulfing_klines():
    """
    Returns a list of 100 klines where the last two form a bearish engulfing pattern.
    Format: [timestamp, open, high, low, close, volume]
    
    Bearish Engulfing Pattern Requirements:
    1. First candle: Small green/bullish candle (close > open)
    2. Second candle: Large red/bearish candle (close < open)
    3. Second candle's open > First candle's close (gap up or engulfing open)
    4. Second candle's close < First candle's open (complete engulfment)
    5. Second candle's body completely engulfs the first candle's body
    """
    # 98 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(98)]
    
    # The last two candles form the bearish engulfing pattern:
    # Candle 99: A small green candle (close > open)
    klines.append([98, 100.0, 103.0, 99.0, 102.5, 1000])  # Green: open=100.0, close=102.5
    
    # Candle 100: A large red candle that completely engulfs the previous green candle
    # For bearish engulfing: open > prev_close AND close < prev_open
    klines.append([99, 103.0, 104.0, 98.0, 99.0, 1500])   # Red: open=103.0, close=99.0
    
    return klines

# Verification of the pattern:
# Previous candle: open=100.0, high=103.0, low=99.0, close=102.5 (GREEN: 102.5 > 100.0)
# Current candle:  open=103.0, high=104.0, low=98.0, close=99.0  (RED: 99.0 < 103.0)
# 
# Bearish Engulfing Validation:
# ✓ prev_candle[4] > prev_candle[1]  # 102.5 > 100.0 (previous was green)
# ✓ last_candle[4] < last_candle[1]  # 99.0 < 103.0 (current is red)  
# ✓ last_candle[1] > prev_candle[4]  # 103.0 > 102.5 (current open > prev close)
# ✓ last_candle[4] < prev_candle[1]  # 99.0 < 100.0 (current close < prev open)
#
# The red candle completely engulfs the green candle's body


def get_mock_bullish_engulfing_klines():
    """
    Returns a list of 100 klines where the last two form a bullish engulfing pattern.
    """
    # 98 dummy historical candles in downtrend
    klines = [[i, 120 - i*0.1, 125 - i*0.1, 115 - i*0.1, 120 - i*0.1, 1000] for i in range(98)]
    
    # The last two candles form the pattern:
    # Candle 99: A small red candle (close < open)
    klines.append([98, 102.5, 103.0, 99.0, 100.0, 1000])
    # Candle 100: A large green candle (close > open) that engulfs the previous one
    klines.append([99, 99.5, 105.0, 98.0, 104.0, 1500])
    
    return klines


def get_mock_hammer_klines():
    """
    Returns a list of 100 klines where the last candle forms a hammer pattern.
    Hammer: Small body at top, long lower shadow, little/no upper shadow
    """
    # 98 dummy historical candles in downtrend
    klines = [[i, 120 - i*0.2, 125 - i*0.2, 115 - i*0.2, 118 - i*0.2, 1000] for i in range(98)]
    
    # Second to last: normal red candle
    klines.append([98, 102.0, 103.0, 98.0, 99.0, 1000])
    # Last candle: Hammer - open around 98, close around 99, high around 99.5, low much lower at 94
    klines.append([99, 98.0, 99.5, 94.0, 99.0, 1200])
    
    return klines


def get_mock_shooting_star_klines():
    """
    Returns a list of 100 klines where the last candle forms a shooting star pattern.
    Shooting Star: Small body at bottom, long upper shadow, little/no lower shadow
    """
    # 98 dummy historical candles in uptrend
    klines = [[i, 80 + i*0.2, 85 + i*0.2, 75 + i*0.2, 82 + i*0.2, 1000] for i in range(98)]
    
    # Second to last: normal green candle
    klines.append([98, 98.0, 102.0, 97.0, 101.0, 1000])
    # Last candle: Shooting Star - open around 101, close around 100, low around 99.5, high much higher at 105
    klines.append([99, 101.0, 105.0, 99.5, 100.0, 1200])
    
    return klines


def get_mock_doji_klines():
    """
    Returns a list of 100 klines where the last candle forms a doji pattern.
    Doji: Open and close are nearly equal, creating a cross-like pattern
    """
    # 98 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(98)]
    
    # Second to last: normal candle
    klines.append([98, 100.0, 103.0, 98.0, 102.0, 1000])
    # Last candle: Doji - open and close very close, with shadows
    klines.append([99, 102.0, 104.0, 99.0, 102.1, 800])
    
    return klines


def get_mock_morning_star_klines():
    """
    Returns a list of 100 klines where the last three form a morning star pattern.
    Morning Star: Large red candle, small candle (gap down), large green candle (gap up)
    """
    # 97 dummy historical candles in downtrend
    klines = [[i, 120 - i*0.15, 125 - i*0.15, 115 - i*0.15, 118 - i*0.15, 1000] for i in range(97)]
    
    # The last three candles form the pattern:
    # Candle 98: Large red candle
    klines.append([97, 105.0, 106.0, 100.0, 100.5, 1500])
    # Candle 99: Small candle (doji-like) with gap down
    klines.append([98, 99.0, 100.0, 98.0, 99.2, 800])
    # Candle 100: Large green candle with gap up
    klines.append([99, 100.5, 105.0, 100.0, 104.5, 1600])
    
    return klines


def get_mock_evening_star_klines():
    """
    Returns a list of 100 klines where the last three form an evening star pattern.
    Evening Star: Large green candle, small candle (gap up), large red candle (gap down)
    """
    # 97 dummy historical candles in uptrend
    klines = [[i, 80 + i*0.15, 85 + i*0.15, 75 + i*0.15, 82 + i*0.15, 1000] for i in range(97)]
    
    # The last three candles form the pattern:
    # Candle 98: Large green candle
    klines.append([97, 95.0, 100.0, 94.0, 99.5, 1500])
    # Candle 99: Small candle (doji-like) with gap up
    klines.append([98, 101.0, 102.0, 100.0, 100.8, 800])
    # Candle 100: Large red candle with gap down
    klines.append([99, 99.5, 100.0, 95.0, 95.5, 1600])
    
    return klines


def get_mock_head_and_shoulders_klines():
    """
    Returns a list of 100 klines forming a head and shoulders pattern.
    Pattern: Left shoulder, head (higher), right shoulder, neckline break
    """
    # 80 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(80)]
    
    # Left shoulder formation (candles 80-85)
    shoulder_left = [
        [80, 100, 102, 99, 101, 1100],
        [81, 101, 104, 100, 103, 1200],
        [82, 103, 106, 102, 105, 1300],  # Peak of left shoulder
        [83, 105, 106, 102, 103, 1100],
        [84, 103, 104, 100, 101, 1000],  # Trough
        [85, 101, 102, 99, 100, 900]
    ]
    
    # Head formation (candles 86-91)
    head = [
        [86, 100, 103, 99, 102, 1200],
        [87, 102, 106, 101, 105, 1400],
        [88, 105, 110, 104, 108, 1600],  # Peak of head (highest point)
        [89, 108, 109, 105, 106, 1300],
        [90, 106, 107, 102, 103, 1100],
        [91, 103, 104, 100, 101, 1000]   # Trough
    ]
    
    # Right shoulder formation (candles 92-97)
    shoulder_right = [
        [92, 101, 103, 100, 102, 1100],
        [93, 102, 105, 101, 104, 1200],
        [94, 104, 106, 103, 105, 1100],  # Peak of right shoulder (lower than head)
        [95, 105, 106, 103, 104, 1000],
        [96, 104, 105, 101, 102, 900],
        [97, 102, 103, 100, 101, 800]
    ]
    
    # Neckline break (candles 98-99)
    neckline_break = [
        [98, 101, 102, 98, 99, 1200],    # Breaking below neckline
        [99, 99, 100, 96, 97, 1500]     # Confirmation of break
    ]
    
    klines.extend(shoulder_left + head + shoulder_right + neckline_break)
    return klines


def get_mock_double_top_klines():
    """
    Returns a list of 100 klines forming a double top pattern.
    Pattern: Peak, trough, similar peak, breakdown
    """
    # 85 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(85)]
    
    # First peak formation (candles 85-88)
    first_peak = [
        [85, 100, 103, 99, 102, 1200],
        [86, 102, 106, 101, 105, 1400],
        [87, 105, 108, 104, 107, 1300],  # First peak
        [88, 107, 108, 104, 105, 1100]
    ]
    
    # Trough formation (candles 89-92)
    trough = [
        [89, 105, 106, 102, 103, 1000],
        [90, 103, 104, 100, 101, 900],
        [91, 101, 102, 98, 99, 800],     # Trough
        [92, 99, 101, 98, 100, 900]
    ]
    
    # Second peak formation (candles 93-96)
    second_peak = [
        [93, 100, 103, 99, 102, 1100],
        [94, 102, 105, 101, 104, 1300],
        [95, 104, 107, 103, 106, 1200],  # Second peak (similar to first)
        [96, 106, 107, 103, 104, 1000]
    ]
    
    # Breakdown (candles 97-99)
    breakdown = [
        [97, 104, 105, 101, 102, 1100],
        [98, 102, 103, 98, 99, 1300],    # Breaking below trough
        [99, 99, 100, 95, 96, 1500]     # Confirmation of breakdown
    ]
    
    klines.extend(first_peak + trough + second_peak + breakdown)
    return klines


def get_mock_triangle_klines():
    """
    Returns a list of 100 klines forming a symmetrical triangle pattern.
    Pattern: Converging trendlines with decreasing volatility
    """
    # 70 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(70)]
    
    # Triangle formation with converging highs and lows
    triangle_data = []
    base_price = 100
    high_range = 8  # Starting range for highs
    low_range = 8   # Starting range for lows
    
    for i in range(30):  # 30 candles forming the triangle
        # Gradually decrease the range to create convergence
        current_high_range = high_range * (1 - i/40)  # Converging highs
        current_low_range = low_range * (1 - i/40)    # Converging lows
        
        # Alternate between higher and lower closes to create the triangle
        if i % 2 == 0:
            # Higher close (touching upper trendline)
            open_price = base_price - current_low_range/2
            close_price = base_price + current_high_range/3
            high_price = base_price + current_high_range
            low_price = base_price - current_low_range/2
        else:
            # Lower close (touching lower trendline)
            open_price = base_price + current_high_range/2
            close_price = base_price - current_low_range/3
            high_price = base_price + current_high_range/2
            low_price = base_price - current_low_range
        
        triangle_data.append([70 + i, open_price, high_price, low_price, close_price, 1000 - i*10])
    
    klines.extend(triangle_data)
    return klines


def get_mock_cup_and_handle_klines():
    """
    Returns a list of 100 klines forming a cup and handle pattern.
    Pattern: Rounded bottom (cup) followed by small consolidation (handle)
    """
    # 40 dummy historical candles
    klines = [[i, 100, 105, 95, 100, 1000] for i in range(40)]
    
    # Cup formation - rounded bottom over 50 candles
    cup_data = []
    for i in range(50):
        # Create a U-shaped curve using sine function
        angle = i * 3.14159 / 49  # 0 to π over 50 candles
        depth = 15  # Maximum depth of the cup
        base_price = 100
        
        # U-shaped price movement
        cup_bottom = base_price - depth * abs(0.5 - i/49) * 2
        
        open_price = cup_bottom + (i % 3 - 1) * 0.5
        close_price = cup_bottom + (i % 3) * 0.5
        high_price = max(open_price, close_price) + 1
        low_price = min(open_price, close_price) - 1
        
        cup_data.append([40 + i, open_price, high_price, low_price, close_price, 1200 - i*5])
    
    # Handle formation - small consolidation near the rim
    handle_data = []
    handle_base = 98  # Slightly below the rim
    for i in range(10):
        # Small sideways movement with slight downward bias
        open_price = handle_base - i*0.3
        close_price = handle_base - i*0.3 + (i % 2) * 0.5
        high_price = max(open_price, close_price) + 0.5
        low_price = min(open_price, close_price) - 0.5
        
        handle_data.append([90 + i, open_price, high_price, low_price, close_price, 800 - i*20])
    
    klines.extend(cup_data + handle_data)
    return klines


# Utility function to get pattern by name
def get_pattern_klines(pattern_name):
    """
    Returns klines for the specified pattern name.
    """
    pattern_functions = {
        'bearish_engulfing': get_mock_bearish_engulfing_klines,
        'bullish_engulfing': get_mock_bullish_engulfing_klines,
        'hammer': get_mock_hammer_klines,
        'bearish_shooting_star': get_mock_shooting_star_klines,
        'bullish_shooting_star': get_mock_shooting_star_klines,  # Same as shooting star
        'standard_doji': get_mock_doji_klines,
        'dragonfly_doji': get_mock_doji_klines,  # Variation of doji
        'gravestone_doji': get_mock_doji_klines,  # Variation of doji
        'morning_star': get_mock_morning_star_klines,
        'evening_star': get_mock_evening_star_klines,
        'bearish_head_and_shoulders': get_mock_head_and_shoulders_klines,
        'double_top': get_mock_double_top_klines,
        'symmetrical_triangle': get_mock_triangle_klines,
        'cup_and_handle': get_mock_cup_and_handle_klines,
    }
    
    if pattern_name in pattern_functions:
        return pattern_functions[pattern_name]()
    else:
        # Return default bearish engulfing if pattern not found
        print(f"Warning: Pattern '{pattern_name}' not implemented, using bearish_engulfing as default")
        return get_mock_bearish_engulfing_klines()


# Quick test function to visualize the patterns
def visualize_pattern_summary(pattern_name):
    """
    Provides a quick summary of the last few candles for the given pattern.
    """
    klines = get_pattern_klines(pattern_name)
    print(f"\n=== {pattern_name.upper()} PATTERN ===")
    print("Last 5 candles (timestamp, open, high, low, close, volume):")
    for i, candle in enumerate(klines[-5:]):
        print(f"  {i+96}: {candle}")
    
    # Calculate some basic metrics
    last_candle = klines[-1]
    prev_candle = klines[-2]
    
    body_size = abs(last_candle[4] - last_candle[1])  # close - open
    upper_shadow = last_candle[2] - max(last_candle[1], last_candle[4])  # high - max(open,close)
    lower_shadow = min(last_candle[1], last_candle[4]) - last_candle[3]  # min(open,close) - low
    
    print(f"Last candle analysis:")
    print(f"  Body size: {body_size:.2f}")
    print(f"  Upper shadow: {upper_shadow:.2f}")
    print(f"  Lower shadow: {lower_shadow:.2f}")
    print(f"  Direction: {'Bullish' if last_candle[4] > last_candle[1] else 'Bearish'}")


if __name__ == "__main__":
    # Test a few patterns
    test_patterns = [
        'bearish_engulfing', 
        'bullish_engulfing', 
        'hammer', 
        'morning_star', 
        'head_and_shoulders',
        'cup_and_handle'
    ]
    
    for pattern in test_patterns:
        visualize_pattern_summary(pattern)