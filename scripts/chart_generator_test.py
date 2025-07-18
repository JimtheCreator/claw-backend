# chart_generator_test.py
import asyncio
import json
import numpy as np
from core.services.pattern_analysis_engine import PatternAnalysisEngine
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.use_cases.market_analysis.detect_patterns_engine.harmonic_patterns import find_significant_swings # Import for diagnostics
from datetime import datetime
import pandas as pd
from core.engines.chart_engine import ChartEngine

# --- User Parameters ---
symbol = "BTCUSDT"
interval = "5m"
timeframe = "24h"  # Fetch more data for better structural analysis
categories_to_run = ["candlestick"] # Run all pattern types
MINIMUM_CONFIDENCE = 0.5 # A reasonable threshold

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Any datetime-like object
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

async def main():
    ohlcv_data = await get_ohlcv_from_db(symbol, interval, timeframe)
    # if not ohlcv_data or len(['close']) < 50:
    #     print("❌ Could not fetch sufficient OHLCV data.")
    #     return

    print(f"✅ Fetched {len(ohlcv_data['close'])} candles for {symbol} on {interval} interval.")
    print("-" * 30)

    # --- DIAGNOSTIC STEP ---
    print("🕵️  Running a diagnostic on the first data segment...")
    diagnostic_segment = {k: v[:250] for k,v in ohlcv_data.items()}
    swings = find_significant_swings(diagnostic_segment)
    print(f"Found {len(swings)} swing points in the first 250 candles. This should be greater than 5.")
    if len(swings) > 0:
        print("Example swings:", swings[:5])
    print("-" * 30)
    # --- END DIAGNOSTIC ---

    analysis_engine = PatternAnalysisEngine(ohlcv=ohlcv_data)
    
    print(f"⚙️  Scanning entire chart for patterns with minimum confidence of {MINIMUM_CONFIDENCE}...")
    analysis_data = await analysis_engine.scan_for_all_patterns(
        categories=categories_to_run,
        min_confidence=MINIMUM_CONFIDENCE
    )
    
    print(f"\n🎉 Found {len(analysis_data)} significant, non-overlapping patterns:")
    print(json.dumps(analysis_data, indent=2, cls=NpEncoder))

    with open("final_pattern_results.json", "w") as f:
        json.dump(analysis_data, f, indent=4, cls=NpEncoder)
    
    print(f"\n💾 Results saved to final_pattern_results.json")

    # FIX: Correct parameter order - ohlcv_data first, then analysis_data
    chart_gen = ChartEngine(ohlcv_data=ohlcv_data, analysis_data=analysis_data)
    image_bytes = chart_gen.create_chart(output_type="image")
    
    # Ensure only bytes are written to the PNG file
    if isinstance(image_bytes, str):
        image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
    with open("sample_chart.png", "wb") as f:
        f.write(image_bytes)
    print("Chart saved as sample_chart.png")

if __name__ == "__main__":
    asyncio.run(main())
