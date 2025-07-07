import asyncio
import json
from core.services.chart_engine import ChartEngine
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI

# --- User parameters ---
symbol = "SOLUSDT"
interval = "1h"
timeframe = "1w"

async def main():
    # Fetch OHLCV data
    ohlcv_data = await get_ohlcv_from_db(symbol, interval, timeframe)
    
    # Debug: Print data structure
    print(f"OHLCV Data type: {type(ohlcv_data)}")
    print(f"OHLCV Data keys: {list(ohlcv_data.keys()) if isinstance(ohlcv_data, dict) else 'Not a dict'}")
    
    # Analyze patterns
    pattern_api = PatternAPI(interval=interval)
    analysis_result = await pattern_api.analyze_market_data(ohlcv=ohlcv_data)
    
    # Debug: Print analysis structure
    print(f"Analysis keys: {list(analysis_result.keys())}")
    print(f"Patterns found: {len(analysis_result.get('patterns', []))}")
    
    # Save data for debugging
    with open("ohlcv_data.json", "w") as f:
        json.dump(ohlcv_data, f, indent=4, default=str)

    with open("analysis_result.json", "w") as f:
        json.dump(analysis_result, f, indent=4, default=str)

    # FIX: Correct parameter order - ohlcv_data first, then analysis_data
    chart_gen = ChartEngine(ohlcv_data=ohlcv_data, analysis_data=analysis_result)
    image_bytes = chart_gen.create_chart(output_type="image")
    
    with open("sample_chart.png", "wb") as f:
        f.write(image_bytes)
    print("Chart saved as sample_chart.png")

if __name__ == "__main__":
    asyncio.run(main())
