import asyncio
from core.services.chart_generator import ChartGenerator
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI

# --- User parameters ---
symbol = "BTCUSDT"  # Change as needed
interval = "1d"     # Change as needed
timeframe = "8w"   # Change as needed (e.g., '30d', '1w', '90d')

async def main():
    # Fetch real OHLCV data
    ohlcv_data = await get_ohlcv_from_db(symbol, interval, timeframe)
    # Analyze patterns using the same logic as the endpoint
    pattern_api = PatternAPI(interval=interval)
    analysis_result = await pattern_api.analyze_market_data(ohlcv=ohlcv_data)
    # Use the result as analysis_data for the chart
    chart_gen = ChartGenerator(analysis_data=analysis_result, ohlcv_data=ohlcv_data)
    image_bytes = chart_gen.create_chart_image()
    with open("test_chart_output2.png", "wb") as f:
        f.write(image_bytes)
    print("Chart saved as test_chart_output2.png")

if __name__ == "__main__":
    asyncio.run(main()) 