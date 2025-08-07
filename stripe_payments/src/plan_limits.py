"""
File: plan_limits.py
"""
PLAN_LIMITS = {
    "free": {
        "price_alerts_limit": 1,
        "pattern_detection_limit": 1,
        "watchlist_limit": 1,
        "market_analysis_limit": 3,
        "trendline_analysis_limit": 3,  # New limit for trendlines
        "sr_analysis_limit": 8,       # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "test_drive": {
        "price_alerts_limit": 5,
        "pattern_detection_limit": 2,
        "watchlist_limit": 1,
        "market_analysis_limit": 7,
        "trendline_analysis_limit": 7,  # New limit for trendlines
        "sr_analysis_limit": 12,      # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": False,
        "video_download_limit": 1
    },
    "starter_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 7,
        "watchlist_limit": 3,
        "market_analysis_limit": 49,
        "trendline_analysis_limit": 49, # New limit for trendlines
        "sr_analysis_limit": 54,      # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "starter_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 60,
        "watchlist_limit": 6,
        "market_analysis_limit": 300,
        "trendline_analysis_limit": 300, # New limit for trendlines
        "sr_analysis_limit": 305,      # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "pro_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "trendline_analysis_limit": -1, # New limit for trendlines
        "sr_analysis_limit": -1,      # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": True,
        "video_download_limit": -1
    },
    "pro_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "trendline_analysis_limit": -1, # New limit for trendlines
        "sr_analysis_limit": -1,      # New limit for S/R (market_analysis_limit + 5)
        "journaling_enabled": True,
        "video_download_limit": -1
    }
}
