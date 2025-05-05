from pydantic import BaseModel, Field
from pydantic import BaseModel
from typing import Optional, List
import numpy as np

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PatternResult(BaseModel):
    pattern: str
    start_idx: int
    end_idx: int
    confidence: float
    key_levels: Dict[str, float]
    detection_time: str

class MarketContextResult(BaseModel):
    scenario: str
    volatility: float
    trend_strength: float
    volume_profile: str
    active_patterns: List[PatternResult]
    support_levels: List[float]
    resistance_levels: List[float]

class ForecastResultPydantic(BaseModel):
    direction: str
    confidence: float
    timeframe: str
    expected_volatility: str
    scenario_continuation_probability: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    scenario_transitions: Optional[Dict[str, float]] = None

class AnalysisResult(BaseModel):
    patterns: List[PatternResult]
    market_context: MarketContextResult
    forecast: ForecastResultPydantic
    analysis_timestamp: str