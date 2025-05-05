# src/core/domain/entities/PatternAnalysisEntity.py

from pydantic import BaseModel
from typing import List, Optional

class Coordinate(BaseModel):
    time: str
    price: float

class PatternMatch(BaseModel):
    type: str
    confidence: float
    start_time: str
    end_time: str
    coordinates: List[Coordinate]

class EntryExitPoint(BaseModel):
    suggested_price: float
    time_estimate: str
    confidence: float

class TrendPrediction(BaseModel):
    direction: str
    next_24h_target: float
    confidence: float

class SupportResistance(BaseModel):
    support_levels: List[float]
    resistance_levels: List[float]

class Volatility(BaseModel):
    average_range: float
    volatility_score: float

class PatternAnalysisResponse(BaseModel):
    pattern_matches: List[PatternMatch]
    entry_point: Optional[EntryExitPoint]
    exit_point: Optional[EntryExitPoint]
    trend_prediction: Optional[TrendPrediction]
    support_resistance: Optional[SupportResistance]
    volatility: Optional[Volatility]
