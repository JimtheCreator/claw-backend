from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime

# Define DemandZone model for demand_zones field
class DemandZone(BaseModel):
    id: str
    bottom: float
    top: float
    strength: float
    avg_volume_at_formation: float
    touch_count: int
    last_timestamp: str
    avg_price: float

# Define SupplyZone model for supply_zones field
class SupplyZone(BaseModel):
    id: str
    bottom: float
    top: float
    strength: float
    touches: int
    touch_timestamps: List[str]
    avg_price: float

# PatternResult class (unchanged from your input)
class PatternResult(BaseModel):
    pattern: str
    start_idx: int
    end_idx: int
    confidence: float
    key_levels: Dict[str, float]
    candle_indexes: List[int]
    timestamp_start: datetime
    timestamp_end: datetime 
    detection_time: str
    exact_pattern_type: str
    market_structure: Optional[str] = None
    demand_zone_interaction: Optional[Dict[str, Any]] = None
    supply_zone_interaction: Optional[Dict[str, Any]] = None
    volume_confirmation_at_zone: Optional[bool] = None

# Updated MarketContextResult class with additional fields
class MarketContextResult(BaseModel):
    scenario: str
    volatility: float
    trend_strength: float
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    context: Dict[str, Any]
    demand_zones: List[DemandZone]  # Added for demand zones
    supply_zones: List[SupplyZone]  # Added for supply zones

# AnalysisResult class (updated to use the new MarketContextResult)
class AnalysisResult(BaseModel):
    patterns: List[PatternResult]
    market_context: MarketContextResult
    analysis_timestamp: str