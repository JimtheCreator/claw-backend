from pydantic import BaseModel, Field
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import numpy as np

from datetime import datetime

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
    market_structure: Optional[str] = None  # Added for context
    # In PatternInstance dataclass
    demand_zone_interaction: Optional[str] = None  # e.g., "approaching", "testing", "rejected_from", "bounced_from"
    supply_zone_interaction: Optional[str] = None  # e.g., "approaching", "testing", "rejected_from", "broke_through"
    volume_confirmation_at_zone: Optional[bool] = None # True if volume confirms the zone's significance

class MarketContextResult(BaseModel):
    scenario: str
    volatility: float
    trend_strength: float
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float] # This will be populated from detect_patterns.py
    context: Dict[str, Any]

class AnalysisResult(BaseModel):
    # Note: The 'patterns' field here seems redundant with market_context.active_patterns
    # but is kept to avoid breaking existing code that might use it.
    patterns: List[PatternResult]
    market_context: MarketContextResult
    analysis_timestamp: str