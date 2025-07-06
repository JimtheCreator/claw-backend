from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

class AnalysisRequest(BaseModel):
    timeframe: str
    user_id: Optional[str] = None

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v):
        if not re.match(r'^\d+[mhdwM]$', v):
            raise ValueError('Invalid timeframe format. Use format like "30m", "4h", "2d"')
        return v

