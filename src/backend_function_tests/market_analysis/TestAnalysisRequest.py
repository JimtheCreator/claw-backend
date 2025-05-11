from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

class AnalysisRequest(BaseModel):
    timeframe: str
    user_id: Optional[str] = None
    end_time: Optional[str] = None

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v):
        if not v:
            raise ValueError('Timeframe is required')
        if not re.match(r'^\d+[mhdwM]$', v):
            raise ValueError('Invalid timeframe format. Use format like "30m", "4h", "2d"')
        return v
        
    @field_validator('end_time')
    @classmethod
    def validate_datetime_format(cls, v):
        if v is None:
            return v
        try:
            # Check if it matches the dd/mm/yy HH:MM format
            if not re.match(r'^(\d{2}/\d{2}/\d{2}\s\d{2}:\d{2})$', v):
                raise ValueError('Invalid date format. Use format like "01/02/23 14:30" (dd/mm/yy HH:MM)')
        except Exception:
            raise ValueError('Invalid date format. Use format like "01/02/23 14:30" (dd/mm/yy HH:MM)')
        return v