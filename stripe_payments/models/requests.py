from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator
import re


class SubscribeRequest(BaseModel):
    user_id: str
    plan_id: str

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{28,36}', v):
            raise ValueError('Invalid user_id format. Must be 28–36 characters long (Firebase ID).')
        return v

class CancelRequest(BaseModel):
    subscription_id: str
    user_id: str