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

    # @field_validator('plan_id')
    # @classmethod
    # def validate_plan_id(cls, v):
    #     if not v.startswith("price_") and not re.fullmatch(r'[a-zA-Z0-9_-]+', v):
    #         raise ValueError('Invalid plan_id. Must be a valid Stripe price ID or custom slug (e.g., "starter_plan").')
    #     return v

    