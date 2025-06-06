from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator, Field, EmailStr, conint, constr
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

# Request Schema for Subscription Cancellation
class CancelSubscriptionRequest(BaseModel):
    user_id: str = Field(..., description="User ID in Firebase")
    subscription_id: Optional[str] = Field(None, description="Stripe subscription ID. If not provided, will attempt to find by user's email")
    cancel_at_period_end: bool = Field(False, description="If true, cancels at end of billing period. If false, cancels immediately")