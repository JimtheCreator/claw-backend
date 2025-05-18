from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class PriceResponse(BaseModel):
    id: str
    type: str
    billing_period: str
    amount: int
    currency: str
    name: str
    description: Optional[str] = None
    features: List[str]

class SubscriptionResponse(BaseModel):
    success: bool
    subscription_id: Optional[str] = None
    plan_type: Optional[str] = None
    error_message: Optional[str] = None
    limits: Optional[Dict[str, Any]] = None