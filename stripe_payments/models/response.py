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

# Create a response model for checkout
class CheckoutResponse(BaseModel):
    checkout_url: str

class NativeCheckoutResponseSchema(BaseModel):
    client_secret: str
    publishable_key: str
    intent_type: str  # "payment_intent" or "setup_intent"
    customer_id: Optional[str] = None
    ephemeral_key_secret: Optional[str] = None
    payment_intent_id: Optional[str] = None # Still useful for reference
    setup_intent_id: Optional[str] = None   # Add this for reference if it's a SetupIntent
    subscription_id: Optional[str] = None
    plan_type: str
    mode: str