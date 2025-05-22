from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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
    client_secret: Optional[str] = None
    intent_type: Optional[str] = None
    publishable_key: str
    customer_id: Optional[str] = None
    ephemeral_key_secret: Optional[str] = None
    payment_intent_id: Optional[str] = None # Still useful for reference
    setup_intent_id: Optional[str] = None   # Add this for reference if it's a SetupIntent
    subscription_id: Optional[str] = None
    plan_type: str
    mode: str
    payment_required: bool = True  # Indicates if payment sheet is needed
    message: Optional[str] = None  # Success message when no payment is required

# Response Schema for Subscription Cancellation
class CancellationResponseSchema(BaseModel):
    success: bool = Field(..., description="Whether the cancellation was successful")
    subscription_id: str = Field(..., description="The Stripe subscription ID that was canceled")
    cancellation_status: str = Field(..., description="Either 'immediate' or 'scheduled'")
    cancellation_date: str = Field(..., description="ISO-formatted date of actual or scheduled cancellation")
    message: str = Field(..., description="Human-readable message about the cancellation")

class UpgradeSuccessSchema(BaseModel):
    message: str