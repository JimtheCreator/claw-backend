# In a new file, e.g., promo_codes.py
from fastapi import APIRouter, HTTPException, Depends
import stripe
import os
from pydantic import BaseModel
from src.common.logger import logger

router = APIRouter(tags=["Stripe Promo Codes"])

stripe.api_key = os.getenv("PRODUCTION_STRIPE_API_KEY")

class PromoValidationRequest(BaseModel):
    promo_code: str
    price_id: str

class PromoValidationResponse(BaseModel):
    is_valid: bool
    original_amount: int
    discounted_amount: int
    discount_amount_off: int
    coupon_id: str
    coupon_name: str | None = None
    message: str

def calculate_discounted_amount(original_amount: int, coupon: stripe.Coupon) -> int:
    """Calculates the final amount after applying a coupon discount."""
    if coupon.amount_off:
        # Fixed amount discount
        discounted_amount = original_amount - coupon.amount_off
    elif coupon.percent_off:
        # Percentage discount
        discount = original_amount * (coupon.percent_off / 100)
        discounted_amount = original_amount - int(discount)
    else:
        # No discount type found, return original amount
        return original_amount
        
    # Ensure the price doesn't go below zero
    return max(0, discounted_amount)

@router.post("/stripe/validate-promo-code", response_model=PromoValidationResponse)
async def validate_promo_code(request: PromoValidationRequest):
    """
    Validates a Stripe Promotion Code and returns the potential discount.
    """
    if not stripe.api_key:
        logger.error("Stripe API key is not set")
        raise HTTPException(status_code=500, detail="Server configuration error.")

    try:
        # 1. Find the promotion code
        promo_codes = stripe.PromotionCode.list(code=request.promo_code, active=True, limit=1)
        if not promo_codes.data:
            raise HTTPException(status_code=404, detail="Invalid or expired promo code.")

        promo_code = promo_codes.data[0]
        coupon = promo_code.coupon

        # 2. Check if the coupon is valid
        if not coupon or not coupon.valid:
            raise HTTPException(status_code=404, detail="This promo code is no longer valid.")

        # 3. Get the original price
        original_price = stripe.Price.retrieve(request.price_id)
        original_amount = original_price.unit_amount

        # 4. Calculate the discounted price
        discounted_amount = calculate_discounted_amount(original_amount, coupon)
        
        if discounted_amount == original_amount:
             raise HTTPException(status_code=400, detail="This coupon does not apply to the selected item.")

        return PromoValidationResponse(
            is_valid=True,
            original_amount=original_amount,
            discounted_amount=discounted_amount,
            discount_amount_off=original_amount - discounted_amount,
            coupon_id=coupon.id,
            coupon_name=coupon.name,
            message="Promo code applied successfully!"
        )

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error validating promo code: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e.user_message or "Could not validate promo code."))
    except HTTPException as e:
        # Re-raise known HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error validating promo code: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")