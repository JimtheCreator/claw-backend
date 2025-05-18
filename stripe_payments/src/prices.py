from fastapi import HTTPException
import stripe
from src.common.logger import logger
from fastapi import APIRouter
from typing import List
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.response import PriceResponse

router = APIRouter(tags=["Stripe Plan Prices"])

# API Endpoints
@router.get("/stripe/prices", response_model=List[PriceResponse])
async def get_subscription_prices():
    """Fetch available subscription plans from Stripe"""
    try:
        # Environment variables
        STRIPE_API_KEY = os.getenv("TEST_STRIPE_API_KEY")

        # Check if Stripe API key is set
        if not STRIPE_API_KEY:
            logger.error("Stripe API key is not set")
            raise HTTPException(status_code=500, detail="Stripe API key is not set")
        
        # Initialize Stripe
        stripe.api_key = STRIPE_API_KEY
        # Check if Stripe is initialized
        if not stripe.api_key:
            logger.error("Stripe API key is not set")
            raise HTTPException(status_code=500, detail="Stripe API key is not set")

        # Retrieve all active prices with their products
        prices = stripe.Price.list(active=True, expand=["data.product"])
        if not prices.data:
            logger.error("No active prices found in Stripe")
            raise HTTPException(status_code=404, detail="No active prices found")
        # Format response
        formatted_prices = []
        for price in prices.data:
            product = price.product
            
            # Skip prices that don't belong to Watchers subscription products
            if not product.active or not product.metadata.get("app") == "watchers":
                continue
                
            # Determine billing period
            interval = price.recurring.interval if price.recurring else "one_time"
            
            # Extract features from product metadata or description
            features = []
            if product.metadata.get("features"):
                features = product.metadata.get("features").split(",")
            elif product.description:
                features = [line.strip() for line in product.description.split(",")]
            
            formatted_price = PriceResponse(
                id=price.id,
                type=product.metadata.get("plan_type", "unknown"),
                billing_period=interval,
                amount=price.unit_amount,
                currency=price.currency,
                name=product.name,
                description=product.description,
                features=features
            )
            
            formatted_prices.append(formatted_price)
        
        return formatted_prices
    except Exception as e:
        logger.error(f"Error fetching prices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching subscription prices: {str(e)}")
    
