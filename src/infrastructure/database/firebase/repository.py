# paid_plans.py (updated)
from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, db
from common.logger import logger
from datetime import datetime, timezone
import firebase_admin
import os

class FirebaseRepository:
    def __init__(self, app_name=None):
        # Initialize Firebase Admin SDK only once or with a unique name
        if not firebase_admin._apps:
            # No apps initialized yet
            cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
            })
        elif app_name and app_name not in firebase_admin._apps:
            # Initialize with a unique name if provided
            cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
            }, name=app_name)
        
        # Reference to the database
        self.db = db.reference('users')
    
    async def check_user_exists(self, user_id: str) -> bool:
        """Check if a user exists in Firebase"""
        try:
            user_ref = self.db.child(user_id).get()
            
            # Fix for the 'dict' object has no attribute 'val' error
            # Check if user_ref is a dict (direct data) or has val() method
            if hasattr(user_ref, 'val'):
                # It's a DataSnapshot object with val() method
                user_data = user_ref.val()
            else:
                # It's already a dict or another data type
                user_data = user_ref
                
            if user_data is None:
                logger.error(f"User {user_id} not found in Firebase")
                raise HTTPException(status_code=404, detail=f"User {user_id} not found in Firebase")
                
            logger.info(f"User {user_id} exists in Firebase")
            return True
        except Exception as e:
            logger.error(f"Firebase error: {str(e)}")
            raise HTTPException(500, "Database access failed")
    
    async def update_subscription(self, user_id: str, plan_type: str) -> bool:
        try:
            updates = {
                'subscriptionType': plan_type,
                'usingTestDrive': plan_type == "test_drive",
                'updatedAt': datetime.now(timezone.utc).isoformat(),
                'userPaid': True,
            }
            
            # Secure write operation with validation
            self.db.child(user_id).update(updates)
            logger.info(f"Firebase updated for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Firebase error: {str(e)}")
            raise HTTPException(500, "Database update failed")
        
    async def get_user_subscription(self, user_id: str) -> str:
        """Retrieve the current subscription type for a user from Firebase."""
        try:
            user_ref = self.db.child(user_id).get()
            
            # Handle whether user_ref is a DataSnapshot or direct data
            if hasattr(user_ref, 'val'):
                user_data = user_ref.val()
            else:
                user_data = user_ref
                
            if user_data is None:
                logger.error(f"User {user_id} not found in Firebase")
                raise HTTPException(status_code=404, detail=f"User {user_id} not found in Firebase")
            
            subscription_type = user_data.get('subscriptionType', 'free')
            logger.info(f"User {user_id} has subscription type: {subscription_type}")
            return subscription_type
        except Exception as e:
            logger.error(f"Firebase error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database access failed")