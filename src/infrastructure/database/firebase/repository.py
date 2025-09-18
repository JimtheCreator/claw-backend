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
    
    # In repository.py - Corrected approach for userPaid flag
    async def update_subscription(self, user_id: str, plan_type: str) -> bool:
        """Update subscription for successful payments - sets userPaid=True permanently"""
        try:
            updates = {
                'subscriptionType': plan_type,
                'usingTestDrive': plan_type == "test_drive",
                'updatedAt': datetime.now(timezone.utc).isoformat(),
                'userPaid': True,  # ✅ This should ONLY be called for successful payments
            }
            
            self.db.child(user_id).update(updates)
            logger.info(f"Firebase updated for {user_id}: plan={plan_type}, userPaid=True (payment successful)")
            return True
        except Exception as e:
            logger.error(f"Firebase error: {str(e)}")
            raise HTTPException(500, "Database update failed")

    async def handle_payment_failure(self, user_id: str) -> bool:
        """Handle payment failures - does NOT modify userPaid flag"""
        try:
            # Get current user data first to preserve userPaid status
            user_ref = self.db.child(user_id).get()
            if hasattr(user_ref, 'val'):
                user_data = user_ref.val()
            else:
                user_data = user_ref
                
            current_user_paid = user_data.get('userPaid', False) if user_data else False
            
            updates = {
                'subscriptionType': 'free',
                'usingTestDrive': False,
                'updatedAt': datetime.now(timezone.utc).isoformat(),
                # ✅ Preserve the existing userPaid value - don't change it
                'userPaid': current_user_paid,
            }
            
            self.db.child(user_id).update(updates)
            logger.info(f"Payment failure handled for {user_id}: set to free plan, userPaid preserved as {current_user_paid}")
            return True
        except Exception as e:
            logger.error(f"Firebase error handling payment failure: {str(e)}")
            raise HTTPException(500, "Database update failed")

    async def revert_to_free_plan(self, user_id: str) -> bool:
        """Revert user to free plan (for subscription cancellations, etc.) - preserves userPaid"""
        try:
            # Get current user data to preserve userPaid status
            user_ref = self.db.child(user_id).get()
            if hasattr(user_ref, 'val'):
                user_data = user_ref.val()
            else:
                user_data = user_ref
                
            current_user_paid = user_data.get('userPaid', False) if user_data else False
            
            updates = {
                'subscriptionType': 'free',
                'usingTestDrive': False,
                'updatedAt': datetime.now(timezone.utc).isoformat(),
                # ✅ Preserve existing userPaid - once someone has paid successfully, they never see test-drive again
                'userPaid': current_user_paid,
            }
            
            self.db.child(user_id).update(updates)
            logger.info(f"User {user_id} reverted to free plan, userPaid preserved as {current_user_paid}")
            return True
        except Exception as e:
            logger.error(f"Firebase error reverting to free plan: {str(e)}")
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
        

    async def get_fcm_tokens_for_users(self, user_ids: list[str]) -> dict[str, str]:
        """
        Fetches FCM tokens for a given list of user IDs from Firebase Realtime Database.

        Args:
            user_ids (list[str]): A list of user IDs to fetch tokens for.

        Returns:
            dict[str, str]: A dictionary mapping user_id to its fcm_token.
        """
        if not user_ids:
            return {}

        tokens_map = {}
        # NOTE: This fetches users one by one. For very high throughput,
        # you might restructure your DB for batch gets, but this is a solid start.
        for user_id in user_ids:
            try:
                # Assuming the token is stored at path /users/{user_id}/fcmToken
                token = self.db.child(user_id).child('fcmToken').get()

                if token:
                    tokens_map[user_id] = token
                else:
                    logger.warning(f"FCM token not found for user {user_id} in Firebase.")

            except Exception as e:
                logger.error(f"Error fetching FCM token for user {user_id} from Firebase: {str(e)}")
                # Continue to the next user
                continue
        
        logger.info(f"Fetched {len(tokens_map)} FCM tokens from Firebase.")
        return tokens_map