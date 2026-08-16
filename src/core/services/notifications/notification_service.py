import firebase_admin
from firebase_admin import credentials, messaging
from common.logger import logger
import os
from typing import List, Dict, Optional
import asyncio
from src.infrastructure.database.supabase.markets_repo import MarketRepository


class NotificationService:
    _initialized = False

    def __init__(self):
        if not NotificationService._initialized:
            if not firebase_admin._apps:
                cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            NotificationService._initialized = True
            # Add a repository instance for cleaning up tokens
            self.repo = MarketRepository()
            logger.info("NotificationService initialized with Firebase Admin.")

    async def send_batch_fcm_notifications(
        self,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None,
        android_config: Optional[messaging.AndroidConfig] = None,
        apns_config: Optional[messaging.APNSConfig] = None
    ):
        """
        Sends notifications to multiple devices in batches and cleans up invalid tokens.
        """
        if not tokens:
            return []

        all_failed_tokens = []
        unregistered_tokens = []  # Tokens that are permanently invalid
        BATCH_SIZE = 500
        for i in range(0, len(tokens), BATCH_SIZE):
            batch_tokens = tokens[i:i + BATCH_SIZE]
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data,
                tokens=batch_tokens,
                android=android_config,
                apns=apns_config
            )
            try:
                response = await asyncio.to_thread(
                    messaging.send_each_for_multicast,
                    message
                )
                
                logger.info(f"Batch {i // BATCH_SIZE + 1}: {response.success_count} successes, {response.failure_count} failures for: {title}")
                
                if response.failure_count > 0:
                    for idx, resp in enumerate(response.responses):
                        if not resp.success:
                            token = batch_tokens[idx]
                            all_failed_tokens.append(token)
                            error_info = getattr(resp, 'exception', None)
                            if error_info:
                                logger.error(f"FCM failure for token: {token} | Error: {error_info}")
                                code = getattr(error_info, 'code', None)
                                # If the token is no longer registered, it should be deleted
                                if code == 'UNREGISTERED':
                                    unregistered_tokens.append(token)
            except Exception as e:
                logger.error(f"Error sending batch: {e}")
                all_failed_tokens.extend(batch_tokens)
        
        # After all batches, clean up the invalid tokens from the database
        if unregistered_tokens:
            logger.info(f"Found {len(unregistered_tokens)} unregistered FCM tokens to clean up.")
            try:
                await self.repo.remove_invalid_fcm_tokens(unregistered_tokens)
                logger.info(f"Successfully cleaned up {len(unregistered_tokens)} invalid tokens.")
            except Exception as e:
                logger.error(f"Failed to clean up invalid FCM tokens: {e}")
                
        return all_failed_tokens