import firebase_admin
from firebase_admin import credentials, messaging
from common.logger import logger
import os
from typing import List, Dict, Optional
import asyncio

class NotificationService:
    _initialized = False

    def __init__(self):
        if not NotificationService._initialized:
            if not firebase_admin._apps:
                cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            NotificationService._initialized = True
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
        Sends notifications to multiple devices in batches to respect FCM limits.
        FCM allows up to 500 tokens per multicast message.
        """
        if not tokens:
            return

        all_failed_tokens = []
        BATCH_SIZE = 500  # FCM limit per multicast message
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
                # Run the blocking call in a separate thread
                response = await asyncio.to_thread(
                    messaging.send_each_for_multicast,
                    message
                )
                
                logger.info(f"Batch {i // BATCH_SIZE + 1}: {response.success_count} successes, {response.failure_count} failures for: {title}")
                
                if response.failure_count > 0:
                    # Find failed tokens from this specific batch
                    failed_in_batch = [
                        batch_tokens[idx] 
                        for idx, resp in enumerate(response.responses) 
                        if not resp.success
                    ]
                    logger.error(f"Failed tokens in batch: {failed_in_batch}")
                    all_failed_tokens.extend(failed_in_batch)
            except Exception as e:
                logger.error(f"Error sending batch: {e}")
                # Assume all tokens in this batch failed on exception
                all_failed_tokens.extend(batch_tokens)
                
        return all_failed_tokens