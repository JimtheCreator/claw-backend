"""FCM adapter; invoked only by the explicitly enabled delivery worker."""
import asyncio
from datetime import datetime, timezone
import re
from firebase_admin import db, messaging

from core.services.scanner_alerts import PermanentDeliveryError
from infrastructure.database.firebase.scanner_app import scanner_firebase_app


class FirebaseScannerSender:
    async def send(self, delivery):
        return await asyncio.to_thread(self._send, delivery)

    @staticmethod
    def _send(delivery):
        app = scanner_firebase_app()
        uid = delivery['user_id']
        if re.search(r'[.#$\[\]/]', uid):
            raise PermanentDeliveryError()
        token = db.reference('users', app=app).child(uid).child('fcmToken').get()
        if not isinstance(token, str) or not token:
            # A device may register later; retry within this alert's freshness window.
            raise RuntimeError('No registered notification device')
        match = delivery['payload']['match']
        notification_id = str(delivery['id'])
        remaining = delivery['expires_at'] - datetime.now(timezone.utc)
        if remaining.total_seconds() <= 0:
            raise PermanentDeliveryError()
        message = messaging.Message(
            token=token,
            notification=messaging.Notification(title=match['pattern_id'].replace('_', ' ').title(),
                body=f"{match['symbol']} · {match['interval']} · Pattern detected"),
            data={'notification_id': notification_id, 'watch_id': str(delivery['watch_id']),
                  'event_id': delivery['event_id'], 'symbol': match['symbol'],
                  'pattern_id': match['pattern_id'], 'interval': match['interval'],
                  'provider': match['provider'], 'market': match['market']},
            android=messaging.AndroidConfig(collapse_key=notification_id, ttl=remaining),
            apns=messaging.APNSConfig(headers={'apns-collapse-id': notification_id,
                'apns-expiration': str(int(delivery['expires_at'].timestamp()))}))
        try:
            return messaging.send(message, app=app)
        except (messaging.UnregisteredError, messaging.SenderIdMismatchError):
            raise PermanentDeliveryError() from None
