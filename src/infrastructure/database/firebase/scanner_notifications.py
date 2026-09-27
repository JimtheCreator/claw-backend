"""FCM adapter; invoked only by the explicitly enabled delivery worker."""
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
import re
from firebase_admin import db, messaging

from core.services.scanner_alerts import PermanentDeliveryError, PushConfigurationError
from infrastructure.database.firebase.scanner_app import scanner_firebase_app

log = logging.getLogger(__name__)


class FirebaseScannerSender:
    def __init__(self, repository=None):
        self.repository = repository

    async def send(self, delivery):
        if self.repository is None:
            return await asyncio.to_thread(self._send, delivery)
        tokens, receipts = await self.repository.notification_devices(delivery)
        if not tokens:
            if receipts:
                return 'devices-processed'
            # Android's existing registration remains supported during migration.
            return await asyncio.to_thread(self._send, delivery)
        last_id = 'devices-processed'
        transient_failure = False
        configuration_failure = False
        accepted = bool(receipts)
        for token in tokens:
            if hashlib.sha256(token.encode()).hexdigest() in receipts:
                continue
            try:
                last_id = await asyncio.to_thread(self._send, delivery, token)
            except PermanentDeliveryError:
                await self.repository.record_device_delivery(delivery,token,invalid=True)
            except PushConfigurationError:
                configuration_failure = True
            except Exception as exc:
                log.warning('Scanner device delivery failed (%s); token and payload omitted', type(exc).__name__)
                transient_failure = True
            else:
                accepted = True
                await self.repository.record_device_delivery(delivery,token)
        if configuration_failure:
            raise PushConfigurationError()
        if transient_failure:
            raise RuntimeError('Some notification devices need retry')
        if not accepted:
            raise PermanentDeliveryError()
        return last_id

    @staticmethod
    def _send(delivery, token=None):
        app = scanner_firebase_app()
        uid = delivery['user_id']
        if re.search(r'[.#$\[\]/]', uid):
            raise PermanentDeliveryError()
        if token is None:
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
                  'provider': match['provider'], 'market': match['market'],
                  'type': 'event_match', 'user_id': uid,
                  'universe': delivery.get('universe', 'binance-spot-pilot')},
            android=messaging.AndroidConfig(collapse_key=notification_id, ttl=remaining),
            apns=messaging.APNSConfig(headers={'apns-collapse-id': notification_id,
                'apns-expiration': str(int(delivery['expires_at'].timestamp())),
                'apns-push-type': 'alert', 'apns-priority': '10'},
                payload=messaging.APNSPayload(messaging.Aps(sound='default', thread_id='event-' + match['pattern_id']))))
        try:
            return messaging.send(message, app=app)
        except messaging.ThirdPartyAuthError:
            raise PushConfigurationError() from None
        except (messaging.UnregisteredError, messaging.SenderIdMismatchError):
            raise PermanentDeliveryError() from None
