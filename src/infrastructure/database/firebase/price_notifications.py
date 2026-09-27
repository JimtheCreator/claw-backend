"""Same device registry and per-device receipts as event notifications."""
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import logging
import time

log = logging.getLogger(__name__)
from firebase_admin import messaging
from core.services.scanner_alerts import PermanentDeliveryError,PushConfigurationError
from infrastructure.database.firebase.scanner_app import scanner_firebase_app
from infrastructure.database.firebase.scanner_notifications import FirebaseScannerSender

def display_number(value):
    rendered = format(Decimal(str(value)), ',f')
    return rendered.rstrip('0').rstrip('.') if '.' in rendered else rendered


def price_notification_body(payload):
    movement = 'rose' if payload['direction'] == 'above' else 'fell'
    message = f"{payload['symbol']} {movement} to {display_number(payload['price'])}"
    target = f"Target {display_number(payload['target'])}"
    if payload.get('kind') == 'percentage':
        sign = '+' if payload['direction'] == 'above' else '−'
        target += f" ({sign}{display_number(payload['amount'])}%)"
    return f"{message} · {target}"


class PriceDeliveryReceipts:
    """Short-lived receipt buffer; committed with outbox completion by the worker."""
    def __init__(self, delivery):
        self.delivery = delivery
        delivery['accepted_token_hashes'] = []
        delivery['invalid_tokens'] = []

    async def notification_devices(self, delivery):
        return delivery['device_tokens'], set(delivery['device_receipts'])

    async def record_device_delivery(self, delivery, token, *, invalid=False):
        if invalid:
            delivery['invalid_tokens'].append(token)
        else:
            delivery['accepted_token_hashes'].append(hashlib.sha256(token.encode()).hexdigest())


class FirebasePriceSender(FirebaseScannerSender):
    async def send(self, delivery):
        if 'device_tokens' not in delivery:
            return await super().send(delivery)
        sender = FirebasePriceSender(PriceDeliveryReceipts(delivery))
        return await FirebaseScannerSender.send(sender, delivery)

    @staticmethod
    def _send(delivery,token=None):
        if not token: raise RuntimeError('No registered notification device')
        p=delivery['payload']; identifier=str(delivery['id'])
        remaining=delivery['expires_at']-datetime.now(timezone.utc)
        if remaining.total_seconds()<=0: raise PermanentDeliveryError()
        message=messaging.Message(token=token,
            notification=messaging.Notification(title=f"Alert on {p['symbol']}",
                body=price_notification_body(p)),
            data={'type':'price_alert','notification_id':identifier,'user_id':delivery['user_id'],
                  'symbol':p['symbol'],'provider':'binance','market':'spot','alert_id':str(delivery['rule_id'])},
            android=messaging.AndroidConfig(collapse_key=identifier,ttl=remaining),
            apns=messaging.APNSConfig(headers={'apns-collapse-id':identifier,'apns-expiration':str(int(delivery['expires_at'].timestamp())),
                'apns-push-type':'alert','apns-priority':'10'},
                payload=messaging.APNSPayload(messaging.Aps(sound='default',thread_id='price-'+p['symbol'],
                    category='WATCHERS_PRICE_ALERT', custom_data={'interruption-level':'time-sensitive'}))))
        started = time.monotonic()
        try:
            result = messaging.send(message,app=scanner_firebase_app())
            age = (datetime.now(timezone.utc)-delivery.get('triggered_at', delivery['created_at'])).total_seconds()
            log.info('Price push accepted: notification=%s firebase=%.3fs since_tick=%.3fs',
                     identifier,time.monotonic()-started,age)
            return result
        except messaging.ThirdPartyAuthError: raise PushConfigurationError() from None
        except (messaging.UnregisteredError,messaging.SenderIdMismatchError): raise PermanentDeliveryError() from None
