import asyncio
from datetime import datetime, timedelta, timezone
import time
from unittest.mock import AsyncMock, Mock
import uuid

from fastapi import HTTPException
from firebase_admin import auth
from pydantic import ValidationError
import pytest

from core.scanner.watches import WatchCreate
from core.services.scanner_alerts import consume_events, deliver_one, PermanentDeliveryError
from presentation.api.dependencies.scanner_auth import FirebaseIDVerifier, verify_token_sync


def test_watch_symbols_are_canonical_and_identity_cannot_be_supplied():
    spec = WatchCreate(pattern_id='bullish_engulfing', symbols=['ETHUSDT','BTCUSDT','ETHUSDT'])
    assert spec.symbols == ['BTCUSDT','ETHUSDT']
    for extra in ({'user_id':'someone-else'}, {'symbols':['btc/usdt']}, {'interval':'1m'}, {'mode':'forever'}):
        with pytest.raises(ValidationError):
            WatchCreate(pattern_id='bullish_engulfing', **extra)


def test_firebase_sdk_must_verify_signature_and_revocation(monkeypatch):
    app = object()
    verify = Mock(return_value={'uid':'alice'})
    monkeypatch.setattr('presentation.api.dependencies.scanner_auth.scanner_firebase_app', lambda: app)
    monkeypatch.setattr(auth, 'verify_id_token', verify)
    assert verify_token_sync('signed-token') == {'uid':'alice'}
    verify.assert_called_once_with('signed-token', app=app, check_revoked=True)


def test_same_token_verification_is_shared_and_cache_expires(monkeypatch):
    async def scenario():
        verify = Mock(return_value={'uid':'alice', 'exp':time.time()+300})
        verifier = FirebaseIDVerifier(verify)
        assert await asyncio.gather(*(verifier.verify('token') for _ in range(100))) == ['alice']*100
        assert verify.call_count == 1
        key = next(iter(verifier.cache))
        verifier.cache[key] = ('alice', time.time()-1)
        assert await verifier.verify('token') == 'alice'
        assert verify.call_count == 2
    asyncio.run(scenario())


@pytest.mark.parametrize('failure,status', [
    (auth.InvalidIdTokenError('bad signature'),401),
    (auth.RevokedIdTokenError('revoked'),401),
    (auth.UserDisabledError('disabled'),401),
    (auth.CertificateFetchError('offline', RuntimeError('offline')),503),
    (RuntimeError('configuration'),503),
])
def test_authentication_failures_are_not_cached(failure, status):
    async def scenario():
        verify = Mock(side_effect=failure)
        verifier = FirebaseIDVerifier(verify)
        for _ in range(2):
            with pytest.raises(HTTPException) as caught:
                await verifier.verify('token')
            assert caught.value.status_code == status
        assert not verifier.cache and verify.call_count == 2
    asyncio.run(scenario())


def test_expired_identity_and_oversized_tokens_are_rejected():
    async def scenario():
        verify = Mock(return_value={'uid':'alice','exp':time.time()-1})
        verifier = FirebaseIDVerifier(verify)
        for token in ('','x'*8193,'expired'):
            with pytest.raises(HTTPException) as caught:
                await verifier.verify(token)
            assert caught.value.status_code == 401
        assert verify.call_count == 1
    asyncio.run(scenario())


def test_inbox_never_acknowledges_before_commit():
    async def scenario():
        stream = Mock(read=AsyncMock(return_value=[('1-0', {'batch':'data'})]), acknowledge=AsyncMock())
        repo = Mock(accept_batch=AsyncMock(side_effect=RuntimeError('DB unavailable')))
        with pytest.raises(RuntimeError):
            await consume_events(stream, repo, 'worker')
        stream.acknowledge.assert_not_called()
        repo.accept_batch.side_effect = None
        assert await consume_events(stream, repo, 'worker') == 1
        stream.acknowledge.assert_awaited_once_with('1-0')
    asyncio.run(scenario())


@pytest.mark.parametrize('failure,expected,kwargs', [
    (None,'delivered',{'provider_id':'message-1'}),
    (RuntimeError('temporary'),'retry',{'error':'delivery_unavailable'}),
    (PermanentDeliveryError(),'failed',{'error':'invalid_destination','permanent':True}),
])
def test_delivery_results_are_recorded(failure, expected, kwargs):
    async def scenario():
        delivery = {'status':'sending'}
        repo = Mock(claim_delivery=AsyncMock(return_value=delivery),
                    delivery_allowed=AsyncMock(return_value=True), finish_delivery=AsyncMock(return_value=True))
        sender = Mock(send=AsyncMock(return_value='message-1', side_effect=failure))
        assert await deliver_one(repo, sender) == expected
        repo.finish_delivery.assert_awaited_once_with(delivery, **kwargs)
    asyncio.run(scenario())


def test_preflight_cancellation_and_expiry_do_not_send():
    async def scenario():
        sender = Mock(send=AsyncMock())
        repo = Mock(claim_delivery=AsyncMock(return_value={'status':'sending'}),
                    delivery_allowed=AsyncMock(return_value=False), finish_delivery=AsyncMock())
        assert await deliver_one(repo, sender) == 'cancelled'
        repo.claim_delivery.return_value = {'status':'expired'}
        assert await deliver_one(repo, sender) == 'expired'
        sender.send.assert_not_called()
    asyncio.run(scenario())


def test_notification_contains_stable_identity_and_provider_expiration(monkeypatch):
    from infrastructure.database.firebase import scanner_notifications as module
    app = object()
    monkeypatch.setattr(module, 'scanner_firebase_app', lambda: app)
    reference = Mock()
    reference.child.return_value.child.return_value.get.return_value = 'device-token'
    monkeypatch.setattr(module.db, 'reference', Mock(return_value=reference))
    send = Mock(return_value='message-1')
    monkeypatch.setattr(module.messaging, 'send', send)
    delivery = {'id':uuid.uuid4(),'watch_id':uuid.uuid4(),'user_id':'alice','event_id':'event',
                'expires_at':datetime.now(timezone.utc)+timedelta(minutes=5),
                'payload':{'match':{'symbol':'BTCUSDT','pattern_id':'bullish_engulfing',
                    'interval':'15m','provider':'binance','market':'spot'}}}
    assert asyncio.run(module.FirebaseScannerSender().send(delivery)) == 'message-1'
    message = send.call_args.args[0]
    assert message.data['notification_id'] == str(delivery['id'])
    assert message.android.collapse_key == str(delivery['id'])
    assert 0 < message.android.ttl.total_seconds() <= 300
    assert message.apns.headers['apns-expiration'] == str(int(delivery['expires_at'].timestamp()))
    assert message.apns.headers['apns-collapse-id'] == str(delivery['id'])
    send.assert_called_once()
