"""Firebase ID-token verification for private scanner endpoints only."""
import asyncio
from collections import OrderedDict
import hashlib
import time

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
from infrastructure.database.firebase.scanner_app import scanner_firebase_app


def verify_token_sync(token):
    from firebase_admin import auth
    return auth.verify_id_token(token, app=scanner_firebase_app(), check_revoked=True)


class FirebaseIDVerifier:
    """Bounded positive cache and shared verification for concurrent same-token reads."""
    def __init__(self, verify=verify_token_sync):
        self.verify_sync = verify
        self.cache = OrderedDict()
        self.inflight = {}
        self.slots = asyncio.Semaphore(8)

    async def verify(self, token):
        if not token or len(token) > 8192:
            raise HTTPException(401, 'Invalid sign-in token', headers={'WWW-Authenticate': 'Bearer'})
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self.cache.get(key)
        if cached and cached[1] > time.time():
            self.cache.move_to_end(key)
            return cached[0]
        task = self.inflight.get(key)
        if task is None:
            if len(self.inflight) >= 128:
                raise HTTPException(503, 'Authentication busy', headers={'Retry-After': '2'})
            task = asyncio.create_task(self._verify(token, key))
            self.inflight[key] = task
            task.add_done_callback(lambda done: (self.inflight.pop(key, None),
                                                done.exception() if not done.cancelled() else None))
        return await asyncio.shield(task)

    async def _verify(self, token, key):
        from firebase_admin import auth
        try:
            async with self.slots:
                claims = await asyncio.to_thread(self.verify_sync, token)
            uid, expires = claims['uid'], float(claims['exp'])
            if not isinstance(uid, str) or not 1 <= len(uid) <= 128 or expires <= time.time():
                raise ValueError('Invalid identity claims')
        except auth.CertificateFetchError:
            raise HTTPException(503, 'Authentication temporarily unavailable') from None
        except (auth.InvalidIdTokenError, auth.RevokedIdTokenError, auth.UserDisabledError,
                ValueError, KeyError, TypeError):
            raise HTTPException(401, 'Invalid sign-in token', headers={'WWW-Authenticate': 'Bearer'}) from None
        except Exception:
            raise HTTPException(503, 'Authentication temporarily unavailable') from None
        self.cache[key] = (uid, min(expires, time.time() + 30))
        self.cache.move_to_end(key)
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        return uid


def get_identity_verifier(request: Request):
    if not hasattr(request.app.state, 'scanner_identity_verifier'):
        request.app.state.scanner_identity_verifier = FirebaseIDVerifier()
    return request.app.state.scanner_identity_verifier


async def scanner_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
                       verifier=Depends(get_identity_verifier)):
    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise HTTPException(401, 'Sign in to manage pattern watches', headers={'WWW-Authenticate': 'Bearer'})
    return await verifier.verify(credentials.credentials)
