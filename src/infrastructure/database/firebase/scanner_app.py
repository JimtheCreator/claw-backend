"""Lazy, bounded-time Firebase application for authenticated scanner features."""
import os
import threading

_app_lock = threading.Lock()


def scanner_firebase_app():
    import firebase_admin
    from firebase_admin import credentials
    with _app_lock:
        try:
            return firebase_admin.get_app('scanner-auth')
        except ValueError:
            path = os.getenv('FIREBASE_CREDENTIALS_PATH')
            if not path:
                raise RuntimeError('Firebase scanner credentials are not configured')
            options = {'httpTimeout': 5}
            if os.getenv('FIREBASE_DATABASE_URL'):
                options['databaseURL'] = os.environ['FIREBASE_DATABASE_URL']
            return firebase_admin.initialize_app(credentials.Certificate(path), options, name='scanner-auth')
