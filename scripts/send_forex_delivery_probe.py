"""Send clearly labeled price/pattern routing probes to the sole registered owner.

This checks Firebase/APNs and iPhone routing only. It never inserts a market
quote, detection, saved watch or price crossing into the production pipeline.
"""
import asyncio
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import uuid
import asyncpg
from dotenv import load_dotenv
from firebase_admin import messaging
from infrastructure.database.supabase.tls import database_tls_context
from infrastructure.database.firebase.scanner_app import scanner_firebase_app

async def main():
    load_dotenv('.env')
    connection=await asyncpg.connect(os.environ['SCANNER_WORKER_DATABASE_URL'],statement_cache_size=0,
        ssl=database_tls_context(os.getenv('SCANNER_DATABASE_CA_FILE')))
    try:
        await connection.execute('SET ROLE scanner_watch_worker')
        devices=await connection.fetch('SELECT user_id,token FROM scanner_alerts.devices')
    finally:
        await connection.close()
    if len({d['user_id'] for d in devices})!=1 or len(devices)!=1:
        raise RuntimeError('Probe requires exactly one registered device and owner')
    device=devices[0]; results=[]
    for kind in ('price','pattern'):
        identifier=str(uuid.uuid4())
        data=dict(type='price_alert' if kind=='price' else 'event_match',notification_id=identifier,
            user_id=device['user_id'],symbol='EURUSD',provider='massive',market='forex',
            universe='massive-forex',interval='1h',pattern_id='ascending_channel',diagnostic='true')
        message=messaging.Message(token=device['token'],data=data,
            notification=messaging.Notification(title=f'Watchers Forex {kind} test',
                body='Delivery test, not a market alert. Tap to open EURUSD'+(' · 1h.' if kind=='pattern' else '.')),
            apns=messaging.APNSConfig(headers={'apns-push-type':'alert','apns-priority':'10',
                'apns-expiration':str(int(time.time()+600)),'apns-collapse-id':identifier},
                payload=messaging.APNSPayload(messaging.Aps(sound='default',thread_id='watchers-forex-test',
                    category='WATCHERS_PRICE_ALERT' if kind=='price' else None))))
        started=time.monotonic()
        await asyncio.to_thread(messaging.send,message,app=scanner_firebase_app())
        results.append(dict(kind=kind,provider_accepted=True,seconds=round(time.monotonic()-started,3)))
    report=dict(sent_at=datetime.now(timezone.utc).isoformat(),checks=results,
        scope='delivery-and-routing-only',live_market_trigger_verified=False,phone_confirmation='pending')
    Path('logs/forex-phone-probe-20261004.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        print(json.dumps(dict(status='failed',error_type=type(exc).__name__)))
        raise SystemExit(1) from None
