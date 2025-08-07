# src/celery_worker.py
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Define the Redis URL for Celery broker
REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"

# Create a Celery instance
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL, # Backend to store task results
    include=['src.core.services.tasks'] # Point to the new tasks module
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Add Celery Beat schedule for periodic tasks
    # beat_schedule={
    #     # High-frequency intervals - verify more often
    #     'verify-1m-data': {
    #         # OLD TASK: 'task': 'src.core.services.tasks.verify_and_backfill_data_task',
    #         # NEW TASK:
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 3600.0,
    #         'args': ("1m",),
    #     },
    #     'verify-5m-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 7200.0,  # Every 2 hours
    #         'args': ("5m",),
    #     },
    #     'verify-15m-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 14400.0,  # Every 4 hours
    #         'args': ("15m",),
    #     },
    #     'verify-30m-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 21600.0,  # Every 6 hours
    #         'args': ("30m",),
    #     },
        
    #     # Medium-frequency intervals - verify daily
    #     'verify-1h-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 86400.0,  # Every day
    #         'args': ("1h",),
    #     },
    #     'verify-2h-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 86400.0,  # Every day
    #         'args': ("2h",),
    #     },
    #     'verify-4h-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 86400.0,  # Every day
    #         'args': ("4h",),
    #     },
    #     'verify-6h-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 86400.0,  # Every day
    #         'args': ("6h",),
    #     },
        
    #     # Low-frequency intervals - verify less often
    #     'verify-1d-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 172800.0,  # Every 2 days
    #         'args': ("1d",),
    #     },
    #     'verify-3d-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 259200.0,  # Every 3 days
    #         'args': ("3d",),
    #     },
    #     'verify-1w-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 604800.0,  # Every week
    #         'args': ("1w",),
    #     },
    #     'verify-1M-data': {
    #         'task': 'src.core.services.tasks.dispatch_verification_for_interval',
    #         'schedule': 2592000.0,  # Every month (30 days)
    #         'args': ("1M",),
    #     },
    # }
)

# celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=4 --loglevel=info

# celery -A src.core.services.workers.celery_worker beat --loglevel=info