# File: src/core/services/workers/celery_worker.py

import os
import sys
from celery import Celery
from dotenv import load_dotenv

# Path setup to ensure 'src' is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

load_dotenv()

REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    include=['src.core.services.tasks'],
    # Add these for better task handling
    task_routes={
        'telegram_bot.process_update': {'queue': 'telegram'},
    },
    worker_prefetch_multiplier=1,  # Helps with memory management
    task_acks_late=True,  # Better error handling
    worker_max_tasks_per_child=1000,  # Restart workers periodically
)

# celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=4 --loglevel=info