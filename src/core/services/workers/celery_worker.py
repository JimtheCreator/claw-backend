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


celery_app = Celery("tasks", broker=os.getenv('REDIS_URL'), backend=os.getenv('REDIS_URL'))

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    include=[
        'src.core.services.tasks',
        'src.core.services.broadcast_task' # Add this line
    ],
    task_routes={
        # Telegram tasks - lightweight, use default queue
        'src.core.services.tasks.process_telegram_update': {'queue': 'telegram'},
        
        # Data fetching tasks - I/O bound, use default queue
        'src.core.services.tasks.save_market_data_task': {'queue': 'default'},
        'src.core.services.tasks.verify_and_backfill_data_task': {'queue': 'default'},
        'src.core.services.tasks.verify_single_symbol_task': {'queue': 'default'},
        'src.core.services.tasks.dispatch_verification_for_interval': {'queue': 'default'},
        
        # Analysis tasks - CPU intensive, use dedicated queue
        'src.core.services.tasks.analyze_trendlines_task': {'queue': 'analysis'},   
        'src.core.services.tasks.analyze_sr_task': {'queue': 'analysis'},

        # Broadcast tasks - I/O bound, high volume, dedicated queue
        'src.core.services.broadcast_task.dispatch_release_notifications': {'queue': 'broadcasts'},
        'src.core.services.broadcast_task.broadcast_chunk': {'queue': 'broadcasts'}
    },

    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,  # Reduced for memory-heavy analysis tasks
    
    # Add task time limits to prevent runaway tasks
    task_time_limit=600,  # 10 minutes hard limit
    task_soft_time_limit=480,  # 8 minutes soft limit
    
    # Configure different settings for different queues
    task_annotations={
        'src.core.services.tasks.analyze_trendlines_task': {
            'time_limit': 900,  # 15 minutes for complex trendline analysis
            'soft_time_limit': 720,  # 12 minutes soft limit
        },
        'src.core.services.tasks.analyze_sr_task': {
            'time_limit': 300,  # 5 minutes for S/R analysis
            'soft_time_limit': 240,  # 4 minutes soft limit
        },
    }
)


# Commands to run workers:

# For general tasks (Telegram, data fetching):
# celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=4 --loglevel=info --queues=default,telegram

# For CPU-intensive analysis tasks (run on separate machine/container if possible):
# celery -A src.core.services.workers.celery_worker worker --pool=processes --concurrency=2 --loglevel=info --queues=analysis

# To run both queues on the same machine:
# celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=4 --loglevel=info --queues=default,telegram,analysis

# Purge all queues:
# celery -A src.core.services.workers.celery_worker purge -f

# Monitor tasks:
# celery -A src.core.services.workers.celery_worker flower

# For broadcast tasks (can handle high I/O):
# celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=10 --loglevel=info --queues=broadcasts