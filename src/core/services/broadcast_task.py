# Path: src/core/services/tasks/broadcast_task.py

import os
import asyncio
from typing import List
from celery.exceptions import MaxRetriesExceededError
from telegram import Bot
from telegram.error import Forbidden, TelegramError

from src.core.services.workers.celery_worker import celery_app
from src.infrastructure.database.supabase.release_store import ReleaseStore
from common.logger import logger

# --- Configuration from Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MESSAGES_PER_SECOND = int(os.getenv("MESSAGES_PER_SECOND", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
BROADCAST_DRY_RUN = os.getenv("BROADCAST_DRY_RUN", "false").lower() == "true"
MAX_RETRIES = int(os.getenv("BROADCAST_MAX_RETRIES", "3"))

# --- Helper to run async code from sync Celery task ---
def run_async(coro):
    """A helper to run an async coroutine in a new event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

@celery_app.task(name="src.core.services.broadcast_task.dispatch_release_notifications")
def dispatch_release_notifications(release_id: str):
    """
    Main task to start the broadcast.
    It fetches subscriber chat_ids in batches and creates a subtask for each batch.
    """
    logger.info(f"Starting broadcast dispatch for release_id: {release_id}...")
    db = ReleaseStore()
    release = db.get_latest_release() # Assuming the latest at time of dispatch
    
    if not release or str(release.get('id')) != release_id:
        logger.error(f"Release {release_id} not found or is not the latest. Aborting broadcast.")
        return

    message_text = f"🎉 *New Watchers Update: v{release['version']}*\n\n"
    message_text += f"_{release['changelog']}_\n\n"
    message_text += "To get the latest version, open our chat and use the /download command."

    # Fetch subscribers in chunks and dispatch subtasks
    batch_count = 0
    for chat_id_batch in db.get_active_subscriber_chat_ids_in_batches(BATCH_SIZE):
        broadcast_chunk.delay(chat_id_batch, message_text)
        batch_count += 1
        logger.info(f"Dispatched broadcast chunk #{batch_count} with {len(chat_id_batch)} subscribers.")
    
    logger.info(f"Completed dispatch of {batch_count} chunks for release {release_id}.")

@celery_app.task(
    name="src.core.services.broadcast_task.broadcast_chunk",
    bind=True,
    autoretry_for=(TelegramError,),
    retry_kwargs={'max_retries': MAX_RETRIES},
    retry_backoff=True,
    retry_backoff_max=60
)
def broadcast_chunk(self, chat_ids: List[int], message: str):
    """
    Celery subtask that sends a message to a chunk of subscribers.
    Handles rate limiting and marks users as inactive if they've blocked the bot.
    """
    logger.info(f"Processing broadcast chunk of {len(chat_ids)} users. Dry run: {BROADCAST_DRY_RUN}")
    
    # This coroutine will be run by the sync wrapper
    async def _send_messages():
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        db = ReleaseStore()
        inactive_count = 0
        
        for chat_id in chat_ids:
            if BROADCAST_DRY_RUN:
                logger.info(f"[DRY RUN] Would send message to chat_id: {chat_id}")
            else:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Forbidden:
                    logger.warning(f"Bot is blocked by chat_id: {chat_id}. Marking as inactive.")
                    db.mark_subscriber_inactive(chat_id)
                    inactive_count += 1
                except TelegramError as e:
                    logger.error(f"Telegram API error for chat_id {chat_id}: {e}. Task will be retried.")
                    # Let Celery's autoretry handle this
                    raise
            
            # Rate limiting delay
            await asyncio.sleep(1 / MESSAGES_PER_SECOND)
            
        logger.info(f"Chunk completed. Found {inactive_count} inactive users.")

    try:
        run_async(_send_messages())
    except MaxRetriesExceededError:
        logger.critical(f"Broadcast chunk failed after {MAX_RETRIES} retries. Some users may not have been notified.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in broadcast_chunk: {e}", exc_info=True)
        # Manually retry for non-Telegram errors if needed, but be cautious
        # self.retry(exc=e, countdown=60)