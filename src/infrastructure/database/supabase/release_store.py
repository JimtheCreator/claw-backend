# Path: src/infrastructure/database/supabase/release_store.py
import os
from typing import List, Optional, Dict, Any, Generator
from supabase import create_client, Client
from common.logger import logger
import uuid

class ReleaseStore:
    """
    Handles all database operations related to APK releases and subscribers.
    """
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key must be set.")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        self.releases_table = "apk_releases"
        self.subscribers_table = "telegram_subscribers"

    def upsert_subscriber(self, chat_id: int, username: str, first_name: str, last_name: str) -> None:
        """
        Adds a new subscriber or updates an existing one, marking them as active.
        """
        try:
            subscriber_data = {
                "chat_id": chat_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "active": True
            }
            self.client.table(self.subscribers_table).upsert(subscriber_data).execute()
            logger.info(f"Upserted subscriber with chat_id: {chat_id}")
        except Exception as e:
            logger.error(f"Error upserting subscriber {chat_id}: {e}")
            raise

    def insert_release(self, version: str, file_id: str, changelog: str) -> Optional[uuid.UUID]:
        """
        Inserts a new APK release into the database and marks it as the latest.
        The database trigger will handle setting other releases 'is_latest' to false.
        """
        try:
            release_data = {
                "version": version,
                "file_id": file_id,
                "changelog": changelog,
                "is_latest": True
            }
            result = self.client.table(self.releases_table).insert(release_data).execute()
            
            if result.data:
                release_id = result.data[0]['id']
                logger.info(f"Inserted new release {version} with ID {release_id}")
                return uuid.UUID(release_id)
            return None
        except Exception as e:
            # Handle potential duplicate version error gracefully
            if "duplicate key value violates unique constraint" in str(e):
                logger.warning(f"Release version {version} already exists. Skipping insert.")
                return None
            logger.error(f"Error inserting release {version}: {e}")
            raise

    def get_latest_release(self) -> Optional[Dict[str, Any]]:
        """
        Fetches the single release marked as the latest from the database.
        """
        try:
            result = self.client.table(self.releases_table).select("*").eq("is_latest", True).limit(1).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching latest release: {e}")
            return None

    def get_release_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Fetches a specific release by its version string."""
        try:
            result = self.client.table(self.releases_table).select("*").eq("version", version).limit(1).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching release by version {version}: {e}")
            return None

    def get_active_subscriber_chat_ids_in_batches(self, batch_size: int = 200) -> Generator[list[int], None, None]:
        """
        Generator that yields batches of active subscriber chat_ids.
        This is memory-efficient for very large subscriber counts.
        """
        try:
            offset = 0
            while True:
                result = self.client.table(self.subscribers_table) \
                    .select("chat_id") \
                    .eq("active", True) \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                
                if not result.data:
                    break
                
                yield [item['chat_id'] for item in result.data]
                
                if len(result.data) < batch_size:
                    break
                
                offset += batch_size
        except Exception as e:
            logger.error(f"Error fetching subscriber batches: {e}")

    def mark_subscriber_inactive(self, chat_id: int) -> None:
        """
        Marks a subscriber as inactive, likely because they blocked the bot.
        """
        try:
            self.client.table(self.subscribers_table) \
                .update({"active": False}) \
                .eq("chat_id", chat_id) \
                .execute()
            logger.info(f"Marked subscriber {chat_id} as inactive.")
        except Exception as e:
            logger.error(f"Error marking subscriber {chat_id} inactive: {e}")