# bot/handlers/apk_release.py
import os
import re
import asyncio
from typing import Optional
from telegram import Update, Bot, Document
from telegram.ext import ContextTypes
from src.common.logger import logger
from tasks.broadcast_release import broadcast_release
from db.release_store import ReleaseStore

class APKReleaseHandler:
    def __init__(self, release_store: ReleaseStore):
        self.release_store = release_store
        self.upload_passwords = self._get_upload_passwords()
        self.max_file_size = int(os.getenv("APK_MAX_FILE_SIZE", "100")) * 1024 * 1024  # Default 100MB
        self.admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        
    def _get_upload_passwords(self) -> set:
        """Get allowed upload passwords from environment variables."""
        passwords_env = os.getenv("APK_UPLOAD_PASSWORDS", os.getenv("APK_UPLOAD_PASSWORD", ""))
        if not passwords_env:
            logger.warning("No APK upload passwords configured!")
            return set()
        return set(pwd.strip() for pwd in passwords_env.split(",") if pwd.strip())
    
    def _parse_apk_filename(self, filename: str) -> Optional[dict]:
        """
        Parse APK filename: appName-password-versionName.apk
        Returns dict with parsed components or None if invalid.
        """
        if not filename.endswith('.apk'):
            return None
            
        # Remove .apk extension
        base_name = filename[:-4]
        
        # Split by '-' and work backwards
        parts = base_name.split('-')
        if len(parts) < 3:
            return None
            
        # Last part is version, second to last is password, rest is app name
        version_name = parts[-1]
        password = parts[-2]
        app_name = '-'.join(parts[:-2])
        
        return {
            "app_name": app_name,
            "password": password,
            "version": version_name
        }
    
    def _validate_version_format(self, version: str) -> bool:
        """Validate version format (e.g., 1.0.0, 1.2.3-beta)."""
        version_pattern = r'^(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9]+)?$'
        return bool(re.match(version_pattern, version))
    
    async def _send_error_message(self, chat_id: int, error_msg: str):
        """Send error message to uploader."""
        try:
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ **APK Upload Failed**\n\n{error_msg}\n\n"
                     f"Please follow the format: `appName-password-versionName.apk`"
            )
        except Exception as e:
            logger.error(f"Failed to send error message to {chat_id}: {e}")
    
    async def _log_to_admin(self, message: str):
        """Log important events to admin if configured."""
        if not self.admin_chat_id:
            return
            
        try:
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
            await bot.send_message(
                chat_id=self.admin_chat_id,
                text=f"🔔 **APK Release System**\n\n{message}"
            )
        except Exception as e:
            logger.error(f"Failed to log to admin: {e}")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming document messages (potential APK uploads)."""
        if not update.message or not update.message.document:
            return
            
        document: Document = update.message.document
        chat_id = update.message.chat.id
        
        # Basic validation
        if not document.file_name or not document.file_name.endswith('.apk'):
            logger.debug(f"Non-APK file received from {chat_id}: {document.file_name}")
            return
        
        # File size check
        if document.file_size and document.file_size > self.max_file_size:
            size_mb = document.file_size / (1024 * 1024)
            await self._send_error_message(
                chat_id, 
                f"File too large: {size_mb:.1f}MB (max: {self.max_file_size/(1024*1024):.0f}MB)"
            )
            return
        
        # Parse filename
        parsed = self._parse_apk_filename(document.file_name)
        if not parsed:
            await self._send_error_message(
                chat_id,
                f"Invalid filename format: `{document.file_name}`\n"
                f"Expected format: `appName-password-versionName.apk`\n"
                f"Example: `Watchers-mypass123-1.0.0.apk`"
            )
            return
        
        # Verify password
        if not self.upload_passwords or parsed["password"] not in self.upload_passwords:
            await self._send_error_message(
                chat_id,
                "Authentication failed. Invalid upload password."
            )
            await self._log_to_admin(f"Failed upload attempt from {chat_id}: Invalid password in {document.file_name}")
            return
        
        # Validate version format
        if not self._validate_version_format(parsed["version"]):
            await self._send_error_message(
                chat_id,
                f"Invalid version format: `{parsed['version']}`\n"
                f"Expected format: `X.Y.Z` or `X.Y.Z-suffix`\n"
                f"Examples: `1.0.0`, `2.1.3-beta`"
            )
            return
        
        # Extract changelog from caption
        changelog = "No changelog provided"
        if update.message.caption and update.message.caption.strip():
            changelog = update.message.caption.strip()
        
        try:
            # **CRITICAL: Always capture the fresh file_id**
            file_id = document.file_id
            version = parsed["version"]
            app_name = parsed["app_name"]
            
            logger.info(f"Processing APK upload: {app_name} v{version} (file_id: {file_id})")
            
            # Check for existing version and handle accordingly
            existing_release = await self.release_store.get_release_by_version(version)
            force_update = False  # Could be extended to parse from caption like "force=true"
            
            if existing_release and not force_update:
                # Update changelog if different, but don't broadcast
                if existing_release.get("changelog") != changelog:
                    await self.release_store.update_release_changelog(existing_release["id"], changelog)
                    await self._send_confirmation(chat_id, version, "changelog updated", broadcast=False)
                else:
                    await self._send_confirmation(chat_id, version, "already exists", broadcast=False)
                return
            
            # **ALWAYS INSERT/UPDATE WITH FRESH FILE_ID**
            release_id = await self.release_store.insert_release(
                version=version,
                file_id=file_id,  # Fresh file_id every time
                changelog=changelog,
                app_name=app_name,
                is_latest=True
            )
            
            await self._send_confirmation(chat_id, version, "uploaded successfully", broadcast=True)
            await self._log_to_admin(f"New release uploaded: {app_name} v{version} by user {chat_id}")
            
            # Enqueue broadcast task ONLY after successful DB insert
            try:
                broadcast_release.delay(release_id)
                logger.info(f"Enqueued broadcast task for release {release_id}")
            except Exception as e:
                logger.error(f"Failed to enqueue broadcast task: {e}")
                await self._log_to_admin(f"⚠️ Broadcast task failed for {version}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing APK upload: {e}")
            await self._send_error_message(
                chat_id,
                f"Upload processing failed: {str(e)}\nPlease try again or contact support."
            )
            await self._log_to_admin(f"❌ Upload processing error for {document.file_name}: {e}")
    
    async def _send_confirmation(self, chat_id: int, version: str, status: str, broadcast: bool):
        """Send confirmation message to uploader."""
        try:
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
            
            broadcast_msg = "\n🔔 **Broadcast notification will be sent to subscribers**" if broadcast else ""
            
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ **APK Release: v{version}**\n\n"
                     f"Status: {status.title()}{broadcast_msg}\n\n"
                     f"Users can now download via `/download` command.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send confirmation to {chat_id}: {e}")
    
    async def handle_force_broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle admin-only force broadcast command: /force_broadcast <version>"""
        if not self.admin_chat_id or str(update.message.chat.id) != self.admin_chat_id:
            return  # Silently ignore non-admin users
        
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "Usage: `/force_broadcast <version>`\nExample: `/force_broadcast 1.0.0`"
            )
            return
        
        version = context.args[0]
        
        try:
            release = await self.release_store.get_release_by_version(version)
            if not release:
                await update.message.reply_text(f"❌ Release v{version} not found")
                return
            
            # Force broadcast
            broadcast_release.delay(release["id"])
            await update.message.reply_text(f"✅ Force broadcast initiated for v{version}")
            logger.info(f"Admin {update.message.chat.id} initiated force broadcast for v{version}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Force broadcast failed: {e}")
            logger.error(f"Force broadcast error: {e}")