# Path: src/bot/release_pipeline.py
import os
import re
from typing import List, Optional
from telegram.ext import ContextTypes
from src.common.logger import logger
from src.infrastructure.database.supabase.release_store import ReleaseStore
from src.core.services.broadcast_task import dispatch_release_notifications
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup

# --- Configuration from Environment Variables ---
AUTHORIZED_UPLOADER = os.getenv("AUTHORIZED_UPLOADER", "WatchersAssistant")  # Without @
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))
MAX_APK_SIZE_MB = int(os.getenv("MAX_APK_SIZE_MB", "200"))

db = ReleaseStore()

# --- Updated Utility Functions ---
def _extract_version_from_filename(filename: str) -> Optional[str]:
    """
    Extracts version from filename. Supports multiple formats:
    - AppName-v1.2.3.apk
    - AppName-1.2.3.apk  
    - AppName_v1.2.3.apk
    - Or returns None if no clear version pattern
    """
    if not filename.lower().endswith(".apk"):
        return None
    
    base_name = filename[:-4]  # Remove .apk
    
    # Try to find version patterns
    import re
    
    # Look for version patterns like v1.2.3, 1.2.3, etc.
    version_patterns = [
        r'[-_]v?(\d+\.\d+\.\d+)$',  # Ends with -v1.2.3 or -1.2.3
        r'[-_]v?(\d+\.\d+)$',       # Ends with -v1.2 or -1.2
        r'[-_]v?(\d+)$'             # Ends with -v1 or -1
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, base_name)
        if match:
            return match.group(1)
    
    return None

def _extract_version_from_caption(caption: str) -> Optional[str]:
    """
    Extracts version from caption. Looks for patterns like:
    - Version: 1.2.3
    - v1.2.3
    - Version 1.2.3
    """
    if not caption:
        return None
    
    import re
    version_patterns = [
        r'version:?\s*v?(\d+\.\d+\.\d+)',
        r'\bv(\d+\.\d+\.\d+)\b',
        r'(\d+\.\d+\.\d+)'
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

# --- Updated Message Handler for APK Uploads ---
async def handle_apk_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles new APK file uploads from authorized users.
    Now uses username-based authentication instead of filename passwords.
    """
    uploader_username = update.effective_user.username
    uploader_id = update.effective_user.id
    
    # 1. Username-based authentication
    if not uploader_username:
        logger.warning(f"APK upload attempt by user {uploader_id} with no username")
        await update.message.reply_text(
            "❌ What you tryna do pal??? Access denied!"
        )
        return
    
    if uploader_username != AUTHORIZED_UPLOADER:
        logger.warning(f"Unauthorized APK upload attempt by user {uploader_id} (@{uploader_username})")
        await update.message.reply_text(
            f"❌ Unauthorized. Only @{AUTHORIZED_UPLOADER} can upload releases."
        )
        return
    
    # 2. Basic validation
    if not update.message.document:
        return
    
    document = update.message.document
    if document.file_size > MAX_APK_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"❌ File size exceeds the {MAX_APK_SIZE_MB}MB limit.")
        return
    
    if not document.file_name.lower().endswith('.apk'):
        await update.message.reply_text("❌ We don't need to make our conversation complicated. No file uploads please.")
        return

    logger.info(f"Authorized APK upload received from @{uploader_username} (ID: {uploader_id})")

    # 3. Version extraction (try filename first, then caption, then prompt)
    version = _extract_version_from_filename(document.file_name)
    
    if not version:
        version = _extract_version_from_caption(update.message.caption)
    
    if not version:
        await update.message.reply_text(
            "❌ Could not determine version from filename or caption.\n\n"
            "Please either:\n"
            "• Include version in filename: `AppName-1.2.3.apk`\n"
            "• Or mention version in caption: `Version: 1.2.3`"
        )
        return

    # 4. Process the upload
    file_id = document.file_id
    changelog = update.message.caption or f"Version {version} release"
    
    # Remove version info from changelog if it was extracted from there
    if update.message.caption:
        import re
        # Clean up common version patterns from changelog
        cleaned_changelog = re.sub(r'version:?\s*v?\d+\.\d+\.\d+\s*', '', 
                                 changelog, flags=re.IGNORECASE).strip()
        if cleaned_changelog:
            changelog = cleaned_changelog
        else:
            changelog = f"Version {version} release"

    release_id = db.insert_release(version, file_id, changelog)

    if release_id:
        # 5. Enqueue broadcast task
        dispatch_release_notifications.delay(str(release_id))
        await update.message.reply_text(
            f"✅ Successfully registered version *{version}*.\n"
            f"📤 Notification broadcast queued for all subscribers.\n\n"
            f"📁 Original filename: `{document.file_name}`\n"
            f"📝 Changelog: _{changelog}_",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"⚠️ Version *{version}* already exists in the database.\n"
            f"The release was not updated or re-broadcasted.",
            parse_mode='Markdown'
        )

# --- Command Handlers ---
async def handle_start_command(update: Update, bot, context: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int] = None):
    """
    Handles the /start command. Upserts the user into the subscribers list
    and sends a welcome message.
    """
    user = update.effective_user
    db.upsert_subscriber(
        chat_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    welcome_message = (
        "Hey there! 👋 I'm **Frontman**, your guide to the **Watchers** app.\n\n"
        "I'm here to provide you with the latest app updates. "
        "You are now subscribed to release notifications.\n\n"
        "Use the /download command at any time to get the latest version of the app."
    )

    """Handle /start command with inline keyboard."""
    keyboard = [
        [InlineKeyboardButton("📲 Download App", callback_data="download")],
        [InlineKeyboardButton("📊 View Plans", callback_data="plans")],
        [InlineKeyboardButton("❓ Get Help", callback_data="help")],
        [InlineKeyboardButton("🆘 Support", callback_data="support")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await bot.send_message(
        chat_id=chat_id,
        text=welcome_message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def handle_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /download command. Fetches the latest release and sends the APK
    using the stored file_id.
    """
    chat_id = update.effective_chat.id
    latest_release = db.get_latest_release()
    
    if not latest_release:
        await update.message.reply_text("Sorry, no release is available at the moment. Please check back later.")
        return

    caption = (
        f"📲 *Watchers v{latest_release['version']}*\n\n"
        f"Changelog:\n_{latest_release['changelog']}_"
    )
    
    logger.info(f"Sending file_id {latest_release['file_id']} to chat_id {chat_id}")
    await context.bot.send_document(
        chat_id=chat_id,
        document=latest_release['file_id'],
        caption=caption,
        parse_mode='Markdown'
    )

async def handle_force_broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin-only command to re-broadcast a specific version.
    Usage: /force_broadcast 1.2.3
    """
    user_id = update.effective_user.id
    if user_id != ADMIN_CHAT_ID:
        await update.message.reply_text("You are not authorized to use this command.")
        return
        
    try:
        version = context.args[0]
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /force_broadcast <version>")
        return

    release = db.get_release_by_version(version)
    if not release:
        await update.message.reply_text(f"Version '{version}' not found in the database.")
        return

    release_id = str(release['id'])
    dispatch_release_notifications.delay(release_id)
    await update.message.reply_text(f"✅ Broadcast for version {version} has been queued.")