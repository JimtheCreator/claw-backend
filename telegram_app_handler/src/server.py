import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager

# Path setup (consider structuring as a package to avoid this)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# FIXED: Import from the correct path
from src.common.logger import logger

# Imports
from dotenv import load_dotenv
import redis
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from fastapi import FastAPI, Request, HTTPException, APIRouter
import uvicorn
from prometheus_client import Counter, Histogram

# Load environment variables at the very top
load_dotenv()

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WATCHERS_APK_URL = os.getenv("WATCHERS_APK_URL")

# --- DYNAMIC CONFIGURATION FOR FLY.IO ---
# FIXED: Consistent webhook endpoint
APP_BASE_URL = os.getenv("APP_PUBLIC_URL", "https://watchers-core-api.fly.dev")
WEBHOOK_URL = f"{APP_BASE_URL}/api/v1/watchers-telegram-server/webhook"  # Fixed to match router

# Get the port from an environment variable, defaulting to 8000 for this service
SERVER_PORT = int(os.getenv("TELEGRAM_PORT", "8001"))  # Different from main API

if not TELEGRAM_BOT_TOKEN:
    logger.error("⚠ TELEGRAM_BOT_TOKEN is not set! The bot will not work.")

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# --- Bot Messages & Content ---
GREETING_MESSAGE = """
Hey there! 👋 I'm **Frontman**, your guide to the **Watchers** app.

Watchers is a powerful Android app that helps traders like you by automatically detecting chart patterns, candlestick formations, and harmonic patterns. Think of it as your personal trading assistant.

What would you like to do?
"""

HELP_MESSAGE = """
I'm here to help! Here's what you can ask me to do:

**Commands:**
- `/start` - See my welcome message and main options.
- `/download` - Get the latest version of the Watchers app.
- `/plans` - See our subscription plans.
- `/help` - Show this help message.
- `/support` - Get help with an issue.

You can also just chat with me using the menu buttons!
"""

PLANS_MESSAGE = """
We have a few options to get you started with Watchers:

**1. Free Plan**
- Access to basic pattern recognition.
- A great way to see how the app works.

**2. Test Drive Plan (£1.99)**
- A one-time purchase to unlock all premium features for a limited time.
- Perfect for seeing the full power of Watchers before committing.

**3. Subscription Plans**
- Unlock all features, including advanced harmonic patterns and real-time alerts.
- Billed monthly or annually for a discount.

Ready to get started? Download the app and check out the plans inside!
"""

DOWNLOAD_GUIDE = """
Here is the latest version of the **Watchers** app.

**Installation Guide:**
1. Tap the file above to download it.
2. Once downloaded, open the file.
3. Your phone might ask for permission to "install from unknown sources." Please enable this for the installation to proceed. This is standard for apps installed outside the Play Store.
4. Follow the on-screen instructions.

Enjoy! Let me know if you run into any trouble.
"""

SUPPORT_MESSAGE = """
Need help? I'd love to assist you! 🤝

For the best support experience, please reach out to our dedicated support specialist:

👤 **@KateSolves**

Kate is our expert support agent who can help you with:
- Technical issues with the Watchers app
- Account and subscription questions
- Trading guidance and app features
- Installation troubleshooting

Just send her a message directly - she's super helpful and knows the app inside out! 😊
"""

# --- Initializations ---
redis_client = redis.from_url(REDIS_URL)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# --- Metrics ---
message_counter = Counter('telegram_messages_total', 'Total messages processed')
response_time = Histogram('telegram_response_seconds', 'Response time')

# --- Bot Command Handlers ---
async def handle_start_command(chat_id: int, message_id: int = None):
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
        text=GREETING_MESSAGE,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_help_command(chat_id: int):
    """Handle /help command."""
    await bot.send_message(
        chat_id=chat_id,
        text=HELP_MESSAGE,
        parse_mode='Markdown'
    )

async def handle_plans_command(chat_id: int):
    """Handle /plans command."""
    await bot.send_message(
        chat_id=chat_id,
        text=PLANS_MESSAGE,
        parse_mode='Markdown'
    )

async def handle_download_command(chat_id: int):
    """Handle /download command."""
    if not WATCHERS_APK_URL:
        await bot.send_message(
            chat_id=chat_id,
            text="Sorry, the download link isn't configured right now. Please check back later or contact @KateSolves for support."
        )
        return

    await bot.send_message(
        chat_id=chat_id,
        text="Great! I'm fetching the latest version for you now..."
    )
    
    try:
        await bot.send_document(
            chat_id=chat_id,
            document=WATCHERS_APK_URL,
            filename="Watchers-latest.apk",
            caption=DOWNLOAD_GUIDE,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to send APK file: {e}")
        await bot.send_message(
            chat_id=chat_id,
            text=f"I couldn't send the file directly. Here is the download link instead:\n{WATCHERS_APK_URL}"
        )

async def handle_support_command(chat_id: int):
    """Handle /support command - direct to @KateSolves."""
    await bot.send_message(
        chat_id=chat_id,
        text=SUPPORT_MESSAGE,
        parse_mode='Markdown'
    )

async def handle_callback_query(callback_query):
    """Handle inline button callbacks."""
    chat_id = callback_query.message.chat.id
    data = callback_query.data
    
    # Acknowledge the callback
    await bot.answer_callback_query(callback_query.id)
    
    if data == "download":
        await handle_download_command(chat_id)
    elif data == "plans":
        await handle_plans_command(chat_id)
    elif data == "help":
        await handle_help_command(chat_id)
    elif data == "support":
        await handle_support_command(chat_id)

async def handle_unknown_command(chat_id: int):
    """Handle unknown commands or messages."""
    keyboard = [
        [InlineKeyboardButton("🏠 Main Menu", callback_data="start")],
        [InlineKeyboardButton("🆘 Support", callback_data="support")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await bot.send_message(
        chat_id=chat_id,
        text="I'm not sure how to help with that. You can use the menu below or contact @KateSolves for support.",
        reply_markup=reply_markup
    )

async def process_update(update_data: dict):
    """Process a Telegram update."""
    try:
        update = Update.de_json(update_data, bot)
        if not update:
            logger.warning("Received invalid update data")
            return

        message_counter.inc()
        
        # Handle callback queries (inline button presses)
        if update.callback_query:
            await handle_callback_query(update.callback_query)
            return

        # Handle regular messages
        if update.message:
            chat_id = update.message.chat.id
            text = update.message.text
            
            if not text:
                return
            
            # Handle commands
            if text.startswith('/start'):
                await handle_start_command(chat_id, update.message.message_id)
            elif text.startswith('/help'):
                await handle_help_command(chat_id)
            elif text.startswith('/plans'):
                await handle_plans_command(chat_id)
            elif text.startswith('/download'):
                await handle_download_command(chat_id)
            elif text.startswith('/support'):
                await handle_support_command(chat_id)
            else:
                # Handle any other text as unknown
                await handle_unknown_command(chat_id)
                
    except Exception as e:
        logger.error(f"Error processing update: {e}")

# --- FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up Telegram bot server...")
    try:
        webhook_info_data = await bot.get_webhook_info()
        logger.info(f"Current webhook URL: {webhook_info_data.url}")
        logger.info(f"Desired webhook URL: {WEBHOOK_URL}")
        
        if webhook_info_data.url != WEBHOOK_URL:
            logger.warning("Webhook URL mismatch. Setting new webhook...")
            success = await bot.set_webhook(url=WEBHOOK_URL)
            if success:
                logger.info("✅ Webhook set successfully!")
            else:
                logger.error("⚠ Failed to set webhook")
        else:
            logger.info("✅ Webhook is already set correctly.")
    except Exception as e:
        logger.error(f"⚠ Startup error during webhook setup: {e}")
    yield
    logger.info("🚪 Shutting down Telegram bot server...")

app = FastAPI(
    title="Telegram Bot Server",
    version="1.0.0",
    lifespan=lifespan
)

router = APIRouter()

@router.post("/webhook")
async def webhook_handler(request: Request):
    try:
        json_data = await request.json()
        logger.info(f"🎯 Webhook received: {len(str(json_data))} chars")
        await process_update(json_data)
        return {"status": "processed"}
    except Exception as e:
        logger.error(f"⚠ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/webhook-info")
async def webhook_info():
    """Check current webhook status."""
    try:
        return await bot.get_webhook_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-webhook")
async def set_webhook_endpoint():
    """Manually set webhook."""
    try:
        success = await bot.set_webhook(url=WEBHOOK_URL)
        if success:
            return {"status": "success", "webhook_url": WEBHOOK_URL}
        else:
            raise HTTPException(status_code=500, detail="Failed to set webhook")
    except Exception as e:
        logger.error(f"Error setting webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Root and Health Check Endpoints ---
@router.get("/")
async def root():
    return {"status": "healthy", "message": "Telegram Bot Server is running"}

@router.get("/health")
async def health():
    return {"status": "ok", "bot_token_set": bool(TELEGRAM_BOT_TOKEN)}


# FIXED: Use consistent prefix
app.include_router(router, prefix="/api/v1/watchers-telegram-server")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("🤖 Starting Telegram Bot Server for local development...")
        print(f"🔗 Listening on port: {SERVER_PORT}")
        print(f"🎯 Full Webhook URL will be set to: {WEBHOOK_URL}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=SERVER_PORT,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start telegram server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)