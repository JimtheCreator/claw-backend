import os
import sys
import json
import asyncio
import time
from contextlib import asynccontextmanager

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from src.common.logger import logger

# Imports
from dotenv import load_dotenv
import redis
from celery import Celery
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from fastapi import FastAPI, Request, HTTPException, APIRouter
import uvicorn
import asyncpg
from prometheus_client import Counter, Histogram, generate_latest

# Load environment variables at the very top
load_dotenv()

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WATCHERS_APK_URL = os.getenv("WATCHERS_APK_URL")  # URL to your APK file

if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN is not set! The bot will not work.")

# Make sure to update your NGROK URL to include the new prefix
WEBHOOK_URL = "https://stable-wholly-crappie.ngrok-free.app/api/v1/telegram/webhook"
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
app = FastAPI(
    title="Telegram Bot Server", 
    version="1.0.0",
    description="Handles Telegram bot webhooks and processes messages.",
    lifespan=lambda app: lifespan(app)
)

# Create an APIRouter to hold all the telegram-related endpoints
router = APIRouter()

@router.post("/webhook")
async def webhook_handler(request: Request):
    """Webhook handler that processes updates directly."""
    try:
        json_data = await request.json()
        logger.info(f"🎯 Webhook received data: {json.dumps(json_data, indent=2)}")
        
        # Process the update directly instead of queuing
        await process_update(json_data)
        
        return {"status": "processed"}
    
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
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

# Include the router in the main app with the desired prefix
app.include_router(router, prefix="/api/v1/telegram")

# --- Root and Health Check Endpoints ---
@app.get("/")
async def root():
    """Root health check endpoint."""
    return {"status": "healthy", "message": "Telegram Bot Server is running"}

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "ok", 
        "bot_token_set": bool(TELEGRAM_BOT_TOKEN),
        "apk_url_set": bool(WATCHERS_APK_URL)
    }

# --- Startup Logic ---
async def startup_tasks():
    """Run startup tasks."""
    logger.info("🚀 Starting up bot server...")
    try:
        webhook_info_data = await bot.get_webhook_info()
        if webhook_info_data.url != WEBHOOK_URL:
            logger.warning(f"Webhook URL mismatch. Current: {webhook_info_data.url}, Desired: {WEBHOOK_URL}")
            logger.info("Attempting to set new webhook...")
            await bot.set_webhook(url=WEBHOOK_URL)
            logger.info("✅ Webhook set successfully!")
        else:
            logger.info("✅ Webhook is already set correctly.")
    except Exception as e:
        logger.error(f"❌ Startup error during webhook setup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 Startup
    try:
        await startup_tasks()

    except Exception as e:
        logger.error(f"Failed to preload tickers: {e}")
        # In a real-world scenario, you might want to handle this more gracefully
        # For now, we'll let the application fail to start if critical services are unavailable
        raise

    yield  # 🧘 Everything after this happens at shutdown


# --- Main Execution ---
if __name__ == "__main__":
    print("🤖 Starting Telegram Bot Server...")
    print(f"🔗 Endpoint Prefix: /api/v1/telegram")
    print(f"🎯 Full Webhook URL: {WEBHOOK_URL}")
    print(f"🔑 Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing!'}")
    print(f"📦 APK URL: {'✅ Set' if WATCHERS_APK_URL else '❌ Missing!'}")
    print("🆘 Support: Users will be directed to @KateSolves")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )