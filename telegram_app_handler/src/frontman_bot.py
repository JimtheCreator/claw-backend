# frontman_bot.py
#
# Production-ready Python code for the Frontman Telegram bot.
# This bot serves as the first point of contact for the Watchers trading app.
#
# How to Run:
# 1. Install dependencies:
#    pip install python-telegram-bot python-dotenv httpx concurrent-log-handler
#
# 2. Create a `.env` file in the same directory with the following content:
#    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
#    OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
#    WATCHERS_APK_URL="YOUR_DIRECT_DOWNLOAD_URL_FOR_THE_APK"
#    # Example: WATCHERS_APK_URL="https://your-domain.com/watchers-latest.apk"
#
# 3. Run the bot from your terminal:
#    python frontman_bot.py
#

import os
import sys
# If logger is in D:\ClawInc\Claw-Backend\src\common\logger.py
current_dir = os.path.dirname(os.path.abspath(__file__))  # telegram/src
parent_dir = os.path.dirname(current_dir)  # telegram
grandparent_dir = os.path.dirname(parent_dir)  # Claw-Backend
sys.path.append(grandparent_dir)

from src.common.logger import logger

import httpx
import asyncio
from dotenv import load_dotenv
from telegram_app_handler import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)

# --- Load Environment Variables ---
# Safely load environment variables from a .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("DEEPSEEK_R1_OPENROUTER_API_KEY")
WATCHERS_APK_URL = os.getenv("WATCHERS_APK_URL") # URL to your APK file


# --- Constants & Configurable Text ---
# Keep all user-facing text here for easy updates.

# Bot Persona & Core Messages
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

You can also just chat with me. If you have a question about trading or the app, just ask!
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

# --- Deepseek AI Integration ---
OPENROUTER_API_URL = os.getenv("DEEPSEEK_R1_OPENROUTER_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

# Rate limiting - track last API call time and user cooldowns
last_api_call = 0
user_last_request = {}  # Track last request time per user
MIN_REQUEST_INTERVAL = 2  # seconds between requests
USER_COOLDOWN = 10  # seconds between requests per user

async def get_deepseek_response(query: str) -> str:
    """
    Sends a query to the Deepseek model via OpenRouter and returns the response.
    """
    global last_api_call
    
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set. AI features are disabled.")
        return "I'm sorry, but my connection to the AI brain is currently offline. Please try again later."

    # Rate limiting - ensure minimum time between requests
    import time
    current_time = time.time()
    time_since_last_call = current_time - last_api_call
    if time_since_last_call < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last_call
        logger.info(f"Rate limiting: waiting {sleep_time:.1f} seconds before API call")
        await asyncio.sleep(sleep_time)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",  # Optional: helps with rate limits
        "X-Title": "Watchers Trading Bot",  # Optional: helps with rate limits
    }
    # The system prompt shapes the AI's personality and goals.
    system_prompt = (
        "You are Frontman, a friendly and professional support assistant for the Watchers trading app. "
        "Your goal is to help users, answer their questions about trading and the app, and encourage them to use Watchers. "
        "Keep your answers concise, clear, and helpful. Use emojis sparingly. "
        "The app's name is 'Watchers'."
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": 500,  # Limit response length to reduce costs/usage
    }

    try:
        async with httpx.AsyncClient() as client:
            last_api_call = time.time()  # Update the last call time
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 429:
                logger.warning("Hit rate limit. Telling user to wait.")
                return "I'm receiving too many requests right now. Please wait a moment and try again."
            
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limited by OpenRouter API")
            return "I'm busy helping other users right now. Please wait 30 seconds and try again."
        logger.error(f"HTTP error occurred while calling OpenRouter: {e}")
        return "I had trouble connecting to the AI service. The server might be busy. Please try again in a moment."
    except Exception as e:
        logger.error(f"An unexpected error occurred during AI call: {e}")
        return "An unexpected error occurred. I've logged the issue for the development team."


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command and displays the main menu."""
    keyboard = [
        [InlineKeyboardButton("📲 Download App", callback_data="download")],
        [InlineKeyboardButton("📊 View Plans", callback_data="plans")],
        [InlineKeyboardButton("❓ Get Help", callback_data="help")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(GREETING_MESSAGE, reply_markup=reply_markup, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    await update.message.reply_text(HELP_MESSAGE, parse_mode='Markdown')

async def plans_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /plans command."""
    await update.message.reply_text(PLANS_MESSAGE, parse_mode='Markdown')

async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /download command by sending the APK."""
    if not WATCHERS_APK_URL:
        await update.message.reply_text(
            "Sorry, the download link isn't configured right now. Please check back later or contact support."
        )
        return

    await update.message.reply_text("Great! I'm fetching the latest version for you now...")
    
    # Send the APK file from the URL. Telegram will handle the download and upload.
    # We send it as a document.
    try:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=WATCHERS_APK_URL,
            filename="Watchers-latest.apk",
            caption=DOWNLOAD_GUIDE,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to send APK file: {e}")
        await update.message.reply_text(
            "I couldn't send the file directly. Here is the download link instead:\n"
            f"{WATCHERS_APK_URL}"
        )


# --- Support Conversation Handler ---
SUPPORT_ENTRY = 0

async def support_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the support conversation flow."""
    await update.message.reply_text("Of course. Please describe your issue or question in detail. I'll do my best to help you out.")
    return SUPPORT_ENTRY

async def handle_support_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes the user's support query using Deepseek."""
    user_query = update.message.text
    await update.message.reply_text("Thanks. I'm analyzing your query now, please wait a moment...")

    # Get the AI-powered response
    ai_response = await get_deepseek_response(user_query)
    
    await update.message.reply_text(ai_response)
    await update.message.reply_text("Was this helpful? If not, feel free to ask another question or rephrase your issue.")
    
    return ConversationHandler.END

async def cancel_support(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the support conversation."""
    await update.message.reply_text("Support request cancelled. Let me know if you need anything else!")
    return ConversationHandler.END


# --- General Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general text messages that are not commands."""
    message_text = update.message.text.lower()
    user_id = update.effective_user.id
    
    # Simple keyword matching for common, predefined queries
    if any(keyword in message_text for keyword in ["download", "apk", "install"]):
        await download_command(update, context)
        return
    if "what is watchers" in message_text or "what's watchers" in message_text:
        await update.message.reply_text(
            "Watchers is an Android app that helps traders by automatically detecting chart patterns. "
            "You can use it to spot opportunities in the market. Want to give it a try?"
        )
        return

    # If it's not a simple query, treat it as a complex one for the AI
    # A simple heuristic: if the message is longer than 5 words, it's likely a complex query.
    if len(message_text.split()) > 5:
        # Check user cooldown
        import time
        current_time = time.time()
        if user_id in user_last_request:
            time_since_last = current_time - user_last_request[user_id]
            if time_since_last < USER_COOLDOWN:
                remaining = USER_COOLDOWN - time_since_last
                await update.message.reply_text(
                    f"Please wait {remaining:.0f} more seconds before asking another question."
                )
                return
        
        user_last_request[user_id] = current_time
        await update.message.reply_chat_action('typing')
        ai_response = await get_deepseek_response(update.message.text)
        await update.message.reply_text(ai_response)
    else:
        # For very short, unrecognized messages
        await update.message.reply_text("I'm not sure how to help with that. You can try `/help` to see what I can do.")


# --- Callback Query Handler (for inline buttons) ---

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and executes the corresponding command."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button press

    command = query.data
    if command == "download":
        await query.message.reply_text("Great! I'm fetching the latest version for you now...")
        await download_command(update, context)
    elif command == "plans":
        await query.message.reply_text(PLANS_MESSAGE, parse_mode='Markdown')
    elif command == "help":
        await query.message.reply_text(HELP_MESSAGE, parse_mode='Markdown')


# --- Main Application Setup ---
def main() -> None:
    """Start the bot."""
    # Basic validation to ensure tokens are set
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable not set. Exiting.")
        return
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set. AI features will be disabled.")
    if not WATCHERS_APK_URL:
        logger.warning("WATCHERS_APK_URL is not set. The /download command will not work.")

    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Conversation handler for the /support command
    support_handler = ConversationHandler(
        entry_points=[CommandHandler("support", support_command)],
        states={
            SUPPORT_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_support_query)],
        },
        fallbacks=[CommandHandler("cancel", cancel_support)],
    )

    # Register all handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("plans", plans_command))
    application.add_handler(CommandHandler("download", download_command))
    application.add_handler(support_handler)
    
    # Handler for inline button callbacks
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Handler for all other text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    logger.info("Starting bot polling...")
    application.run_polling()

if __name__ == "__main__":
    main()