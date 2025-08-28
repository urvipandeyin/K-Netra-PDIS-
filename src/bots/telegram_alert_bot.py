import os
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_ID = int(os.getenv("TG_API_ID", 0))
API_HASH = os.getenv("TG_API_HASH", "")
CHAT_ID = os.getenv("TG_ALERT_CHAT_ID", "")  # can be @username or group/channel ID
SESSION_NAME = os.getenv("TG_SESSION_NAME", "alert_bot")

# Create client
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)


async def _send_alert_async(message: str):
    """
    Internal async function to send alert message.
    """
    try:
        await client.start()
        if CHAT_ID:
            await client.send_message(CHAT_ID, message)
        else:
            print("[ERROR] TG_ALERT_CHAT_ID not set in .env")
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {e}")
    finally:
        await client.disconnect()


def send_alert(message: str):
    """
    Sends alert message to Telegram synchronously (safe for use inside predict.py).
    """
    asyncio.run(_send_alert_async(message))


if __name__ == "__main__":
    # Quick test
    send_alert("🚨 Test alert from telegram_alert_bot.py")