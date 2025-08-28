from telethon import TelegramClient
import os
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("TG_API_ID"))
api_hash = os.getenv("TG_API_HASH")
session_name = os.getenv("TG_SESSION", "alert_bot")

client = TelegramClient(session_name, api_id, api_hash)

async def main():
    print("Logging in...")
    await client.start()
    print("✅ Session generated and saved!")

with client:
    client.loop.run_until_complete(main())
