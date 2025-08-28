#!/usr/bin/env python3
"""
run_telegram_alert.py
- Script to send an alert message to Telegram using telegram_alert_bot.py
Usage:
    python run_telegram_alert.py
    python run_telegram_alert.py --message "Custom alert message"
"""

import argparse
from src.bots.telegram_alert_bot import send_alert

def main():
    parser = argparse.ArgumentParser(description="Send an alert message to Telegram.")
    parser.add_argument(
        "--message",
        type=str,
        default="🚨 Test Telegram alert from my bot!",
        help="Message text to send via Telegram"
    )
    args = parser.parse_args()

    if not args.message:
        print("❌ Message text is required for Telegram alert.")
        return

    try:
        send_alert(args.message)
        print("✅ Telegram alert sent successfully.")
    except Exception as e:
        print(f"🔥 Failed to send Telegram alert: {e}")

if __name__ == "__main__":
    main()