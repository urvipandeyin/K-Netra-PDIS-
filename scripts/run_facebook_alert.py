#!/usr/bin/env python3
"""
run_facebook_alert.py
- Script to send a test or custom alert message to a Facebook Page using facebook_bot.py
Usage:
    python run_facebook_alert.py
    python run_facebook_alert.py --message "Custom alert message" --image_url "https://example.com/image.png"
"""

import argparse
from src.bots.facebook_bot import post_to_facebook

def main():
    parser = argparse.ArgumentParser(description="Send an alert message to Facebook Page.")
    parser.add_argument("--message", type=str, default="🚨 Test alert from my bot!", help="Message to post")
    parser.add_argument("--image_url", type=str, default=None, help="Optional image URL to post")
    args = parser.parse_args()

    try:
        post_to_facebook(args.message, args.image_url)
    except Exception as e:
        print(f"🔥 Failed to send Facebook alert: {e}")

if __name__ == "__main__":
    main()