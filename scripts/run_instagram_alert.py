#!/usr/bin/env python3
"""
run_instagram_alert.py
- Script to send an alert message (with image) to Instagram using instagram_bot.py
Usage:
    python run_instagram_alert.py
    python run_instagram_alert.py --message "Custom alert message" --image_url "https://example.com/image.png"
"""

import argparse
from src.bots.instagram_bot import post_to_instagram

def main():
    parser = argparse.ArgumentParser(description="Send an alert message to Instagram.")
    parser.add_argument(
        "--message",
        type=str,
        default="🚨 Test Instagram alert from my bot!",
        help="Message/caption to post"
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
        help="URL of the image to post (required)"
    )
    args = parser.parse_args()

    if not args.image_url:
        print("❌ Image URL is required for Instagram post.")
        return

    try:
        post_to_instagram(args.message, args.image_url)
    except Exception as e:
        print(f"🔥 Failed to send Instagram alert: {e}")

if __name__ == "__main__":
    main()