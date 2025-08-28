#!/usr/bin/env python3
"""
run_x_alert.py
- Script to post a message to X (Twitter) using OAuth 1.0a User Context.
"""

import argparse
from src.bots.x_bot import post_to_x  # make sure this matches your module path

def main():
    parser = argparse.ArgumentParser(description="Send an alert to X (Twitter).")
    parser.add_argument("--message", type=str, required=True, help="Message text to post on X")
    args = parser.parse_args()

    try:
        post_to_x(args.message)
    except Exception as e:
        print(f"[ERROR] Failed to send X alert: {e}")

if __name__ == "__main__":
    main()