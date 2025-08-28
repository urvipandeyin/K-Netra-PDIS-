#!/usr/bin/env python3
"""
run_all.py
- Runs detection (single text or CSV batch)
- Sends alerts to Telegram, Facebook, Instagram, X if tier != LOG_ONLY
- Updates public log CSV
"""

import os
import json
import asyncio
import argparse
from src.predict import load_configs, load_model_and_tokenizer, predict_text_single, predict_csv_file, now_iso
from src.bots.telegram_alert_bot import send_alert
from src.bots.facebook_bot import post_to_facebook
from src.bots.instagram_bot import post_to_instagram
from src.bots.x_bot import post_to_x
from scripts.run_public_log import append_public_log

def process_single_text(text):
    cfg = load_configs()
    tokenizer, model, le = load_model_and_tokenizer()
    rec = predict_text_single(tokenizer, model, le, text, cfg)

    print(f"Predicted label: {rec['predicted_label']}")
    print(f"Threat score: {rec['threat_score']:.4f}")
    print(f"Tier: {rec['tier']}")

    if rec["tier"] != "LOG_ONLY":
        msg = (
            f"⚠️ ALERT DETECTED\n"
            f"Label: {rec['predicted_label']}\n"
            f"Score: {rec['threat_score']:.2f}\n"
            f"Tier: {rec['tier']}\n"
            f"Content: {rec['content']}"
        )

        # Telegram
        chat_id = os.getenv("TG_ALERT_CHAT_ID")
        if chat_id:
            asyncio.run(send_alert(msg))
        # Facebook
        fb_page_id = os.getenv("FB_PAGE_ID")
        if fb_page_id:
            try:
                post_to_facebook(msg)
            except Exception as e:
                print(f"[ERROR] Facebook alert failed: {e}")
        # Instagram
        ig_user_id = os.getenv("IG_USER_ID")
        if ig_user_id:
            try:
                post_to_instagram(msg, image_url="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png")
            except Exception as e:
                print(f"[ERROR] Instagram alert failed: {e}")
        # X
        x_token = os.getenv("X_BEARER_TOKEN")
        if x_token:
            try:
                post_to_x(msg)
            except Exception as e:
                print(f"[ERROR] X alert failed: {e}")

        # Public log
        append_public_log(
            content=rec["content"],
            threat_score=rec["threat_score"],
            tier=rec["tier"],
            model_version=rec["model_version"]
        )

def process_csv(input_csv, output_csv=None, text_col="text"):
    cfg = load_configs()
    tokenizer, model, le = load_model_and_tokenizer()
    df_out = predict_csv_file(
        tokenizer,
        model,
        le,
        input_csv,
        text_column=text_col,
        cfg=cfg,
        output_csv=output_csv if output_csv else None,
        append_public_log=True
    )
    print(f"Processed {len(df_out)} rows. Tier counts:")
    print(df_out["tier"].value_counts().to_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full detection + alert + public log pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to predict")
    group.add_argument("--input_csv", type=str, help="CSV path to run batch predictions")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name in CSV")
    parser.add_argument("--output_csv", type=str, default=None, help="Where to save CSV predictions")
    args = parser.parse_args()

    if args.text:
        process_single_text(args.text)
    elif args.input_csv:
        process_csv(args.input_csv, output_csv=args.output_csv, text_col=args.text_col)
        