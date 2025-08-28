#!/usr/bin/env python3
"""
run_detection.py
- Runs prediction on single text or CSV batch
- Sends alerts (Telegram, Facebook, Instagram, X) if tier != LOG_ONLY
Usage:
    python scripts/run_detection.py --text "Some suspicious text here"
    python scripts/run_detection.py --input_csv data/processed/combined_dataset.csv
"""

import os
import argparse
import asyncio
import json
import logging
from src.predict import load_model_and_tokenizer, predict_text_single, predict_csv_file, load_configs
from src.bots.telegram_alert_bot import send_alert
from src.bots.facebook_bot import post_to_facebook
from src.bots.instagram_bot import post_to_instagram
from src.bots.x_bot import post_to_x

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(description="Run threat detection and send alerts if needed.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to predict")
    group.add_argument("--input_csv", type=str, help="CSV path to run batch predictions")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name in CSV")
    parser.add_argument("--output_csv", type=str, default=None, help="Where to save batch predictions (optional)")
    args = parser.parse_args()

    # Load configs, model, tokenizer, label encoder
    cfg = load_configs()
    tokenizer, model, le = load_model_and_tokenizer()

    # -------------------------------
    # Single text prediction
    # -------------------------------
    if args.text:
        rec = predict_text_single(tokenizer, model, le, args.text, cfg)
        logging.info("Prediction result:")
        logging.info(json.dumps(rec, indent=2))
        print(f"Predicted label: {rec['predicted_label']}")
        print(f"Threat score: {rec['threat_score']:.4f}")
        print(f"Tier: {rec['tier']}")

        # Send alerts if tier is not LOG_ONLY
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
            else:
                logging.warning("TG_ALERT_CHAT_ID not set — skipping Telegram alert")
            # Facebook
            fb_page_id = os.getenv("FB_PAGE_ID")
            if fb_page_id:
                try:
                    post_to_facebook(fb_page_id, msg)
                    logging.info(f"Facebook alert sent to {fb_page_id}")
                except Exception as e:
                    logging.error(f"Failed to send Facebook alert: {e}")
            else:
                logging.warning("FB_PAGE_ID not set — skipping Facebook alert")
            # Instagram
            ig_user_id = os.getenv("IG_USER_ID")
            if ig_user_id:
                try:
                    post_to_instagram(ig_user_id, msg, image_url=None)  # optional image
                    logging.info(f"Instagram alert posted to {ig_user_id}")
                except Exception as e:
                    logging.error(f"Failed to send Instagram alert: {e}")
            else:
                logging.warning("IG_USER_ID not set — skipping Instagram alert")
            # X
            x_token = os.getenv("X_BEARER_TOKEN")
            if x_token:
                try:
                    post_to_x(msg)
                    logging.info("X alert tweeted successfully")
                except Exception as e:
                    logging.error(f"Failed to send X alert: {e}")
            else:
                logging.warning("X_BEARER_TOKEN not set — skipping X alert")

    # -------------------------------
    # CSV batch prediction
    # -------------------------------
    elif args.input_csv:
        output_csv = args.output_csv if args.output_csv else None
        df_out = predict_csv_file(
            tokenizer,
            model,
            le,
            input_csv=args.input_csv,
            text_column=args.text_col,
            cfg=cfg,
            output_csv=output_csv,
        )
        logging.info(f"Processed {len(df_out)} rows. Tier counts: {df_out['tier'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()