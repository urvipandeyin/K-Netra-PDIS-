#!/usr/bin/env python3
"""
predict.py
- Loads a HuggingFace transformer classification model + tokenizer + label encoder
- Produces label + confidence (threat score) for input text (single or CSV batch)
- Maps threat score to tier (AUTO_INTERVENTION / HUMAN_REVIEW / LOG_ONLY)
- Appends high-confidence flagged posts to public_record_site/_data/flagged_posts.csv
- Saves batch predictions to data/processed/predictions.csv by default

Usage:
    python src/predict.py --input_csv data/processed/combined_dataset.csv
    python src/predict.py --text "Some suspicious text here"
"""

import os
import sys
import uuid
import time
import json
import hashlib
import logging
from datetime import datetime

import joblib
import torch
import pandas as pd
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import argparse
from src.bots.telegram_alert_bot import send_alert
import asyncio
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

# -------------------------------
# Defaults / Paths
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "final")
LABEL_ENCODER_PATH = os.path.join(ROOT, "models", "checkpoints", "label_encoder.pkl")
DEFAULT_INPUT_CSV = os.path.join(ROOT, "data", "processed", "combined_dataset.csv")
DEFAULT_OUTPUT_CSV = os.path.join(ROOT, "data", "processed", "predictions.csv")
PUBLIC_RECORD_CSV = os.path.join(ROOT, "public_record_site", "_data", "flagged_posts.csv")
BOT_CONFIG_PATH = os.path.join(ROOT, "configs", "bot_config.yaml")
TRAINING_CONFIG_PATH = os.path.join(ROOT, "configs", "training_config.yaml")

# -------------------------------
# Load environment (optional)
# -------------------------------
load_dotenv()  # safe to call even if .env missing

# -------------------------------
# Utility helpers
# -------------------------------
def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_of_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -------------------------------
# Config loader
# -------------------------------
def load_configs():
    cfg = {}
    # First try bot_config.yaml (for thresholds etc.)
    if os.path.exists(BOT_CONFIG_PATH):
        try:
            with open(BOT_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
            logging.info(f"Loaded config from {BOT_CONFIG_PATH}")
        except Exception as e:
            logging.warning(f"Failed to load {BOT_CONFIG_PATH}: {e}")
    elif os.path.exists(TRAINING_CONFIG_PATH):
        try:
            with open(TRAINING_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
            logging.info(f"Loaded config from {TRAINING_CONFIG_PATH}")
        except Exception as e:
            logging.warning(f"Failed to load {TRAINING_CONFIG_PATH}: {e}")
    else:
        logging.info("No YAML config found, using defaults.")

    # Default thresholds / params if not present
    thresholds = cfg.get("thresholds", {})
    cfg_thresholds = {
        "auto_intervention": float(thresholds.get("auto_intervention", 0.9)),
        "human_review_min": float(thresholds.get("human_review_min", 0.5)),
        # human_review_max is implicitly auto_intervention
    }
    cfg["thresholds"] = cfg_thresholds

    # tokenization/truncation defaults
    cfg["max_length"] = int(cfg.get("max_length", 256))
    cfg["batch_size"] = int(cfg.get("batch_size", 32))
    cfg["model_version"] = cfg.get("model_version", "unknown")

    return cfg

# -------------------------------
# Device selection
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# -------------------------------
# Load tokenizer, model, label encoder
# -------------------------------
def load_model_and_tokenizer(model_dir=MODEL_DIR, label_encoder_path=LABEL_ENCODER_PATH):
    # Tokenizer + Model
    if not os.path.isdir(model_dir):
        logging.error(f"Model directory not found at {model_dir}. Exiting.")
        sys.exit(1)

    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Label encoder (optional)
    le = None
    if os.path.exists(label_encoder_path):
        try:
            le = joblib.load(label_encoder_path)
            logging.info(f"Loaded label encoder from {label_encoder_path}")
        except Exception as e:
            logging.warning(f"Failed to load label encoder: {e}. Predictions will use numeric indices.")
    else:
        logging.warning(f"Label encoder not found at {label_encoder_path}. Using numeric labels.")

    return tokenizer, model, le

# -------------------------------
# Prediction helpers
# -------------------------------
def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)

def map_score_to_tier(score: float, thresholds: dict) -> str:
    """
    thresholds dict expected to contain:
      - auto_intervention (float, e.g. 0.9)
      - human_review_min (float, e.g. 0.5)
    Mapping:
      score >= auto_intervention -> AUTO_INTERVENTION
      human_review_min <= score < auto_intervention -> HUMAN_REVIEW
      else -> LOG_ONLY
    """
    ai = thresholds["auto_intervention"]
    hr_min = thresholds["human_review_min"]
    if score >= ai:
        return "AUTO_INTERVENTION"
    if score >= hr_min:
        return "HUMAN_REVIEW"
    return "LOG_ONLY"

def predict_batch_texts(tokenizer, model, texts, max_length=256, batch_size=32):
    """
    Predict labels and confidences for a list of texts.
    Returns two lists: pred_indices (int), confidences (float)
    """
    pred_indices = []
    confidences = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            # move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (batch_size, num_labels)
                probs = softmax_probs(logits)
                top_probs, top_idx = torch.max(probs, dim=-1)
                # transfer to cpu and to python types
                for idx_val, prob_val in zip(top_idx.cpu().tolist(), top_probs.cpu().tolist()):
                    pred_indices.append(int(idx_val))
                    confidences.append(float(prob_val))
        except Exception as e:
            logging.warning(f"Batch prediction failed for batch starting at {i}: {e}")
            # fallback per-text
            for t in batch_texts:
                try:
                    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = softmax_probs(logits)
                        top_prob, top_idx = torch.max(probs, dim=-1)
                        pred_indices.append(int(top_idx.cpu().item()))
                        confidences.append(float(top_prob.cpu().item()))
                except Exception as e2:
                    logging.warning(f"Single-item fallback failed: {e2}")
                    pred_indices.append(-1)
                    confidences.append(0.0)

    return pred_indices, confidences

# -------------------------------
# High-level predict functions
# -------------------------------
def predict_text_single(tokenizer, model, le, text, cfg):
    """
    Predict a single text and return dict with metadata.
    """
    if text is None or (isinstance(text, str) and text.strip() == ""):
        return {
            "post_id": str(uuid.uuid4()),
            "content": "" if text is None else text,
            "predicted_index": -1,
            "predicted_label": "unknown",
            "threat_score": 0.0,
            "tier": map_score_to_tier(0.0, cfg["thresholds"]),
            "timestamp": now_iso(),
            "model_version": cfg.get("model_version", "unknown"),
            "sha256": sha256_of_string(str(text) + now_iso()),
        }

    pred_idx_list, conf_list = predict_batch_texts(
        tokenizer, model, [text], max_length=cfg["max_length"], batch_size=1
    )
    pred_idx = pred_idx_list[0] if pred_idx_list else -1
    conf = conf_list[0] if conf_list else 0.0

    if le is not None and pred_idx >= 0:
        try:
            label = str(le.inverse_transform([pred_idx])[0])
        except Exception:
            label = str(pred_idx)
    else:
        label = str(pred_idx)

    tier = map_score_to_tier(conf, cfg["thresholds"])
    rec = {
        "post_id": str(uuid.uuid4()),
        "content": text,
        "predicted_index": pred_idx,
        "predicted_label": label,
        "threat_score": conf,
        "tier": tier,
        "timestamp": now_iso(),
        "model_version": cfg.get("model_version", "unknown"),
        "sha256": sha256_of_string(str(text) + now_iso()),
    }
    return rec

def predict_csv_file(tokenizer, model, le, input_csv, text_column="text", cfg=None, output_csv=DEFAULT_OUTPUT_CSV, append_public_log=True):
    """
    Predict for all rows in input_csv (expects a column 'text' by default).
    Returns dataframe with appended columns.
    Optionally appends flagged posts (tier != LOG_ONLY) to public_record_site/_data/flagged_posts.csv.
    """
    if cfg is None:
        cfg = load_configs()

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    logging.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not in input CSV columns: {df.columns.tolist()}")

    texts = df[text_column].astype(str).tolist()
    logging.info(f"Running predictions on {len(texts)} rows (batch_size={cfg['batch_size']})...")

    pred_indices, confidences = predict_batch_texts(tokenizer, model, texts, max_length=cfg["max_length"], batch_size=cfg["batch_size"])

    # map predicted labels (if label encoder provided)
    if le is not None:
        try:
            labels = le.inverse_transform([int(i) if i >= 0 else -1 for i in pred_indices])
            labels = [str(l) for l in labels]
        except Exception:
            labels = [str(i) for i in pred_indices]
    else:
        labels = [str(i) for i in pred_indices]

    # Append prediction results to dataframe
    df["predicted_index"] = pred_indices
    df["predicted_label"] = labels
    df["threat_score"] = confidences
    df["tier"] = [map_score_to_tier(s, cfg["thresholds"]) for s in confidences]
    df["prediction_timestamp"] = now_iso()
    df["model_version"] = cfg.get("model_version", "unknown")
    # generate a post_id and sha256 per row
    post_ids = []
    sha256s = []
    for i, text in enumerate(texts):
        pid = str(uuid.uuid4())
        post_ids.append(pid)
        sha256s.append(sha256_of_string(str(text) + str(confidences[i]) + str(pid)))
    df["post_id"] = post_ids
    df["sha256"] = sha256s

    # Save predictions CSV
    ensure_dir_for_file(output_csv)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved batch predictions to {output_csv}")

    # Optionally append flagged posts to public record
    if append_public_log:
        flagged_df = df[df["tier"] != "LOG_ONLY"].copy()
        if not flagged_df.empty:
            ensure_dir_for_file(PUBLIC_RECORD_CSV)
            # select columns for public record
            public_cols = ["post_id", "content" if "content" in df.columns else text_column, "threat_score", "tier", "prediction_timestamp", "model_version", "sha256"]
            # make sure content column name exists
            if "content" not in flagged_df.columns:
                flagged_df = flagged_df.rename(columns={text_column: "content"})
            public_to_append = flagged_df[["post_id", "content", "threat_score", "tier", "prediction_timestamp", "model_version", "sha256"]]
            # if file exists, append; else create with header
            if os.path.exists(PUBLIC_RECORD_CSV):
                public_to_append.to_csv(PUBLIC_RECORD_CSV, mode="a", header=False, index=False)
            else:
                public_to_append.to_csv(PUBLIC_RECORD_CSV, mode="w", header=True, index=False)
            logging.info(f"Appended {len(public_to_append)} flagged rows to {PUBLIC_RECORD_CSV}")
        else:
            logging.info("No flagged rows to append to public record.")
    return df

# -------------------------------
# Main CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict labels and threat scores using the final model.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="Single text to predict")
    group.add_argument("--input_csv", type=str, help=f"CSV path to run batch predictions (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--text_col", type=str, default="text", help="Name of text column in input CSV")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Where to save batch predictions")
    parser.add_argument("--no_public_log", action="store_true", help="Do not append flagged posts to public record CSV")
    parser.add_argument("--max_length", type=int, default=None, help="Override tokenizer max_length for truncation")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    cfg = load_configs()
    if args.max_length:
        cfg["max_length"] = args.max_length
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    tokenizer, model, le = load_model_and_tokenizer()

    # ---------------- Single text prediction ----------------
    if args.text:
        rec = predict_text_single(tokenizer, model, le, args.text, cfg)
        logging.info("Single prediction result:")
        logging.info(json.dumps(rec, indent=2))

        print(f"Predicted label: {rec['predicted_label']}")
        print(f"Threat score: {rec['threat_score']:.4f}")
        print(f"Tier: {rec['tier']}")

        # Send alert if not LOG_ONLY
        if rec["tier"] != "LOG_ONLY":
            msg = (
                f"⚠️ ALERT DETECTED\n"
                f"Label: {rec['predicted_label']}\n"
                f"Score: {rec['threat_score']:.2f}\n"
                f"Tier: {rec['tier']}\n"
                f"Content: {rec['content']}"
            )
            # --- Telegram ---
            try:
                from src.bots.telegram_alert_bot import send_alert
                chat_id = os.getenv("TG_ALERT_CHAT_ID")
                if chat_id:
                    send_alert(msg)
                    logging.info("Telegram alert sent successfully")
                else:
                    logging.warning("TG_ALERT_CHAT_ID not set in .env — skipping Telegram alert")
            except Exception as e:
                logging.error(f"Failed to send Telegram alert: {e}")

            # --- Facebook ---
            fb_page_id = os.getenv("FB_PAGE_ID")
            if fb_page_id:
                try:
                    post_to_facebook(fb_page_id, msg)
                    logging.info(f"Facebook alert sent to {fb_page_id}")
                except Exception as e:
                    logging.error(f"Failed to send Facebook alert: {e}")
            else:
                logging.warning("FB_PAGE_ID not set in .env — skipping Facebook alert")

            # --- Instagram ---
            ig_user_id = os.getenv("IG_USER_ID")
            if ig_user_id:
                try:
                    post_to_instagram(ig_user_id, msg)
                    logging.info(f"Instagram alert posted to {ig_user_id}")
                except Exception as e:
                    logging.error(f"Failed to send Instagram alert: {e}")
            else:
                logging.warning("IG_USER_ID not set in .env — skipping Instagram alert")

            # --- X ---
            try:
                from src.bots.x_bot import post_to_x
                post_to_x(msg)
                logging.info("X alert posted successfully")
            except Exception as e:
                logging.error(f"Failed to send X alert: {e}")

    # ---------------- Batch CSV prediction ----------------
    else:
        input_csv = args.input_csv if args.input_csv else DEFAULT_INPUT_CSV
        if not os.path.exists(input_csv):
            logging.warning(f"Input CSV not found at {input_csv}. Exiting.")
            return

        df_out = predict_csv_file(
            tokenizer,
            model,
            le,
            input_csv,
            text_column=args.text_col,
            cfg=cfg,
            output_csv=args.output_csv,
            append_public_log=not args.no_public_log,
        )

        total = len(df_out)
        counts = df_out["tier"].value_counts().to_dict()
        logging.info(f"Processed {total} rows. Tier counts: {counts}")

if __name__ == "__main__":
    main()
