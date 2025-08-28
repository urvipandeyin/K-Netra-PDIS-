#!/usr/bin/env python3
"""
run_public_log.py
- Appends a single flagged post to public_record_site/_data/flagged_posts.csv
"""

import os
import argparse
import pandas as pd
from datetime import datetime
import uuid
from src.predict import sha256_of_string, ensure_dir_for_file

PUBLIC_RECORD_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "public_record_site", "_data", "flagged_posts.csv")
)

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def append_public_log(content: str, threat_score: float, tier: str, model_version: str = "unknown"):
    pid = str(uuid.uuid4())
    sha = sha256_of_string(str(content) + str(threat_score) + pid)
    row = pd.DataFrame([{
        "post_id": pid,
        "content": content,
        "threat_score": threat_score,
        "tier": tier,
        "prediction_timestamp": now_iso(),
        "model_version": model_version,
        "sha256": sha
    }])

    ensure_dir_for_file(PUBLIC_RECORD_CSV)
    if os.path.exists(PUBLIC_RECORD_CSV):
        row.to_csv(PUBLIC_RECORD_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(PUBLIC_RECORD_CSV, mode="w", header=True, index=False)
    print(f"✅ Appended flagged post to {PUBLIC_RECORD_CSV}")

# -------------------------------
# CLI usage
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append flagged post to public log CSV")
    parser.add_argument("--content", required=True, help="Text content of the post")
    parser.add_argument("--threat_score", required=True, type=float, help="Threat score")
    parser.add_argument("--tier", required=True, choices=["AUTO_INTERVENTION", "HUMAN_REVIEW", "LOG_ONLY"], help="Tier")
    parser.add_argument("--model_version", default="unknown", help="Model version")
    args = parser.parse_args()

    append_public_log(args.content, args.threat_score, args.tier, args.model_version)