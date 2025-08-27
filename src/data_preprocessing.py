"""
Data Preprocessing Pipeline for PDIS

- Reads multiple raw datasets (CSV/XLSX) from data/raw/
- Normalizes them into a common schema: [text, label, source]
- Handles binary relabeling (harmful=1, safe=0) where needed
- Saves final combined dataset to data/processed/combined_dataset.csv
"""

import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # create if not exists


# =========================
# Helper cleaning functions
# =========================
def clean_text(text):
    """Basic text cleaning: lowercase, strip spaces, handle NaN safely."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def preprocess_dataframe(df, text_column=None, label_column=None):
    """Generic preprocessing: dropna, drop duplicates, clean text."""
    df = df.dropna()
    df = df.drop_duplicates()

    if text_column and text_column in df.columns:
        df[text_column] = df[text_column].apply(clean_text)

    if label_column and label_column in df.columns:
        df[label_column] = df[label_column].astype(str).str.strip().str.lower()

    return df

# =========================
# Dataset-specific loaders
# =========================
def process_bharat_fake_news():
    file_path = RAW_DIR / "bharatfakenewskosh_(3)[1].xlsx"
    df = pd.read_excel(file_path)

    df = preprocess_dataframe(df, text_column="text", label_column="label")

    out_path = PROCESSED_DIR / "bharatfakenewskosh_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed BharatFakeNews saved to {out_path}")
    return df


def process_indo_hatespeech():
    file_path = RAW_DIR / "Indo-HateSpeech_Dataset.xlsx"
    df = pd.read_csv(file_path)

    df = preprocess_dataframe(df, text_column="content", label_column="misinfo_flag")

    out_path = PROCESSED_DIR / "indo_hatespeech_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed Facebook Misinformation saved to {out_path}")
    return df

def process_dynamic_hate():
    """Process dynamically generated hate dataset (entries + targets)."""
    entries_file = RAW_DIR / "2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1[1].csv"
    targets_file = RAW_DIR / "2020-12-31-DynamicallyGeneratedHateDataset-targets-v0.1[1].csv"

    df_entries = pd.read_csv(entries_file)
    df_targets = pd.read_csv(targets_file)

    # merge side by side
    df = pd.concat([df_entries, df_targets], axis=1)
    df = preprocess_dataframe(df, text_column="sentence", label_column="label")

    out_path = PROCESSED_DIR / "dynamic_hate_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed Dynamic Hate Dataset saved to {out_path}")
    return df


def process_constraint_hindi():
    """Process Constraint Hindi validation dataset."""
    file_path = RAW_DIR / "Constraint_Hindi_Valid.xlsx"
    df = pd.read_excel(file_path)

    df = preprocess_dataframe(df, text_column="tweet", label_column="label")

    out_path = PROCESSED_DIR / "constraint_hindi_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed Constraint Hindi saved to {out_path}")
    return df


def process_ifnd():
    """Process IFND dataset (csv)."""
    file_path = RAW_DIR / "IFND[1].csv"
    df = pd.read_csv(file_path)

    df = preprocess_dataframe(df, text_column="tweet", label_column="label")

    out_path = PROCESSED_DIR / "ifnd_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed IFND saved to {out_path}")
    return df


def process_test_set():
    """Process Test Set Complete dataset."""
    file_path = RAW_DIR / "Test Set Complete.xlsx"
    df = pd.read_excel(file_path)

    df = preprocess_dataframe(df, text_column="text", label_column="label")

    out_path = PROCESSED_DIR / "test_set_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Processed Test Set saved to {out_path}")
    return df

# =========================
# Main Runner
# =========================
def main():
    """Run preprocessing for all datasets and merge into one combined file."""
    print("Starting preprocessing...")

    datasets = []

    try:
        df = process_bharat_fake_news()
        df = df.rename(columns={"text": "text", "label": "label"})
        df["source"] = "BharatFakeNews"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] BharatFakeNews failed: {e}")

    try:
        df = process_indo_hatespeech()
        df = df.rename(columns={"content": "text", "misinfo_flag": "label"})
        df["source"] = "IndoHateSpeech"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] IndoHateSpeech failed: {e}")

    try:
        df = process_dynamic_hate()
        df = df.rename(columns={"sentence": "text", "label": "label"})
        df["source"] = "DynamicHate"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] DynamicHate failed: {e}")

    try:
        df = process_constraint_hindi()
        df = df.rename(columns={"tweet": "text", "label": "label"})
        df["source"] = "ConstraintHindi"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] ConstraintHindi failed: {e}")

    try:
        df = process_ifnd()
        df = df.rename(columns={"tweet": "text", "label": "label"})
        df["source"] = "IFND"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] IFND failed: {e}")

    try:
        df = process_test_set()
        df = df.rename(columns={"text": "text", "label": "label"})
        df["source"] = "TestSet"
        datasets.append(df[["text", "label", "source"]])
    except Exception as e:
        print(f"[✘] TestSet failed: {e}")

    # =========================
    # Merge all datasets
    # =========================
    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        out_path = PROCESSED_DIR / "combined_dataset.csv"
        combined.to_csv(out_path, index=False)
        print(f"[✔] Combined dataset saved to {out_path}")
        print(f"Final shape: {combined.shape}")
    else:
        print("[!] No datasets were processed successfully!")

    print("All datasets processed!")
if __name__ == "__main__":
    main()