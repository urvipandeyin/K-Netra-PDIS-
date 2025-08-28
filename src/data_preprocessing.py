"""
Data Preprocessing Pipeline for PDIS

- Reads multiple raw datasets (CSV/XLSX) from data/raw/
- Normalizes them into a common schema: [text, label, source]
- Converts labels to binary: harmful=1, safe=0
- Saves final combined dataset to data/processed/combined_dataset.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helper functions
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def map_label(label):
    """
    Convert labels to binary:
    'false', 'no' → 0 (safe)
    'true', 'yes', 'harmful' → 1 (harmful)
    """
    if isinstance(label, str):
        label = label.strip().lower()
        if label in ["true", "yes", "harmful", "1"]:
            return 1
        else:
            return 0
    elif isinstance(label, (int, float)):
        return int(label)
    else:
        return 0

# =========================
# Dataset loaders
# =========================
def process_bharatfakenewskosh():
    file_path = RAW_DIR / "bharatfakenewskosh_(3)[1].xlsx"
    df = pd.read_excel(file_path)
    
    df = df.rename(columns={"Text": "text", "Label": "label"})
    df = df[["text", "label"]]
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].apply(map_label)
    df["source"] = "BharatFakeNewsKosh"

    out_path = PROCESSED_DIR / "bharatfakenewskosh_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] BharatFakeNewsKosh processed. Rows: {len(df)}")
    return df

def process_indo_hatespeech():
    file_path = RAW_DIR / "Indo-HateSpeech_Dataset.xlsx"
    df = pd.read_excel(file_path)
    
    # Detect text column (e.g., Column1 or Column2)
    text_col_candidates = ["Column1", "Column2", "Post", "Content"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    if text_col is None:
        print("[⚠] No text column found in Indo-HateSpeech_Dataset.xlsx, skipping...")
        return pd.DataFrame(columns=["text","label","source"])

    label_col_candidates = ["Label", "misinfo_flag"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        label_col = df.columns[-1]  # fallback to last column

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]]
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].apply(map_label)
    df["source"] = "IndoHateSpeech"

    out_path = PROCESSED_DIR / "indo_hatespeech_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] Indo-HateSpeech processed. Rows: {len(df)}")
    return df

def process_ifnd():
    file_path = RAW_DIR / "IFND[1].csv"
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except:
        df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")
    
    text_col_candidates = ["Statement", "Text", "Post"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    label_col_candidates = ["Label", "label"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        print("[⚠] IFND dataset missing required columns, skipping...")
        return pd.DataFrame(columns=["text","label","source"])

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]]
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].apply(map_label)
    df["source"] = "IFND"

    out_path = PROCESSED_DIR / "ifnd_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[✔] IFND processed. Rows: {len(df)}")
    return df

# =========================
# Main runner
# =========================
def main():
    datasets = []

    try:
        datasets.append(process_bharatfakenewskosh())
    except Exception as e:
        print(f"[✘] BharatFakeNewsKosh failed: {e}")

    try:
        datasets.append(process_indo_hatespeech())
    except Exception as e:
        print(f"[✘] IndoHateSpeech failed: {e}")

    try:
        datasets.append(process_ifnd())
    except Exception as e:
        print(f"[✘] IFND failed: {e}")

    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        combined = combined[combined["text"].str.strip() != ""]  # remove empty text
        out_path = PROCESSED_DIR / "combined_dataset.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n✅ Combined dataset saved at {out_path}")
        print(f"Total rows: {len(combined)} | Columns: {combined.columns.tolist()}")
    else:
        print("[!] No datasets processed successfully!")

if __name__ == "__main__":
    main()
