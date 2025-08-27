import os
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -------------------------------
# Load Config
# -------------------------------
config_path = "configs/training_config.yaml"
print(f"[INFO] Loading config from {os.path.abspath(config_path)}")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if config is None:
    raise ValueError(f"Config file {config_path} is empty or invalid YAML")

print(f"[INFO] Loaded config: {config}")

SEED = config.get("seed", 42)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/processed/combined_dataset.csv")

# Encode string labels into integers
if df["label"].dtype == "object":
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    print("[INFO] Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    os.makedirs("models/checkpoints/", exist_ok=True)
    joblib.dump(le, "models/checkpoints/label_encoder.pkl")

X = df["text"].astype(str).tolist()
y = df["label"].astype(int).tolist()

# -------------------------------
# Train/Validation Split
# -------------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    X, y,
    test_size=1 - config["train_val_split"],
    random_state=SEED,
    stratify=y
)

# -------------------------------
# Baseline Models
# -------------------------------
print("\n[INFO] Training baseline models...")
os.makedirs("models/checkpoints/", exist_ok=True)

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_val_tfidf = vectorizer.transform(val_texts)

# Logistic Regression
log_reg = LogisticRegression(max_iter=200, random_state=SEED)
log_reg.fit(X_train_tfidf, train_labels)
y_pred_lr = log_reg.predict(X_val_tfidf)

print("\n[Baseline Logistic Regression Results]")
print(classification_report(val_labels, y_pred_lr))
joblib.dump((log_reg, vectorizer), "models/checkpoints/baseline_logreg.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf.fit(X_train_tfidf, train_labels)
y_pred_rf = rf.predict(X_val_tfidf)

print("\n[Baseline Random Forest Results]")
print(classification_report(val_labels, y_pred_rf))
joblib.dump((rf, vectorizer), "models/checkpoints/baseline_rf.pkl")

# -------------------------------
# Transformer Model (XLM-R, BERT, etc.)
# -------------------------------
print("\n[INFO] Training transformer model...")

model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=config["max_length"]
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=config["max_length"]
)

train_dataset = Dataset.from_dict({**train_encodings, "labels": train_labels})
val_dataset = Dataset.from_dict({**val_encodings, "labels": val_labels})

num_labels = len(set(y))
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="models/checkpoints/",
    save_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_epochs"],
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    no_cuda=True,   
    fp16=False      # Disable mixed precision on CPU
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------------
# Save Final Model
# -------------------------------
print("\n[INFO] Saving final transformer model...")
os.makedirs("models/final/", exist_ok=True)
trainer.save_model("models/final/")
tokenizer.save_pretrained("models/final/")

print("\n Training complete. Models saved in 'models/checkpoints/' and 'models/final/'")

