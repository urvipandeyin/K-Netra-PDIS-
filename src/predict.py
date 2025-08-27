import os
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# -------------------------------
# Paths
# -------------------------------
MODEL_DIR = "models/final/"
LABEL_ENCODER_PATH = "models/checkpoints/label_encoder.pkl"

# -------------------------------
# Load Model, Tokenizer, and Label Encoder
# -------------------------------
print("[INFO] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # set model to evaluation mode

print("[INFO] Loading label encoder...")
le = joblib.load(LABEL_ENCODER_PATH)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_text(text):
    """
    Predict the label and confidence for a single text input.
    Handles empty strings and unexpected errors with fallback.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "unknown", 0.0  # fallback for empty text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_idx].item()
        label = le.inverse_transform([pred_idx])[0]
        return label, confidence
    except Exception as e:
        print(f"[WARNING] Prediction failed for text: {text}. Error: {e}")
        return "unknown", 0.0

# -------------------------------
# Batch Prediction Function
# -------------------------------
def predict_csv(csv_path, text_column="text"):
    """
    Predict labels for a CSV file containing a text column.
    Returns a DataFrame with predicted labels and confidence scores.
    """
    df = pd.read_csv(csv_path)
    labels, confidences = [], []

    for text in df[text_column].astype(str).tolist():
        label, conf = predict_text(text)
        labels.append(label)
        confidences.append(conf)

    df["predicted_label"] = labels
    df["confidence"] = confidences
    return df

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Example single prediction
    sample_text = "This is a harmful message"
    label, confidence = predict_text(sample_text)
    print(f"Text: {sample_text}\nPredicted: {label}, Confidence: {confidence:.2f}")

    # Example batch prediction
    csv_path = "data/processed/combined_dataset.csv" 
    if os.path.exists(csv_path):
        print(f"[INFO] Running batch prediction on {csv_path}...")
        result_df = predict_csv(csv_path)
        output_csv = "data/processed/predictions.csv"
        result_df.to_csv(output_csv, index=False)
        print(f"[INFO] Batch predictions saved to {output_csv}")
    else:
        print(f"[WARNING] CSV file not found at {csv_path}, skipping batch prediction.")
