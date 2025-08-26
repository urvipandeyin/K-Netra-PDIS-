# Data Information for K-Netra Hackathon Project

## Datasets Acquired (as of 26-08-2025)

This document outlines the datasets we have acquired and our plan for combining them into a single, unified file for model training.

---

### **1. BharatFakeNewsKosh**

* **Status:** Downloaded and placed in `data/raw/`.
* **Source:** [https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh](https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh)
* **Notes:** This is a valuable resource with over 26,000 news samples in 9 Indian languages. It is a primary dataset on fraudulent news and will be crucial for training our model.

---

### **2. Indo-HateSpeech**

* **Status:** Downloaded and placed in `data/raw/`.
* **Source:** [https://data.mendeley.com/datasets/snc7mxpj6t](https://data.mendeley.com/datasets/snc7mxpj6t)
* **Notes:** This dataset focuses on hate speech in code-mixed Hindi and English. We will need to relabel its content to fit our binary classification system.

---

### **3. india-hate-speech-superset**

* **Status:** Downloaded and placed in `data/raw/`.
* **Source:** [https://huggingface.co/datasets/manueltonneau/india-hate-speech-superset](https://huggingface.co/datasets/manueltonneau/india-hate-speech-superset)
* **Notes:** An already combined dataset that expands our hate speech training data. It will be very useful in ensuring our model generalizes well to different forms of hateful content.

---

### **4. Learning from the Worst**

* **Status:** Downloaded and placed in `data/raw/`.
* **Source:** [https://www.kaggle.com/datasets/usharengaraju/dynamically-generated-hate-speech-dataset](https://www.kaggle.com/datasets/usharengaraju/dynamically-generated-hate-speech-dataset)
* **Notes:** A synthetically generated hate speech dataset that provides a large volume of data for training. It will help make our model more robust against subtly hateful text.

---

### **5. IFND (Indian Fake News Dataset)**

* **Status:** Downloaded and placed in `data/raw/`.
* **Source:** [https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
* **Notes:** A dataset specifically curated for Indian news, containing both text and images. It provides excellent content for training a multimodal model.

---

### **6. FactDrill**

* **Status:** Email sent to researchers. Awaiting response.

## Data Integration Plan

**Goal:** Create a single `combined_dataset.csv` file in the `data/processed/` folder.

**Steps:**
1.  Use a Python script with **pandas** to load all acquired datasets.
2.  Standardize the column names across all files (e.g., all text columns should be named `text`, all label columns should be named `label`).
3.  Standardize the labels. The combined dataset will use a binary classification system: `1` for fake/hateful content and `0` for legitimate/non-hateful content.
4.  Remove duplicate entries to ensure the model is not trained on redundant data.
5.  Save the final, cleaned file as `data/processed/final_dataset.csv`.ed `label`).
3.  Standardize the labels. The combined dataset will use a binary classification system: `1` for fake/hateful content and `0` for legitimate/non-hateful content.
4.  Remove duplicate entries to ensure the model is not trained on redundant data.
5.  Save the final, cleaned file as `data/processed/final_dataset.csv`.

