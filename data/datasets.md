# Data Information for K-Netra Hackathon Project

## Datasets Acquired (as of 26-08-2025)

This document outlines the datasets we have acquired and our plan for combining them into a single, unified file for model training.

---

### **1. BharatFakeNewsKosh**

* **Status:** Downloaded and placed in `data/raw/`.
* **Notes:** This is a valuable resource with over 26,000 news samples in 9 Indian languages. It is a primary dataset on fraudulent news and will be crucial for training our model.

---

### **2. Indo-HateSpeech**

* **Status:** Downloaded and placed in `data/raw/`.
* **Notes:** This dataset focuses on hate speech in code-mixed Hindi and English. We will need to relabel its content to fit our binary classification system (e.g., hate speech will be labeled as a type of fake/harmful content).

---

### **3. india-hate-speech-superset**

* **Status:** Downloaded and placed in `data/raw/`.
* **Notes:** An already combined dataset that expands our hate speech training data. It will be very useful in ensuring our model generalizes well to different forms of hateful content.

---

### **4. Learning from the Worst**

* **Status:** Downloaded and placed in `data/raw/`.
* **Notes:** A synthetically generated hate speech dataset that provides a large volume of data for training. It will help make our model more robust against subtly hateful text.

---

### **5. FactDrill**

* **Status:** Email sent to researchers. Awaiting response.

## Data Integration Plan

**Goal:** Create a single `combined_dataset.csv` file in the `data/processed/` folder.

**Steps:**
1.  Use a Python script with **pandas** to load all acquired datasets.
2.  Standardize the column names across all files (e.g., all text columns should be named `text`, all label columns should be named `label`).
3.  Standardize the labels. The combined dataset will use a binary classification system: `1` for fake/hateful content and `0` for legitimate/non-hateful content.
4.  Remove duplicate entries to ensure the model is not trained on redundant data.
5.  Save the final, cleaned file as `data/processed/final_dataset.csv`.

This is a professional and effective way to document your work, especially in a fast-paced hackathon.
