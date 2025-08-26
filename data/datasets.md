# Data Information for K-Netra Hackathon Project

## Datasets Acquired (as of 26-08-2025)

This document outlines the datasets we have acquired and our plan for combining them into a single, unified file for model training.

1.  **BharatFakeNewsKosh:**
    * **Status:** Downloaded and placed in `data/raw/`.
    * **Key Columns:** [List the main columns like `text`, `label`, etc.]
    * **Notes:** This is our primary dataset on fraudulent news in multiple Indian languages.

2.  **Indo-HateSpeech:**
    * **Status:** Downloaded and placed in `data/raw/`.
    * **Key Columns:** [List the main columns like `text`, `label`, `class` etc.]
    * **Notes:** Focuses on hate speech, which we will need to re-label to fit our binary classification (e.g., hate speech = fake news).

3.  **india-hate-speech-superset:**
    * **Status:** Downloaded and placed in `data/raw/`.
    * **Key Columns:** [List the main columns]
    * **Notes:** An already combined dataset. We will use this to expand our hate speech training data.

4.  **Learning from the Worst:**
    * **Status:** Downloaded and placed in `data/raw/`.
    * **Key Columns:** [List the main columns]
    * **Notes:** Synthetically generated hate speech data.

5.  **FactDrill:**
    * **Status:** Email sent to researchers. Awaiting response.

## Data Integration Plan

**Goal:** Create a single `combined_dataset.csv` file in the `data/processed/` folder.

**Steps:**
1.  Use a Python script with **pandas** to load all acquired datasets.
2.  Standardize the column names across all files (e.g., all text columns should be named `text`, all label columns should be named `label`).
3.  Standardize the labels. The combined dataset will use a binary classification system: `1` for fake/hateful content and `0` for legitimate/non-hateful content.
4.  Remove duplicate entries to ensure the model is not trained on redundant data.
5.  Save the final, cleaned file as `data/processed/final_dataset.csv`.

---

This is a professional and effective way to document your work, especially in a fast-paced hackathon.
