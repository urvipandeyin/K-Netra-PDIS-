# 📘 Data Dictionary – Project K-Netra (PDIS)

This file documents the datasets used in Project K-Netra and explains their fields, formats, and purpose.  
The final combined dataset is stored at:  
`data/processed/combined_dataset.csv`

---

## 1. Dataset Sources

- **BharatFakeNewsKosh**  
  Source: [Kaggle](https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh)  
  - Multilingual dataset with ~26,000 news samples in 9 Indian languages.  
  - Labels: Legitimate / Fraudulent.  

- **IFND (Indian Fake News Dataset)**  
  Source: [Kaggle](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)  
  - Multimodal dataset with text + image references.  
  - Labels: Fake / Real.  

- **Indo-HateSpeech**  
  Source: [Mendeley](https://data.mendeley.com/datasets/snc7mxpj6t)  
  - Code-mixed Hindi-English dataset.  
  - Labels: Hate Speech / Non-Hate Speech.  

---

## 2. Common Data Fields

| Field Name       | Type        | Description |
|------------------|------------|-------------|
| `id`             | Integer    | Unique identifier for each record |
| `text`           | String     | The news article, post, or message content |
| `language`       | String     | Language of the content (e.g., Hindi, English, Tamil) |
| `source`         | String     | Dataset source (BharatFakeNewsKosh, IFND, Indo-HateSpeech) |
| `label`          | Categorical | Classification label (malicious / non-malicious, fake / real, hate / normal) |
| `date` (if available) | Date | Date when the content was posted / collected |
| `image_url` (IFND only) | String | Link to associated image (if any) |

---

## 3. Unified Labels in Final Dataset

Since the raw datasets had different labeling schemes, we re-labeled them into a **binary classification**:  

- `1` → Malicious (Fake News, Hate Speech, Propaganda, Misinformation)  
- `0` → Non-Malicious (Legitimate News, Normal Speech, Verified Content)  

---

## 4. Final Combined Dataset

File: `data/processed/combined_dataset.csv`

| Column Name      | Description |
|------------------|-------------|
| `id`             | Auto-generated unique identifier |
| `text`           | Content text (from all datasets) |
| `language`       | Language code (hi, en, ta, etc.) |
| `source`         | Which dataset this record came from |
| `label`          | Final binary label (0 = non-malicious, 1 = malicious) |

---

## 5. Data Flow (Mermaid Diagram)

```mermaid
flowchart TD
    A[BharatFakeNewsKosh] --> D[Preprocessing & Cleaning]
    B[IFND Dataset] --> D
    C[Indo-HateSpeech] --> D
    D --> E[Relabeling (0/1)]
    E --> F[Combined Dataset: combined_dataset.csv]
