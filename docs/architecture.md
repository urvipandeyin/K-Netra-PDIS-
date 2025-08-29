# System Architecture

This document explains the architecture of the **PDIS (Proactive Digital Intervention System)** project.  
It shows how different components interact: data, preprocessing, AI models, alerts, and legal framework.

---

## 1. System Components

1. **Data Sources**  
   - BharatFakeNewsKosh (fake/real news articles)  
   - IFND (Indian Fake News Dataset)  
   - Indo-HateSpeech Dataset (hate/offensive text)  

2. **Preprocessing Pipeline**  
   - Cleaning text (removing stopwords, symbols, URLs)  
   - Normalization (lowercasing, stemming/lemmatization)  
   - Label encoding (0 = real, 1 = fake, 2 = hate speech)  

3. **Feature Extraction**  
   - TF-IDF  
   - Word2Vec embeddings  
   - Transformer embeddings (optional advanced)  

4. **AI Models**  
   - Logistic Regression / Random Forest (baseline)  
   - Deep Learning (LSTM/Transformer)  

5. **Alert & Action System**  
   - Telegram bot for real-time alerts  
   - Risk scoring of content (0 → safe, 1 → high-risk)  
   - Takedown recommendation  

6. **Legal & Compliance Layer**  
   - IT Act 2000, IT Rules 2021  
   - Maintains transparency & audit logs  

---

## 2. Architecture Flow (Mermaid Diagram)

```mermaid
flowchart TD
    A[User Generated Content] --> B[Data Collection]
    B --> C[Preprocessing & Cleaning]
    C --> D[Feature Extraction<br/>(TF-IDF, Word2Vec, Transformer)]
    D --> E[AI Models<br/>(LR, RF, LSTM, Transformer)]
    E --> F[Prediction<br/>(Real, Fake, Hate Speech)]
    F --> G[Risk Scoring Engine]
    G -->|Low Risk| H[Allow Content]
    G -->|Medium Risk| I[Flagged for Review]
    G -->|High Risk| J[Auto Takedown + Alert System]

    I --> K[Compliance Framework<br/>(Dashboard)]
    J --> K
    K --> L[Law Enforcement<br/>(Regulatory Action)]
---

## 3. Transparency & Extensibility

- **Transparency**: Every decision is logged (why content was flagged, model confidence).
- **Extensibility**: New datasets and models can be plugged in easily.
- **Audit Trail**: Digital evidence is maintained for law enforcement.
