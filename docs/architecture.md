# System Architecture – Project K-Netra (PDIS)

## Architecture Flow (Mermaid Diagram)
```mermaid
flowchart TD
    A[Input Content] --> B[AI Detection Engine]
    B --> C{Threat Confidence Score}
    
    C -->|>0.9 High Confidence| D[Automated Public Intervention]
    C -->|0.5 - 0.9 Moderate| E[Human Analyst Review]
    C -->|<0.5 Low| F[Log for Trend Monitoring]
    
    D --> G[Automated Takedown Request under IT Act 2000 + IT Rules 2021]
    D --> H[Public Counter-Narrative + Official Record]
    
    E --> I[Analyst Validates -> Send to D or F]
    
    G --> J[Social Media Platforms]
    H --> K[Public Repository + Neutral Bot Comment]
    
    subgraph Encrypted Platforms (Telegram/WhatsApp)
        L[Detected Malicious Content]
        L --> M[Digital Evidence Package Generation]
        M --> N[Police Cyber Cell Alert via Secure Bot]
    end
System Components
-----------------
1. AI Detection Engine  
   - NLP + ML model (datasets: BharatFakeNewsKosh, IFND, Indo-HateSpeech)  
   - Generates Threat Confidence Score (0–1)  
   - Risk levels: High >0.9 | Moderate 0.5–0.9 | Low <0.5  

2. Public Intervention (Phase 2)  
   - Automated Takedown Requests (IT Act 2000, IT Rules 2021)  
   - Official Public Record (immutable log on GitHub Pages)  
   - Neutral Comment Bot (context + verified link)  

3. Encrypted Platform Strategy (Phase 3)  
   - Human-in-the-loop for Telegram/WhatsApp  
   - Digital Evidence Package (DEP) under BSA 2023  
   - Secure alert → police cyber cells via encrypted bot  

Transparency Principles
-----------------------
- Public repository ensures accountability  
- Evidence admissible under BSA 2023  
- No hidden takedowns, all actions logged  

File Extensions
---------------
.py   → Python scripts & modules  
.ipynb → Notebooks for EDA/experiments  
.yaml → Config files (hyperparameters, paths)  
.csv / .xlsx → Raw & processed datasets  
.pkl → Saved ML artifacts (vectorizer, encoder, model)  
.pt → PyTorch transformer weights  
.md → Documentation (README, dictionary, architecture)  
