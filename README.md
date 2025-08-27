# K-Netra-PDIS
# Proactive Digital Integrity System (PDIS)
**An AI-powered cyber shield to detect, score, and act on misinformation and harmful content.**
## 1) Why this project is needed (Problem)

Misinformation and harmful digital content spread faster than the truth.  
They can create **panic, violence, financial scams, and distrust in society**.  

Examples:  
- Fake post: *"XYZ Bank is shutting down, withdraw all your money now!"* → causes chaos.  
- Rumor: *"Drink bleach to cure COVID-19"* → health danger.  
- Hate messages → lead to riots and unrest.  

Current systems are **reactive** (take action after harm is done) and often lack **transparency**.
👉 We need a **proactive, AI-powered system** that:  
- Detects threats early,  
- Decides risk automatically,  
- Takes legal + transparent action,  
- Provides digital evidence for law enforcement.  
## 2) Our Solution (Overview)

PDIS is a **cyber shield** that detects misinformation early, scores its risk, and triggers quick action.

1. **AI Detection Engine** – Monitors social media, news, and chat apps in multiple languages.  
2. **Threat Confidence Score (0–1)** – Transparent scoring system:  
   - 0.0–0.3 → Log only  
   - 0.3–0.7 → Human review  
   - 0.7–1.0 → Auto takedown + evidence  
3. **Intervention Protocol** – Sends legal takedown requests, generates digital evidence, and updates a public transparency log.  

👉 Unlike traditional systems, PDIS is **proactive, legally compliant, and transparent**, ensuring trust and safety online.  
## 3) How to Contribute

- Check the `docs/` folder for detailed project documentation:  
  - `data_dictionary.md` → datasets used  
  - `architecture.md` → system workflow and diagrams  
  - `legal_framework.md` → compliance with IT Act, IT Rules, etc.  
- Fork the repo and open a Pull Request for improvements.  
## 4) Transparency & Evidence
- Every takedown request is logged.  
- Digital evidence is preserved with **hash values, timestamps, and metadata** (valid under Bharatiya Sakshya Adhiniyam 2023).  
- Public portal ensures accountability.  
## 5) File Extensions Used
- **.py** → Python modules & scripts (reusable + testable).  
- **.ipynb** → Jupyter notebooks for EDA, experiments, and visualization.  
- **.yaml** → Config files (easy to tweak without touching code).  
- **.csv / .xlsx** → Datasets (raw & processed).  
- **.pkl** → Saved scikit-learn artifacts (models, vectorizers, encoders).  
- **.pt** → Saved PyTorch transformer weights (for advanced models).  
- **.md** → Documentation files (README, architecture, data dictionary, legal framework).  
