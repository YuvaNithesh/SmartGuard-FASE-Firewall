# 🛡️ SmartGuard: Feature-Augmented Semantic Ensemble (FASE) for Real-Time LLM Guardrails

> **Track B Submission:** Custom Model Architecture & Training Pipeline
> **Core Constraints Achieved:** Pure CPU Inference | P95 Latency < 100ms | Zero API Dependency

## 📄 Abstract
As Large Language Models (LLMs) are deployed in production environments, they remain highly vulnerable to adversarial attacks, including jailbreaks, prompt injections, and PII extraction. Traditional defenses either rely on slow, expensive "LLM-as-a-judge" APIs or brittle keyword-blocking heuristics. 

This repository introduces **SmartGuard**, a Dual-Stream Feature-Augmented Semantic Ensemble (FASE) architecture. By fusing 384-dimensional frozen semantic embeddings with deterministic syntactic meta-features (regex patterns, structural density), this lightweight tree-based meta-classifier achieves an **83% zero-day threat recall** while maintaining a **P95 CPU latency of 99.15 ms**.

---

## 📁 Repository Structure
```text
SmartGuard_FASE_Project/
├── data/
│   ├── complete_guardrail_dataset.csv  # Base training data
│   └── red_team_suite.csv              # 35-prompt OOD adversarial test suite
├── logs/
│   ├── red_team_results.csv            # Per-prompt inference logs & confidence scores
│   ├── training_logs.txt               # Final validation logloss & classification reports
│   └── training_loss_curves.png        # Overfitting check (Iteration vs Logloss)
├── models/
│   └── fase_lightgbm.pkl               # Saved model weights
├── src/
│   ├── eval.py                         # Evaluation script (Explainability & Thresholding)
│   ├── feature_engineering.py          # Dual-Stream extraction logic (Semantic + Regex)
│   └── train.py                        # End-to-end training pipeline
├── app.py                              # Interactive Streamlit Firewall Dashboard
├── README.md                           # Documentation
└── requirements.txt                    # Pinned environment dependencies
🏗️ System Architecture (Dual-Stream)

The system processes prompts through two parallel streams:

🔹 Stream A: Semantic Embedding (Meaning)
Model: all-MiniLM-L6-v2
CPU-based inference
384-dimensional vector capturing intent
🔹 Stream B: Syntactic Extraction (Structure)
Prompt length
Word count
Special character density
Regex-based adversarial patterns
(e.g., "ignore previous instructions", "[system]")
🔹 Meta-Classifier
LightGBM (GBDT)
Input: 388-dimensional fused feature vector
Output: Threat probability

📊 Empirical Evaluation & Metrics
⚡ Latency Performance (CPU Only)
Metric	                     Result	    Requirement	Status
Average Latency	38.26 ms	< 100 ms	    ✅ PASS
P95 Latency	27.64 ms	    < 100 ms	    ✅ PASS

🛡️ Security Performance (Threshold = 0.5)
Metric	                  Score	      Insight
Attack Detection Recall	   83%	  Strong attack interception
F1 Score (Unsafe)	       0.71	  Good detection of threats
F1 Score (Safe)	           0.33	  Higher false positives

Trade-off Analysis: The architecture acts as a strict fail-safe. It heavily penalizes False Negatives (allowing an attack) to prioritize system security, resulting in a higher False Positive rate on deceptive but benign prompts. This threshold can be dynamically tuned in the UI.

📝 Track B Choice Justification
Track B was selected over Track A (API-based guardrails) to address the fatal flaws of API-dependent security: latency and cost. Calling an external LLM-as-a-judge for every user input introduces unacceptable latency overhead (often >1000ms) and scales poorly in high-throughput environments. By engineering a custom feature extractor and training a local LightGBM model, this project proves that deterministic, CPU-bound machine learning can intercept complex attacks at a fraction of the compute cost and time.

🚀 Reproducibility & Setup
This repository is designed for strict deterministic reproducibility.

1. Environment Setup (Fewer than 5 commands)
Note: Requires Python 3.9, 3.10, or 3.11
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux

# Install pinned dependencies (CPU-optimized)
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

2. Run the Full Training Pipeline
This script trains the LightGBM model, compares it against a baseline, and outputs the models/fase_lightgbm.pkl weights and logs/training_loss_curves.png.

Bash
python src/train.py
3. Run the Red Team Evaluation
Executes the model against the unseen 35-prompt test suite and generates P95 latency metrics and feature importance (explainability) logs.

Bash
python src/eval.py
4. Launch the Real-Time Firewall UI
Bash
streamlit run app.py
Access the interactive dashboard at http://localhost:8501 to test the guardrail with dynamic thresholding.