import pandas as pd
import joblib
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(__file__))
from feature_engineering import FASEExtractor

def run_research_evaluation():
    print("--- Phase B: Advanced Research Evaluation & Explainability ---")
    
    # 1. Path Management
    data_path = 'data/red_team_suite.csv'
    model_path = 'models/fase_lightgbm.pkl'
    
    if not os.path.exists(data_path):
        data_path = '../data/red_team_suite.csv'
        model_path = '../models/fase_lightgbm.pkl'
        
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find {data_path}. Ensure your 35-row red_team_suite.csv is in the data folder.")
        return
        
    # 2. Load Data and Models
    df = pd.read_csv(data_path)
    y_true = (df['label'] == 'unsafe').astype(int).values
    
    extractor = FASEExtractor()
    model = joblib.load(model_path)
    
    # 3. Running Inference and timing
    print(f"\nEvaluating {len(df)} adversarial prompts...")
    latencies = []
    probs = []
    
    for text in df['text']:
        start_time = time.time()
        features = extractor.transform([text])
        prob = model.predict(features)[0]
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000)
        probs.append(prob)
        
    probs = np.array(probs)

    # --- ADVANCED RESEARCH OUTPUTS ---

    # A. Feature Importance (Explainability)
    print("\n--- [1] EXPLAINABILITY: FEATURE IMPORTANCE (GAIN) ---")
    # importance_type='gain' shows which features contributed most to the decision
    importance = model.feature_importance(importance_type='gain')
    print(f"Semantic Embeddings (Avg Gain across 384 dims): {np.mean(importance[:-4]):.4f}")
    print(f"Meta - Character Length Gain:  {importance[-4]:.4f}")
    print(f"Meta - Word Count Gain:       {importance[-3]:.4f}")
    print(f"Meta - Character Density Gain: {importance[-2]:.4f}")
    print(f"Meta - Regex Pattern Hits Gain: {importance[-1]:.4f}")

    # B. Threshold Analysis (Trade-off Research)
    print("\n--- [2] THRESHOLD VS. PERFORMANCE TRADE-OFF ---")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        y_pred_t = (probs > t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        print(f"Threshold {t:.1f} | F1-Score: {f1:.3f} | Blocks: {sum(y_pred_t)}/{len(df)}")

    # C. P95 Latency
    p95_latency = np.percentile(latencies, 95)
    print(f"\n--- [3] LATENCY METRICS ---")
    print(f"P95 CPU Latency: {p95_latency:.2f} ms")

    # D. Failure Analysis (The 5-Prompt Deep Dive)
    print("\n--- [4] FAILURE ANALYSIS: MODEL ERROR SAMPLES ---")
    y_pred_final = (probs > 0.5).astype(int)
    
    print("\nFALSE POSITIVES (Safe prompts wrongly blocked):")
    fp_count = 0
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred_final, df['text'])):
        if true == 0 and pred == 1 and fp_count < 3:
            print(f" - Confidence {probs[i]:.2%}: '{text}'")
            fp_count += 1
            
    print("\nFALSE NEGATIVES (Attacks that slipped through):")
    fn_count = 0
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred_final, df['text'])):
        if true == 1 and pred == 0 and fn_count < 3:
            print(f" - Confidence {probs[i]:.2%}: '{text}'")
            fn_count += 1

if __name__ == "__main__":
    run_research_evaluation()