import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
from feature_engineering import FASEExtractor

def main():
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Load Data
    data_path = 'data/complete_guardrail_dataset.csv'
    if not os.path.exists(data_path):
        data_path = '../data/complete_guardrail_dataset.csv'
        
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    y = (df['label'] == 'unsafe').astype(int).values
    
    # 2. Extract Features
    extractor = FASEExtractor()
    X = extractor.transform(df['text'].tolist())
    
    # 3. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 4. Train LightGBM & Record Evaluation for the Curve
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}
    evals_result = {} # This stores the data for our loss curve
    
    model = lgb.train(
        params, train_data, num_boost_round=500, valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(30), lgb.record_evaluation(evals_result)]
    )
    
    # 5. Generate and Save the Loss Curve (Artifact 1)
    plt.figure(figsize=(8, 6))
    lgb.plot_metric(evals_result, metric='binary_logloss')
    plt.title('Training vs Validation Loss (Overfitting Check)')
    plt.tight_layout()
    plt.savefig('logs/training_loss_curves.png')
    
    # 6. Generate and Save the Training Logs (Artifact 2)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Unsafe (1)'])
    
    with open('logs/training_logs.txt', 'w') as f:
        f.write("--- FASE Architecture Training Logs ---\n")
        f.write(f"Best Iteration: {model.best_iteration}\n")
        f.write(f"Final Validation LogLoss: {evals_result['valid_1']['binary_logloss'][model.best_iteration-1]:.4f}\n\n")
        f.write("--- Holdout Test Set Report ---\n")
        f.write(report)
        
    # 7. Save Model
    joblib.dump(model, 'models/fase_lightgbm.pkl')
    print("SUCCESS: Generated training_loss_curves.png and training_logs.txt in the logs/ folder!")

if __name__ == "__main__":
    main()