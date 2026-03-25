# Run this temporary snippet in your terminal to save the results file
import pandas as pd
import joblib
from src.feature_engineering import FASEExtractor

df = pd.read_csv('data/red_team_suite.csv')
model = joblib.load('models/fase_lightgbm.pkl')
extractor = FASEExtractor()

results = []
for text in df['text']:
    feat = extractor.transform([text])
    conf = model.predict(feat)[0]
    verdict = "unsafe" if conf > 0.5 else "safe"
    results.append({"text": text, "confidence": conf, "verdict": verdict})

pd.DataFrame(results).to_csv('logs/red_team_results.csv', index=False)
print("Saved logs/red_team_results.csv")