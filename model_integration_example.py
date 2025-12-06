# model_integration_example.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib

# 1) Load dataset (use SynDataset.csv if you have it in repo)
DATA_PATH = "SynDataset.csv"   # or "synthetic_device_dataset.csv"
if not os.path.exists(DATA_PATH):
    # fallback to the synthetic file I produced (if you downloaded it)
    DATA_PATH = "synthetic_device_dataset.csv"

df = pd.read_csv(DATA_PATH)
features = ["heart_rate", "spo2", "temperature", "noise", "voltage"]

# 2) Preprocessing
X = df[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3) Train IsolationForest (typical workflow: train on normal-only)
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)

# 4) Predict & evaluate
scores = -iso.decision_function(X_scaled)     # higher = more anomalous
preds = (iso.predict(X_scaled) == -1).astype(int)  # 1 = anomaly, 0 = normal

print("Confusion matrix:")
print(confusion_matrix(df.get("label", np.zeros(len(df))), preds))
print("Classification report:")
print(classification_report(df.get("label", np.zeros(len(df))), preds))

# 5) (Optional) Load an XGBoost pipeline if it exists in repo
if os.path.exists("xgboost_pipeline.pkl"):
    print("Loading xgboost_pipeline.pkl ...")
    xgb_pipe = joblib.load("xgboost_pipeline.pkl")
    # Example: predict using pipeline
    y_pred = xgb_pipe.predict(df[features])
    print("XGBoost predicted class counts:", pd.Series(y_pred).value_counts())
