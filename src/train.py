# src/train.py
import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import xgboost as xgb
import joblib

# --- Load environment variables ---
load_dotenv()
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH"))
MODEL_DIR = Path(os.getenv("MODEL_DIR"))
BEST_MODEL_PATH = Path(os.getenv("BEST_MODEL_PATH"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Load processed data ---
def load_processed_data():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data file not found at {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if "No-show" not in df.columns:
        raise ValueError("Target column 'No-show' not found in processed data.")
    X = df.drop(columns=["No-show"])
    y = df["No-show"]
    return X, y

# --- Evaluate model ---
def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    return roc_auc, pr_auc

# --- Training pipeline ---
def training_pipeline():
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optional: scale numerical columns for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
    }

    best_model = None
    best_pr_auc = -1
    results = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            roc_auc, pr_auc = evaluate_model(model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            roc_auc, pr_auc = evaluate_model(model, X_test, y_test)

        results[name] = {"ROC-AUC": roc_auc, "PR-AUC": pr_auc}

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model = model

    # Save the best model
    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n📊 Model Evaluation Results:")
    for name, scores in results.items():
        print(f"{name:20s} | ROC-AUC: {scores['ROC-AUC']:.4f} | PR-AUC: {scores['PR-AUC']:.4f}")

    print(f"\n🏆 Best model: {type(best_model).__name__} (PR-AUC = {best_pr_auc:.4f})")
    print(f"Model saved to: {BEST_MODEL_PATH}")

    return best_model, results, X_test, y_test

# --- Run training ---
if __name__ == "__main__":
    training_pipeline()
