# src/train.py
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
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

# --- Threshold optimisation ---
def find_best_threshold(y_true, y_prob, beta: float = 2.0):
    """Return the probability threshold that maximises the F-beta score."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if thresholds.size == 0:
        # Edge case where model predicts a constant probability
        default_pred = (y_prob >= 0.5).astype(int)
        score = fbeta_score(y_true, default_pred, beta=beta, zero_division=0)
        return 0.5, score, precision[-1], recall[-1]

    best_threshold = 0.5
    best_score = -np.inf
    best_precision = precision[0]
    best_recall = recall[0]

    for threshold, prec, rec in zip(thresholds, precision[1:], recall[1:]):
        predictions = (y_prob >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_precision = prec
            best_recall = rec

    return best_threshold, best_score, best_precision, best_recall


# --- Evaluate model ---
def evaluate_model(
    model,
    X_test,
    y_test,
    beta: float = 2.0,
    threshold: Optional[float] = None,
    search_threshold: bool = True,
):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default)

    if search_threshold or threshold is None:
        best_threshold, best_fbeta, best_precision, best_recall = find_best_threshold(
            y_test, y_pred_proba, beta=beta
        )
    else:
        best_threshold = threshold
        y_pred_best = (y_pred_proba >= best_threshold).astype(int)
        best_fbeta = fbeta_score(y_test, y_pred_best, beta=beta, zero_division=0)
        best_precision = precision_score(y_test, y_pred_best, zero_division=0)
        best_recall = recall_score(y_test, y_pred_best)

    y_pred_best = (y_pred_proba >= best_threshold).astype(int)

    metrics = {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Precision@0.5": precision_default,
        "Recall@0.5": recall_default,
        "Best Threshold": best_threshold,
        "F{:.0f} Score".format(beta): best_fbeta,
        "Precision@BestThreshold": precision_score(
            y_test, y_pred_best, zero_division=0
        ),
        "Recall@BestThreshold": recall_score(y_test, y_pred_best),
    }

    return metrics, y_pred_proba


def cross_validate_model(model, X, y, beta: float, cv: StratifiedKFold):
    oof_pred = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X, y):
        model_clone = clone(model)
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        model_clone.fit(X_train_fold, y_train_fold)
        oof_pred[val_idx] = model_clone.predict_proba(X_val_fold)[:, 1]

    roc_auc = roc_auc_score(y, oof_pred)
    pr_auc = average_precision_score(y, oof_pred)
    y_pred_default = (oof_pred >= 0.5).astype(int)
    precision_default = precision_score(y, y_pred_default, zero_division=0)
    recall_default = recall_score(y, y_pred_default)

    best_threshold, best_fbeta, _, _ = find_best_threshold(
        y, oof_pred, beta=beta
    )

    y_pred_best = (oof_pred >= best_threshold).astype(int)
    best_precision = precision_score(y, y_pred_best, zero_division=0)
    best_recall = recall_score(y, y_pred_best)

    metrics = {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Precision@0.5": precision_default,
        "Recall@0.5": recall_default,
        "Best Threshold": best_threshold,
        "F{:.0f} Score".format(beta): best_fbeta,
        "Precision@BestThreshold": best_precision,
        "Recall@BestThreshold": best_recall,
    }

    return metrics, best_threshold

# --- Training pipeline ---
def training_pipeline():
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    positive_cases = y_train.sum()
    negative_cases = len(y_train) - positive_cases
    scale_pos_weight = (
        float(negative_cases) / float(positive_cases)
        if positive_cases > 0
        else 1.0
    )

    print("\nℹ️ Class distribution in training set:")
    print(f"   • Show (0): {int(negative_cases)}")
    print(f"   • No-Show (1): {int(positive_cases)}")
    print(f"   • Positive rate: {positive_cases / len(y_train):.3f}")

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=5,
        ),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
    }

    best_model_name = None
    best_artifact = None
    best_score = -np.inf
    results = {}
    beta = 2.0
    score_key = f"F{beta:.0f} Score"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_thresholds = {}

    for name, model in models.items():
        metrics, threshold = cross_validate_model(model, X_train, y_train, beta, cv)
        results[name] = metrics
        cv_thresholds[name] = threshold

        if metrics[score_key] > best_score:
            best_score = metrics[score_key]
            best_model_name = name

    if best_model_name is None:
        raise RuntimeError("Training did not select a best model.")

    best_threshold = cv_thresholds[best_model_name]
    best_model = clone(models[best_model_name])
    best_model.fit(X_train, y_train)

    test_metrics, _ = evaluate_model(
        best_model,
        X_test,
        y_test,
        beta=beta,
        threshold=best_threshold,
        search_threshold=False,
    )

    best_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "metrics": {
            "cv": results[best_model_name],
            "test": test_metrics,
        },
        "features": list(X.columns),
        "scale_pos_weight": scale_pos_weight,
        "beta": beta,
    }

    # Save the best model and threshold as an artifact
    joblib.dump(best_artifact, BEST_MODEL_PATH)

    print("\n📊 Cross-validated model performance (train folds):")
    for name, scores in results.items():
        print(
            f"{name:20s} | ROC-AUC: {scores['ROC-AUC']:.4f} | "
            f"PR-AUC: {scores['PR-AUC']:.4f} | "
            f"Recall@0.5: {scores['Recall@0.5']:.4f} | "
            f"Recall@Best: {scores['Recall@BestThreshold']:.4f}"
        )

    print(
        f"\n🏆 Best model (cross-val F{beta:.0f}): {best_model_name} = {best_score:.4f}"
    )
    print(
        f"   • Cross-val optimal threshold: {best_threshold:.3f}\n"
        f"   • Test F{beta:.0f} Score @ threshold: {test_metrics[score_key]:.4f}\n"
        f"   • Test Recall@threshold: {test_metrics['Recall@BestThreshold']:.4f}\n"
        f"   • Test Precision@threshold: {test_metrics['Precision@BestThreshold']:.4f}"
    )
    print(
        f"   • Model saved to: {BEST_MODEL_PATH}"
    )

    return best_artifact, results, test_metrics

# --- Run training ---
if __name__ == "__main__":
    training_pipeline()
