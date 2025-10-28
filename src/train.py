"""
🤖 Training pipeline for Patient Appointment No-Show prediction.

- Trains Logistic Regression, Random Forest, and LightGBM (with Optuna)
- Compares models on validation set
- Saves best model & evaluation metrics
"""

from pathlib import Path
import warnings
import joblib
import optuna
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")

# -------------------------------
# 0. Directory setup
# -------------------------------
def create_dirs():
    root = Path(__file__).resolve().parents[1]
    dirs = {
        "root": root,
        "data": root / "src" / "data" / "processed",
        "models": root / "models",
        "results": root / "results",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    print("📁 Directories ready:", dirs)
    return dirs

# -------------------------------
# 1. Load data
# -------------------------------
def load_data(dirs):
    print("📥 Loading processed data...")
    train_file = dirs["data"] / "train_processed_smote.csv"
    val_file = dirs["data"] / "val_processed.csv"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    X_train = train_df.drop(columns=["no_show_label"])
    y_train = train_df["no_show_label"]
    X_val = val_df.drop(columns=["no_show_label"])
    y_val = val_df["no_show_label"]

    print(f"✅ Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val

# -------------------------------
# 2. Optuna objective for LightGBM
# -------------------------------
def objective(trial, X_train, X_val, y_train, y_val):
    param = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
    }
    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

# -------------------------------
# 3. Train models
# -------------------------------
def train_logistic_regression(X_train, y_train):
    print("🚀 Training Logistic Regression...")
    # Handle class imbalance to improve precision/recall balance
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("🌲 Training Random Forest...")
    # Class weight helps reduce false positives on imbalanced data
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm_with_optuna(X_train, X_val, y_train, y_val):
    print("⚡ Optimizing LightGBM with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=30)
    best_params = study.best_params
    best_params["n_estimators"] = 500
    print("🏆 Best LightGBM params:", best_params)
    # Add scale_pos_weight to mitigate class imbalance
    pos = int(y_train.sum())
    neg = int((~y_train.astype(bool)).sum()) if hasattr(y_train, 'astype') else int(len(y_train) - y_train.sum())
    if pos > 0:
        best_params.setdefault("scale_pos_weight", max(1.0, neg / pos))
    best_model = LGBMClassifier(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params

# -------------------------------
# 4. Evaluation
# -------------------------------
def evaluate(model, X_val, y_val, name, variant="baseline", threshold=0.5):
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model": name,
        "Variant": variant,
        "Threshold": float(threshold),
        "F1": f1_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "ROC_AUC": roc_auc_score(y_val, y_prob)
    }
    print(f"📊 {name} | F1: {metrics['F1']:.3f} | Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f} | ROC AUC: {metrics['ROC_AUC']:.3f}")
    return metrics


class ThresholdedClassifier:
    def __init__(self, base_model, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(int)


def tune_threshold(model, X_val, y_val, metric: str = "f1"):
    """Find decision threshold on validation set maximizing F1 by default."""
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    # precision_recall_curve returns arrays with len(thresholds) = len(precision) - 1
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])
    return best_threshold

# -------------------------------
# 5. Save metrics + best model
# -------------------------------
def save_results(results, models, dirs):
    results_df = pd.DataFrame(results)
    results_file = dirs["results"] / "metrics.csv"
    if results_file.exists():
        prev = pd.read_csv(results_file)
        out = pd.concat([prev, results_df], ignore_index=True)
    else:
        out = results_df
    out.to_csv(results_file, index=False)
    print(f"📈 Metrics saved/updated at {results_file}")

    # Save artifact for the best by F1 in this batch
    best_row = results_df.sort_values(by="F1", ascending=False).iloc[0]
    best_key = best_row["model"] if "model" in best_row else list(models.keys())[0]
    best_model = models[best_key]

    # Name includes variant when available
    variant = best_row.get("Variant", "")
    variant_suffix = f"_{variant}" if isinstance(variant, str) and variant else ""
    model_path_named = dirs["models"] / f"best_model_{best_key}{variant_suffix}.pkl"
    model_path_stable = dirs["models"] / "best_model.pkl"
    joblib.dump(best_model, model_path_named)
    joblib.dump(best_model, model_path_stable)
    print(f"🏁 Best model '{best_key}{variant_suffix}' saved to {model_path_named} and {model_path_stable}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    dirs = create_dirs()
    X_train, X_val, y_train, y_val = load_data(dirs)

    # Train all models
    log_reg = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    lgbm, best_params = train_lightgbm_with_optuna(X_train, X_val, y_train, y_val)

    # Evaluate (baseline for LR, RF; finetuned for LGBM)
    results = []
    results.append(evaluate(log_reg, X_val, y_val, "LogisticRegression", variant="baseline", threshold=0.5))
    results.append(evaluate(rf, X_val, y_val, "RandomForest", variant="baseline", threshold=0.5))
    results.append(evaluate(lgbm, X_val, y_val, "LightGBM", variant="finetuned", threshold=0.5))

    models = {
        "LogisticRegression": log_reg,
        "RandomForest": rf,
        "LightGBM": lgbm,
    }

    # Pick the best current model by F1 and tune its decision threshold
    tmp_df = pd.DataFrame(results)
    best_row = tmp_df.sort_values(by="F1", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = models[best_name]
    best_threshold = tune_threshold(best_model, X_val, y_val, metric="f1")
    # Attach the tuned threshold to the base model so it can be reloaded without wrapper
    try:
        setattr(best_model, "threshold", float(best_threshold))
    except Exception:
        pass
    thresholded_model = ThresholdedClassifier(best_model, threshold=best_threshold)
    # Log metrics for threshold-tuned variant; use a unique model key
    tuned_key = f"{best_name}_threshold_tuned"
    results.append(evaluate(thresholded_model, X_val, y_val, tuned_key, variant="threshold_tuned", threshold=best_threshold))
    # Map the tuned key to the base model carrying the threshold attribute (not the wrapper)
    models[tuned_key] = best_model

    # Save results (appends to CSV and writes the best artifact as best_model.pkl)
    save_results(results, models, dirs)
