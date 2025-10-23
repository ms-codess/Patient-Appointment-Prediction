"""Training pipeline for Patient Appointment No-Show model.

Loads processed datasets from `data/processed/`, performs Optuna hyperparameter
optimization for a LightGBM classifier, trains the final model, evaluates it on
the validation set, and saves artifacts under `models/` and results under `results/`.
"""

from pathlib import Path
import warnings

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def create_dirs():
    """Create necessary directories relative to project root."""
    root = Path(__file__).resolve().parents[1]
    dirs = {
        "root": root,
        "data": root / "data" / "processed",
        "models": root / "models",
        "results": root / "results",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(dirs):
    """Load and prepare train/validation data."""
    print("Loading processed data...")
    train_df = pd.read_csv(dirs["data"] / "train_processed_smote.csv")
    val_df = pd.read_csv(dirs["data"] / "val_processed.csv")

    X_train = train_df.drop(columns=["No_show_label"])
    y_train = train_df["No_show_label"]
    X_val = val_df.drop(columns=["No_show_label"])
    y_val = val_df["No_show_label"]

    return X_train, X_val, y_train, y_val


def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective balancing F1 and precision."""
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
    }

    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)

    return (f1 * 0.6) + (precision * 0.4)


def save_metrics(metrics, dirs):
    """Save metrics to CSV, appending new results as rows."""
    results_file = dirs["results"] / "metrics.csv"
    current_metrics = pd.DataFrame([metrics])
    if results_file.exists():
        existing = pd.read_csv(results_file)
        out = pd.concat([existing, current_metrics], ignore_index=True)
    else:
        out = current_metrics
    out.to_csv(results_file, index=False)
    print(f"Metrics saved to {results_file}")


def train_optimal_model(X_train, X_val, y_train, y_val, dirs):
    """Train and optimize model, then persist artifact."""
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=30,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params["n_estimators"] = 1000

    final_model = LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    model_path = dirs["models"] / "lightgbm_optimized.pkl"
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    return final_model


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance on validation set."""
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "F1 Score": f1_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "ROC AUC": roc_auc_score(y_val, y_pred_proba),
        "run_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("\nModel Performance (Validation):")
    for k, v in metrics.items():
        if k != "run_date":
            print(f"  - {k}: {v:.4f}")
    return metrics


def main():
    print("Starting training pipeline...")
    dirs = create_dirs()
    X_train, X_val, y_train, y_val = load_data(dirs)
    model = train_optimal_model(X_train, X_val, y_train, y_val, dirs)
    metrics = evaluate_model(model, X_val, y_val)
    save_metrics(metrics, dirs)
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

