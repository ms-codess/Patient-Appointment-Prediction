"""
🤖 Training pipeline for Patient Appointment No-Show prediction

- Stage 1: Train Logistic Regression, Random Forest, LightGBM (baseline)
- Stage 2: Fine-tune Random Forest (GridSearchCV) and LightGBM (Optuna)
- Save all metrics to results/metrics.csv (append, not overwrite)
- Save best model artifact in models/
"""

from pathlib import Path
import warnings
import joblib
import optuna
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

# =====================================================
# 0. Helper functions
# =====================================================
def create_dirs():
    root = Path(__file__).resolve().parents[1]
    dirs = {
        "root": root,
        "data": root/"src" / "data" / "processed",
        "models": root / "models",
        "results": root / "results",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    return dirs

def load_data(dirs):
    print("📥 Loading processed data...")
    train_df = pd.read_csv(dirs["data"] / "train_processed_smote.csv")
    val_df = pd.read_csv(dirs["data"] / "val_processed.csv")

    X_train = train_df.drop(columns=["no_show_label"])
    y_train = train_df["no_show_label"]
    X_val = val_df.drop(columns=["no_show_label"])
    y_val = val_df["no_show_label"]

    return X_train, X_val, y_train, y_val

def evaluate(model, X_val, y_val, name, label="baseline"):
    """Evaluate model performance on validation set."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "Model": name,
        "Type": label,
        "F1": f1_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "ROC_AUC": roc_auc_score(y_val, y_prob),
        "Run_Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def save_metrics(metrics_list, dirs):
    """Append metrics to metrics.csv instead of overwriting."""
    results_file = dirs["results"] / "metrics.csv"
    current = pd.DataFrame(metrics_list)
    if results_file.exists():
        old = pd.read_csv(results_file)
        out = pd.concat([old, current], ignore_index=True)
    else:
        out = current
    out.to_csv(results_file, index=False)
    print(f"📊 Metrics saved/updated at {results_file}")

# =====================================================
# 1. Baseline training
# =====================================================
def train_baseline_models(X_train, X_val, y_train, y_val):
    models = {}
    metrics = []

    #  Logistic Regression
    print("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    models["LogisticRegression_baseline"] = lr
    metrics.append(evaluate(lr, X_val, y_val, "LogisticRegression"))

    # Random Forest
    print("Training Random Forest (baseline)...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest_baseline"] = rf
    metrics.append(evaluate(rf, X_val, y_val, "RandomForest"))

    #  LightGBM
    print("Training LightGBM (baseline)...")
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    lgbm.fit(X_train, y_train)
    models["LightGBM_baseline"] = lgbm
    metrics.append(evaluate(lgbm, X_val, y_val, "LightGBM"))

    return models, metrics

# =====================================================
# 2. Fine-tuning Random Forest (GridSearchCV)
# =====================================================
def finetune_random_forest(X_train, y_train):
    print("🌲 Fine-tuning Random Forest with GridSearchCV...")
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("✅ Best RF params:", grid.best_params_)
    return grid.best_estimator_

# =====================================================
# 3. Fine-tuning LightGBM (Optuna)
# =====================================================
def objective(trial, X_train, X_val, y_train, y_val):
    params = {
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
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

def finetune_lightgbm(X_train, X_val, y_train, y_val):
    print("Fine-tuning LightGBM with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=30)
    best_params = study.best_params
    best_params["n_estimators"] = 500
    print("✅ Best LGBM params:", best_params)
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

# =====================================================
# 4. Main
# =====================================================
def main():
    dirs = create_dirs()
    X_train, X_val, y_train, y_val = load_data(dirs)

    # Stage 1: Baseline models
    baseline_models, baseline_metrics = train_baseline_models(X_train, X_val, y_train, y_val)
    save_metrics(baseline_metrics, dirs)

    # Stage 2: Fine-tuning selected models
    rf_best = finetune_random_forest(X_train, y_train)
    rf_metrics = evaluate(rf_best, X_val, y_val, "RandomForest", label="finetuned")
    joblib.dump(rf_best, dirs["models"] / "randomforest_finetuned.pkl")

    lgbm_best = finetune_lightgbm(X_train, X_val, y_train, y_val)
    lgbm_metrics = evaluate(lgbm_best, X_val, y_val, "LightGBM", label="finetuned")
    joblib.dump(lgbm_best, dirs["models"] / "lightgbm_finetuned.pkl")

    save_metrics([rf_metrics, lgbm_metrics], dirs)

    print("\n🏁 Final Metrics Table:")
    print(pd.read_csv(dirs["results"] / "metrics.csv").sort_values(by="F1", ascending=False))

if __name__ == "__main__":
    main()
