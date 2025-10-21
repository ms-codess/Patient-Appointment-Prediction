#Training

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_recall_curve,
    fbeta_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    f1_score # Import f1_score
)

# ------------------------------
# 1. Paths
# ------------------------------
TRAIN_PATH = Path("data/processed/train_processed_smote.csv") # Updated path to the SMOTE processed training data
VAL_PATH = Path("data/processed/val_processed.csv") # Updated path to the processed validation data
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------------
# 2. Load data
# ------------------------------
print("📥 Loading training and validation data...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.drop(columns=["No_show_label"]) # Corrected column name
y_train = train_df["No_show_label"] # Corrected column name

X_val = val_df.drop(columns=["No_show_label"]) # Corrected column name
y_val = val_df["No_show_label"] # Corrected column name

print(f"✅ Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# ------------------------------
# 3. Model candidates
# ------------------------------
models = {
    "logreg": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "rf": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),
    "xgb": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        # scale_pos_weight is not needed here because SMOTE has balanced the training data
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
}

# ------------------------------
# 4. Train & Evaluate each model
# ------------------------------
def tune_threshold_f1(y_true, y_prob):
    """Return best threshold maximizing F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1_scores = []
    for i in range(len(thr)):
        f1_scores.append(f1_score(y_true, (y_prob >= thr[i]).astype(int)))

    best_idx = np.nanargmax(f1_scores)

    return thr[best_idx], prec[best_idx], rec[best_idx], f1_scores[best_idx]


results = []
best_model_name = None
best_f1 = -1 # Initialize best_f1
best_model = None

print("\n🚀 Training models...\n")
for name, model in models.items():
    print(f"▶ Training {name} ...")
    model.fit(X_train, y_train)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Tune threshold for F1 score
    threshold, p_opt, r_opt, f1_opt = tune_threshold_f1(y_val, y_val_prob) # Use tune_threshold_f1
    pr_auc = average_precision_score(y_val, y_val_prob)

    y_val_pred = (y_val_prob >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)
    current_precision = precision_score(y_val, y_val_pred)


    print(f"\n📊 Model: {name}")
    print(f"  - Best threshold for F1: {threshold:.3f}") # Updated print statement
    print(f"  - Precision at this threshold: {current_precision:.3f}")
    print(f"  - Recall at this threshold: {r_opt:.3f}")
    print(f"  - F1 Score at this threshold: {f1_opt:.3f}") # Added F1 score print
    print(f"  - PR AUC: {pr_auc:.3f}")
    print("  - Confusion Matrix:\n", cm)

    results.append((name, current_precision, r_opt, f1_opt, pr_auc, threshold)) # Added f1_opt to results

    if f1_opt > best_f1: # Compare using F1 score
        best_f1 = f1_opt # Update best_f1
        best_model_name = name
        best_model = model
        best_threshold = threshold

# ------------------------------
# 5. Save best model + threshold
# ------------------------------
print(f"\n🏆 Best Model for F1 Score: {best_model_name} (F1 Score = {best_f1:.3f})") # Updated print statement
joblib.dump(best_model, MODEL_DIR / f"{best_model_name}_model_f1.pkl") # Updated model filename

# Save threshold in a text file
with open(MODEL_DIR / "best_threshold_f1.txt", "w") as f: # Updated threshold filename
    f.write(str(best_threshold))

# ------------------------------
# 6. Final metrics report
# ------------------------------
y_val_prob = best_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= best_threshold).astype(int)

print("\n📈 Final Classification Report (Validation) with F1-Optimized Threshold:") # Updated print statement
print(classification_report(y_val, y_val_pred, target_names=["Show", "No-Show"]))

print("✅ Training completed successfully.")
