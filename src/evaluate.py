"""Model evaluation on test split.

Loads the saved model from `models/`, the processed test split from
`data/processed/`, computes metrics, prints a report, and saves plots.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
import warnings

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Support loading legacy pickles that referenced a custom wrapper
# defined under __main__.ThresholdedClassifier during training.
# If present in the pickle, having this class here prevents load errors.
class ThresholdedClassifier:
    def __init__(self, base_model, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        import numpy as np
        prob = self.predict_proba(X)[:, 1]
        return (prob >= float(self.threshold)).astype(int)


# -------------------------------
# 1. Paths
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
# Keep paths consistent with training (which uses src/data/processed)
TEST_PATH = ROOT / "src" / "data" / "processed" / "test_processed.csv"

# Prefer the generic best model if present; else fall back
def _resolve_model_path():
    if (MODEL_DIR / "best_model.pkl").exists():
        return MODEL_DIR / "best_model.pkl"
    # If best_model_<Name>.pkl exists, pick the most recent
    candidates = sorted(MODEL_DIR.glob("best_model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    # Legacy fallbacks
    for name in ["lightgbm_finetuned.pkl", "lightgbm_optimized.pkl", "randomforest_finetuned.pkl"]:
        p = MODEL_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No trained model artifact found in {MODEL_DIR}")

MODEL_PATH = _resolve_model_path()


# -------------------------------
# 2. Load Data & Model
# -------------------------------
print("Loading test data and model...")
test_df = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)
# If model carries a tuned threshold attribute, use it; else default 0.5
threshold = float(getattr(model, "threshold", 0.50))

# Align with preprocessing/training target naming
X_test = test_df.drop(columns=["no_show_label"])  # features
y_test = test_df["no_show_label"].astype(int)

print(f"Model: {MODEL_PATH.name}")
print(f"Threshold: {threshold:.3f}")
print(f"Test samples: {len(X_test)}")


# -------------------------------
# 3. Predictions
# -------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)


# -------------------------------
# 4. Metrics
# -------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
current_precision = precision_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nEvaluation Metrics (Test Set)")
print(f"  - Precision: {current_precision:.3f}")
print(f"  - Recall: {tp / (tp + fn):.3f}")
print(f"  - F1 Score: {f1:.3f}")
print(f"  - PR AUC: {pr_auc:.3f}")
print(f"  - ROC AUC: {roc_auc:.3f}")
print(f"  - Specificity: {tn / (tn + fp):.3f}")
print(f"  - Sensitivity: {tp / (tp + fn):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Show", "No-Show"]))


# -------------------------------
# 5. Confusion Matrix Plot
# -------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Show", "No-Show"],
    yticklabels=["Show", "No-Show"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.show()


# -------------------------------
# 6. Precision-Recall Curve
# -------------------------------
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.show()


# -------------------------------
# 7. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()


# -------------------------------
# 8. Insights
# -------------------------------
print("\nInsights")
print(f"  - True Positives (TP): {tp}")
print(f"  - False Negatives (FN): {fn}")
print(f"  - True Negatives (TN): {tn}")
print(f"  - False Positives (FP): {fp}")
print(f"  - No-show prevalence in test: {y_test.mean():.2%}")
print(f"  - Predicted no-show rate: {y_pred.mean():.2%}")

print("\nGenerating SHAP explanations...")
try:
    import shap
    import numpy as np
    base_model = getattr(model, "base_model", model)
    # Use a small background/sample for efficiency
    sample_n = min(200, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=42) if sample_n > 0 else X_test
    # Choose explainer based on model type
    explainer = None
    shap_values = None
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
    except Exception:
        try:
            explainer = shap.Explainer(base_model, X_sample)
            sv = explainer(X_sample)
        except Exception:
            explainer = shap.LinearExplainer(base_model, X_sample)
            sv = explainer.shap_values(X_sample)

    # Summary (beeswarm)
    shap.summary_plot(sv, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    # Global importance (bar)
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP plots saved: shap_summary.png, shap_importance.png")
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
except Exception as e:
    print(f"SHAP explanation skipped due to error: {e}")

print("\nEvaluation complete. Metrics and plots saved.")
