#Evaluate
# ===============================
# 🧪 Model Evaluation Pipeline
# ===============================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    precision_score,
    f1_score # Import f1_score
)

# -------------------------------
# 1. Paths
# -------------------------------
MODEL_DIR = Path("models")
TEST_PATH = Path("data/processed/test_processed.csv")
MODEL_PATH = list(MODEL_DIR.glob("*_model_f1.pkl"))[0]  # auto-pick saved model with f1 optimization
THRESHOLD_PATH = MODEL_DIR / "best_threshold_f1.txt" # Use the F1 optimized threshold

# -------------------------------
# 2. Load Data & Model
# -------------------------------
print("📥 Loading test data and best model...")
test_df = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

X_test = test_df.drop(columns=["No_show_label"])
y_test = test_df["No_show_label"]

print(f"✅ Loaded model: {MODEL_PATH.name}")
print(f"✅ Threshold used: {threshold:.3f}")
print(f"✅ Test samples: {len(X_test)}")

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
f1 = f1_score(y_test, y_pred) # Calculate F1 score
current_precision = precision_score(y_test, y_pred) # Calculate precision

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n📊 Evaluation Metrics (Test Set)")
print(f"  • Precision: {current_precision:.3f}") # Use calculated precision
print(f"  • Recall: {tp / (tp + fn):.3f}") # Calculate recall
print(f"  • F1 Score: {f1:.3f}")
print(f"  • PR AUC: {pr_auc:.3f}")
print(f"  • ROC AUC: {roc_auc:.3f}")
print(f"  • Specificity: {tn / (tn + fp):.3f}")
print(f"  • Sensitivity: {tp / (tp + fn):.3f}")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Show", "No-Show"]))

# -------------------------------
# 5. Confusion Matrix Plot
# -------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Show", "No-Show"],
            yticklabels=["Show", "No-Show"])
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
print("\n🧠 **Insights**")
print(f"  - True Positives (TP): {tp}")
print(f"  - False Negatives (FN): {fn}")
print(f"  - True Negatives (TN): {tn}")
print(f"  - False Positives (FP): {fp}")
print(f"  - No-show prevalence in test: {y_test.mean():.2%}")
print(f"  - Predicted no-show rate: {y_pred.mean():.2%}")

print("\n✅ Evaluation complete. Metrics and plots saved.")
