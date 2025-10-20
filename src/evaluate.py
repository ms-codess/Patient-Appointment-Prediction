"""
Model evaluation module for Patient Appointment Prediction.

This module provides comprehensive evaluation capabilities for trained models
including metrics calculation, visualization, and performance analysis.
"""
import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    auc
)

from dotenv import load_dotenv

# ==============================================
# 🧰 Helper: Ensure output directories exist
# ==============================================
def ensure_directories():
    output_dirs = [
        Path("reports"),
        Path("reports/plots"),
        Path("reports/metrics")
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

# ==============================================
# 📂 Load model and processed data
# ==============================================
def load_model_and_data():
    load_dotenv()

    PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH"))
    BEST_MODEL_PATH = Path(os.getenv("BEST_MODEL_PATH"))

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data file not found at {PROCESSED_DATA_PATH}")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    # target column name detection
    if "No-show_encoded" in df.columns:
        target_col = "No-show_encoded"
    elif "No-show" in df.columns:
        target_col = "No-show"
    else:
        raise KeyError("Target column not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return model, X, y

# ==============================================
# 📈 ROC Curve
# ==============================================
def plot_roc_curve(y_true, y_prob, model_name):
    ensure_directories()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("reports/plots/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==============================================
# 📉 Precision-Recall Curve
# ==============================================
def plot_pr_curve(y_true, y_prob, model_name):
    ensure_directories()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("reports/plots/pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==============================================
# 🧮 Compute and print detailed metrics
# ==============================================
def print_detailed_report(y_true, y_pred, y_prob, model_name):
    print("\n📈 Performance Metrics:")
    print("----------------------------------------")
    print(f"Accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score       : {f1_score(y_true, y_pred):.4f}")
    print(f"Roc Auc        : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Pr Auc         : {average_precision_score(y_true, y_prob):.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"Specificity    : {specificity:.4f}")
    print(f"Sensitivity    : {sensitivity:.4f}")

    print("\n🔍 Confusion Matrix Details:")
    print("----------------------------------------")
    print(f"True Negatives  (TN): {tn:,}")
    print(f"False Positives (FP): {fp:,}")
    print(f"False Negatives (FN): {fn:,}")
    print(f"True Positives  (TP): {tp:,}")

    print("\n💡 Additional Insights:")
    print("----------------------------------------")
    total = len(y_true)
    actual_positive_rate = y_true.mean() * 100
    predicted_positive_rate = y_pred.mean() * 100
    print(f"Total Samples: {total:,}")
    print(f"Actual No-Show Rate: {actual_positive_rate:.1f}%")
    print(f"Predicted No-Show Rate: {predicted_positive_rate:.1f}%")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['Show', 'No Show']))

# ==============================================
# 🚀 Evaluation Pipeline
# ==============================================
def evaluate_model():
    print("\n🚀 Loading model and data...")
    model, X, y = load_model_and_data()

    print("\n🔮 Generating predictions...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print_detailed_report(y, y_pred, y_prob, "Best Model")

    print("\n📊 Generating visualizations...")
    plot_roc_curve(y, y_prob, "Best Model")
    plot_pr_curve(y, y_prob, "Best Model")

    print("\n✅ Evaluation complete. Results saved in 'reports/plots/'")

# ==============================================
# 🏁 Entry point
# ==============================================
if __name__ == "__main__":
    evaluate_model()