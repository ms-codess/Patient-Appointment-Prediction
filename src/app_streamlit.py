from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


# Compatibility for legacy pickles
class ThresholdedClassifier:
    def __init__(self, base_model, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= float(self.threshold)).astype(int)


def load_df(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else None


def resolve_model_path() -> Path | None:
    p = MODELS_DIR / "best_model.pkl"
    if p.exists():
        return p
    cands = sorted(MODELS_DIR.glob("best_model_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


@st.cache_resource(show_spinner=False)
def load_model_and_data():
    model_path = resolve_model_path()
    model = joblib.load(model_path) if model_path else None
    # unwrap legacy wrapper but preserve threshold
    if model is not None and hasattr(model, "base_model"):
        base = model.base_model
        thr = getattr(model, "threshold", None)
        if thr is not None and not hasattr(base, "threshold"):
            try:
                setattr(base, "threshold", float(thr))
            except Exception:
                pass
        model = base
    train_df = load_df("train_processed_smote.csv")
    val_df = load_df("val_processed.csv")
    test_df = load_df("test_processed.csv")
    return model, model_path, train_df, val_df, test_df


@st.cache_resource(show_spinner=False)
def load_metrics():
    p = RESULTS_DIR / "metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


# Friendly names for display
FRIENDLY = {
    "age": "Age",
    "awaitingtime": "Waiting Time (days)",
    "total_health_issues": "Total Health Issues",
    "age_scholarship": "Age Ã— Insurance",
    "sms_awaitingtime": "SMS Ã— Waiting Time",
    "health_score": "Health Score",
    "multiple_conditions": "Multiple Conditions",
    "same_day_appointment": "Same-day Appointment",
    "long_wait": "Long Wait",
    "scheduledhour": "Scheduled Hour",
    "appointmenthour": "Appointment Hour",
    "scheduleddayofweek": "Scheduled Day of Week",
    "appointmentdayofweek": "Appointment Day of Week",
    "scheduleddayofmonth": "Scheduled Day of Month",
    "isweekend": "Is Weekend",
    "ismonthend": "Is Month End",
    "ismonthstart": "Is Month Start",
    "appointment_month": "Appointment Month",
    "appointment_dayofyear": "Day of Year",
    "gender": "Gender",
    "scholarship": "Insurance",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "handicap": "Handicap",
    "sms_received": "SMS Received",
    "neighbourhood_te": "Neighbourhood Score",
    "neighbourhood_freq": "Neighbourhood Frequency",
    "no_show_label": "No-Show",
}


def friendly(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: FRIENDLY.get(c, c) for c in df.columns})


def compute_metrics(y_true, y_prob, thr: float):
    from sklearn.metrics import confusion_matrix, average_precision_score

    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Specificity": spec,
        "Sensitivity": sens,
        "CM": cm,
        "FPR": fpr,
        "TPR": tpr,
    }


# App layout
st.set_page_config(page_title="No-Show Prediction Dashboard", layout="wide")
st.title("Patient Appointment No-Show: Project Dashboard")

model, model_path, train_df, val_df, test_df = load_model_and_data()
metrics_df = load_metrics()

tabs = st.tabs([
    "Overview",
    "Data",
    "Preprocess",
    "Models",
    "Tuning",
    "Performance",
    "Explain",
    "Business",
    "Demo",
])


# Overview
with tabs[0]:
    st.subheader("Project Overview")
    st.write(
        "We predict patient appointment no-shows to help clinics proactively reduce missed slots,"
        " optimize scheduling, and focus reminders where they matter most. When a no-show is correctly predicted,"
        " staff can call, remind, or reschedule. A missed no-show (false negative) wastes a time slot, while a false"
        " positive is typically just an extra reminder."
    )
    st.markdown("### Dataset")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("The Medical Appointment No Shows dataset contains more than 110,000 records with features covering:")
        st.markdown(
            """
            - Demographics (age, gender)  
            - Scheduling details (appointment and scheduling times, waiting period)  
            - Communication (SMS reminders)  
            - Neighborhood information  
            - Final show or no-show outcome
            """
        )
    with col2:
        st.info("View Dataset: [Kaggle – Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)")
 
    st.markdown(
        """
      
        We trained Logistic Regression and Random Forest baselines and a LightGBM model. We compared them on the validation set using
        F1, recall, precision, and ROC AUC. LightGBM delivered the best overall balance, and we tuned its decision threshold on the
        validation set to maximize F1. The tuned best model is saved and used for the final evaluation on the held-out test set.
        """
    )


# Data
with tabs[1]:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    if train_df is not None:
        c1.metric("Train rows", f"{len(train_df):,}")
    if val_df is not None:
        c2.metric("Validation rows", f"{len(val_df):,}")
    if test_df is not None:
        c3.metric("Test rows", f"{len(test_df):,}")
    if val_df is not None:
        st.write("Class balance (Validation)")
        vc = val_df["no_show_label"].value_counts(normalize=True).rename(index={0: "Show", 1: "No-Show"})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=vc.index, y=vc.values, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        st.pyplot(fig, use_container_width=True)
        st.caption("Validation shows class imbalance; this motivates class weighting and threshold tuning.")
    if train_df is not None:
        st.write("Sample of training features")
        st.dataframe(friendly(train_df.head(10)))


# Preprocess
with tabs[2]:
    st.subheader("Cleaning, Balancing, and Feature Engineering")
    st.write(
        "Data is cleaned, new time and risk features are created, and class imbalance is handled with SMOTE"
        " on the training split."
    )
    if val_df is not None and train_df is not None:
        colA, colB = st.columns(2)
        vc_before = val_df["no_show_label"].value_counts(normalize=True)
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        sns.barplot(x=["Show", "No-Show"], y=[vc_before.get(0, 0.0), vc_before.get(1, 0.0)], ax=ax_b)
        ax_b.set_ylim(0, 1)
        ax_b.set_ylabel("Proportion")
        ax_b.set_title("Before (Validation)")
        colA.pyplot(fig_b)
        colA.caption("Validation is imbalanced with a smaller share of no-shows.")
        vc_after = train_df["no_show_label"].value_counts(normalize=True)
        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        sns.barplot(x=["Show", "No-Show"], y=[vc_after.get(0, 0.0), vc_after.get(1, 0.0)], ax=ax_a)
        ax_a.set_ylim(0, 1)
        ax_a.set_ylabel("Proportion")
        ax_a.set_title("After SMOTE (Training)")
        colB.pyplot(fig_a)
        colB.caption("Training is balanced to help models learn signals from the minority class.")
    if val_df is not None:
        vf = friendly(val_df)
        feat_names = [
            "Waiting Time (days)",
            "Long Wait",
            "Same-day Appointment",
            "Scheduled Hour",
            "Appointment Hour",
            "Appointment Month",
        ]
        present = [c for c in feat_names if c in vf.columns]
        if present:
            cols = st.columns(3)
            interpretations = {
                "Waiting Time (days)": "Longer waits tend to increase no-show risk.",
                "Long Wait": "Extended waiting time is a risk indicator.",
                "Same-day Appointment": "Same-day bookings can behave differently.",
                "Scheduled Hour": "Time of day may shift attendance.",
                "Appointment Hour": "Later hours may correlate with higher risk.",
                "Appointment Month": "Seasonal patterns may exist across months.",
            }
            for i, c in enumerate(present[:3]):
                fig_f, ax_f = plt.subplots(figsize=(6, 4))
                sns.histplot(vf[c], ax=ax_f, kde=False)
                ax_f.set_title(c)
                cols[i].pyplot(fig_f)
                cols[i].caption(interpretations.get(c, "Distribution helps spot skew and outliers."))


# Models
with tabs[3]:
    st.subheader("Model Choices")
    st.write("We combine simple and powerful models suited to structured healthcare data and class imbalance.")
    comp = pd.DataFrame([
        {"Model": "Logistic Regression", "Strengths": "Fast, interpretable, calibrated", "Limits": "Linear boundary", "Why here": "Clear baseline for precision/recall"},
        {"Model": "Random Forest", "Strengths": "Non-linear, robust, low tuning", "Limits": "Larger models", "Why here": "Captures interactions and risk flags"},
        {"Model": "LightGBM", "Strengths": "State-of-the-art on tabular, fast", "Limits": "More knobs", "Why here": "Best accuracy/recall trade-off"},
    ])
    st.table(comp)
    with st.expander("Why not other models?"):
        st.markdown(
            "SVMs can work but scale poorly and require calibration; k-NN is slow at inference and struggles with many features; "
            "Naive Bayes assumes independence; deep learning often underperforms on tabular data; XGBoost is comparable but LightGBM trains faster here."
        )


# Tuning
with tabs[4]:
    st.subheader("Hyperparameter and Threshold Tuning")
    st.write(
        "LightGBM hyperparameters (depth, leaves, learning rate, regularization) are tuned with Optuna to improve validation F1/recall. "
        "Random Forest can be tuned with GridSearchCV (estimators, depth, min samples). We compare validation results across models and then tune "
        "the decision threshold for the best performer to maximize F1."
    )
    if model is not None and val_df is not None:
        Xv = val_df.drop(columns=["no_show_label"])  # original feature names
        yv = val_df["no_show_label"].astype(int)
        prob = model.predict_proba(Xv)[:, 1]
        st.write("Adjust the decision threshold to see how performance changes")
        thr = st.slider("Classification threshold", 0.0, 1.0, float(getattr(model, "threshold", 0.5)), 0.01)
        pred = (prob >= thr).astype(int)
        from sklearn.metrics import confusion_matrix, roc_auc_score
        precision = precision_score(yv, pred, zero_division=0)
        recall = recall_score(yv, pred, zero_division=0)
        f1 = f1_score(yv, pred, zero_division=0)
        roc_auc = roc_auc_score(yv, prob)
        tn, fp, fn, tp = confusion_matrix(yv, pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
        with col2:
            st.metric("F1 Score", f"{f1:.3f}")
            st.metric("ROC AUC", f"{roc_auc:.3f}")
        with col3:
            st.metric("Specificity", f"{specificity:.3f}")
            st.metric("Sensitivity", f"{sensitivity:.3f}")
        pr_p, pr_r, _ = precision_recall_curve(yv, prob)
        fpr, tpr, _ = roc_curve(yv, prob)
        c1, c2 = st.columns(2)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
        ax_pr.plot(pr_r, pr_p)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall")
        c1.pyplot(fig_pr)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr)
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC Curve")
        c2.pyplot(fig_roc)
        st.markdown("### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=["Predicted Show", "Predicted No-Show"],
                    yticklabels=["Actual Show", "Actual No-Show"])
        ax_cm.set_xlabel("")
        ax_cm.set_ylabel("")
        st.pyplot(fig_cm)
    else:
        st.warning("Load data and train a model to explore thresholds.")


# Performance
with tabs[5]:
    st.subheader("Model Performance")
    if not metrics_df.empty:
        ranked = metrics_df.copy()
        if "Variant" not in ranked.columns:
            ranked["Variant"] = ""
        ranked = ranked.sort_values(by="F1", ascending=False)
        st.dataframe(ranked)
        st.caption(
            "Validation results from results/metrics.csv sorted by F1. We select the best model by F1 and tune its threshold. The tuned model is saved as models/best_model.pkl and used for the final test evaluation below."
        )
    else:
        st.warning("No validation metrics found. Run training first.")
    st.markdown("Final evaluation on the test set")
    if model is not None and test_df is not None:
        Xte = test_df.drop(columns=["no_show_label"])  # original names
        yte = test_df["no_show_label"].astype(int)
        prob_te = model.predict_proba(Xte)[:, 1]
        thr_te = float(getattr(model, "threshold", 0.5))
        m = compute_metrics(yte, prob_te, thr_te)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1", f"{m['F1']:.3f}")
        c2.metric("Precision", f"{m['Precision']:.3f}")
        c3.metric("Recall", f"{m['Recall']:.3f}")
        c4.metric("ROC AUC", f"{m['ROC_AUC']:.3f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("PR AUC", f"{m['PR_AUC']:.3f}")
        d2.metric("Sensitivity", f"{m['Sensitivity']:.3f}")
        d3.metric("Specificity", f"{m['Specificity']:.3f}")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(m["CM"], annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix (Test)")
        st.pyplot(fig_cm)
        c5, c6 = st.columns(2)
        fig_pr2, ax_pr2 = plt.subplots(figsize=(6, 4))
        pr_p2, pr_r2, _ = precision_recall_curve(yte, prob_te)
        ax_pr2.plot(pr_r2, pr_p2)
        ax_pr2.set_xlabel("Recall")
        ax_pr2.set_ylabel("Precision")
        ax_pr2.set_title("Precision-Recall")
        c5.pyplot(fig_pr2)
        fig_roc2, ax_roc2 = plt.subplots(figsize=(6, 4))
        ax_roc2.plot(m["FPR"], m["TPR"]) 
        ax_roc2.plot([0, 1], [0, 1], "k--")
        ax_roc2.set_xlabel("FPR")
        ax_roc2.set_ylabel("TPR")
        ax_roc2.set_title("ROC Curve")
        c6.pyplot(fig_roc2)
        st.caption(
            "Test metrics use the tuned threshold saved with the model. Sensitivity is the share of no-shows correctly identified; specificity is the share of shows correctly identified. F1 summarizes precision/recall balance."
        )
    else:
        st.warning("Best model or test set not found.")


# Explain
with tabs[6]:
    st.subheader("Model Explanation (SHAP)")
    if model is None or val_df is None:
        st.warning("Train and evaluate a model first.")
    else:
        try:
            import shap
            base_model = getattr(model, "base_model", model)
            sample_n = min(500, len(val_df))
            X_sample = val_df.sample(n=sample_n, random_state=42)
            X_sample_disp = friendly(X_sample.drop(columns=["no_show_label"]))
            X_sample_model = X_sample.drop(columns=["no_show_label"])  # original names
            try:
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_sample_model)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            except Exception:
                explainer = shap.Explainer(base_model, X_sample_model)
                sv = explainer(X_sample_model)
            st.write("Overall feature influence")
            fig1 = plt.figure(figsize=(8, 4))
            shap.summary_plot(sv, X_sample_disp, show=False)
            st.pyplot(fig1, use_container_width=True)
            st.write("Most important features")
            fig2 = plt.figure(figsize=(8, 4))
            shap.summary_plot(sv, X_sample_disp, plot_type="bar", show=False)
            st.pyplot(fig2, use_container_width=True)
            st.caption("Clear names connect insights to actions, such as focusing on long waits or reminders.")
            st.write("Why was this patient flagged?")
            idx = st.number_input("Pick a validation row index", min_value=0, max_value=len(X_sample_model)-1, value=0, step=1)
            try:
                row = X_sample_model.iloc[int(idx):int(idx)+1]
                row_disp = X_sample_disp.iloc[int(idx):int(idx)+1]
                try:
                    sv_single = shap.TreeExplainer(base_model).shap_values(row)
                    sv_single = sv_single[1] if isinstance(sv_single, list) else sv_single
                    shap_vals = sv_single[0]
                except Exception:
                    exp = shap.Explainer(base_model, X_sample_model)(row)
                    shap_vals = exp.values[0]
                contrib = pd.Series(shap_vals, index=row_disp.columns).sort_values(key=np.abs, ascending=False)[:10]
                fig_local, ax_local = plt.subplots(figsize=(6, 4))
                colors = ["#1f77b4" if v < 0 else "#d62728" for v in contrib[::-1].values]
                contrib[::-1].plot(kind="barh", ax=ax_local, color=colors)
                ax_local.set_title("Top feature contributions (local)")
                st.pyplot(fig_local, use_container_width=True)
            except Exception as e:
                st.info(f"Local explanation unavailable: {e}")
        except Exception as e:
            st.error(f"SHAP could not run: {e}")


# Business
with tabs[7]:
    st.subheader("Business Impact")
    st.write(
        "Use the model to flag high-risk appointments for reminders or rescheduling. Improving recall reduces missed slots; "
        "a moderate precision trade-off is often acceptable operationally."
    )
    if model is not None and test_df is not None:
        Xte = test_df.drop(columns=["no_show_label"]) 
        yte = test_df["no_show_label"].astype(int)
        prob = model.predict_proba(Xte)[:, 1]
        thr = float(getattr(model, "threshold", 0.5))
        m = compute_metrics(yte, prob, thr)
        base_no_show_rate = yte.mean()
        st.write(f"Approximate no-show rate in test: {base_no_show_rate:.1%}")
        assumed_appointments = st.number_input("Appointments per week", min_value=100, max_value=10000, value=1000, step=50)
        expected_no_shows = assumed_appointments * base_no_show_rate
        caught = expected_no_shows * m["Sensitivity"]
        st.write(f"Expected no-shows: {expected_no_shows:.1f}; caught by model: {caught:.1f}")
        st.caption("Simple planning aid; integrate with scheduling to track outcomes.")


# Demo
with tabs[8]:
    st.subheader("Model Inference Demo")
    if model is not None and test_df is not None:
        st.write("Select a test row and tweak values to see the prediction.")
        idx = st.number_input("Row index", min_value=0, max_value=len(test_df)-1, value=0)
        row = test_df.drop(columns=["no_show_label"]).iloc[int(idx)].copy()
        age = st.slider("Age", 0, 100, int(row.get("age", 30)))
        wait = st.slider("Waiting Time (days)", 0, 100, int(row.get("awaitingtime", 5)))
        sms = st.selectbox("SMS Received", [0, 1], index=int(row.get("sms_received", 0)))
        ins = st.selectbox("Insurance", [0, 1], index=int(row.get("scholarship", 0)))
        same_day = st.selectbox("Same-day Appointment", [0, 1], index=int(row.get("same_day_appointment", 0)))
        row.update({
            "age": age,
            "awaitingtime": wait,
            "sms_received": sms,
            "scholarship": ins,
            "same_day_appointment": same_day,
        })
        X = pd.DataFrame([row])
        prob = float(model.predict_proba(X)[:, 1][0])
        thr = float(getattr(model, "threshold", 0.5))
        pred = int(prob >= thr)
        st.metric("Predicted no-show probability", f"{prob:.3f}")
        st.metric("Decision", "No-Show" if pred == 1 else "Show")
    else:
        st.warning("Train and evaluate a model to enable the demo.")
