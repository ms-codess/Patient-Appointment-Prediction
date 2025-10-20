import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Patient No-Show Prediction", layout="wide")
st.title("🏥 Patient No-Show Risk Prediction Dashboard")

# =========================================================
# HELPERS
# =========================================================
TARGET_CANDIDATES = ["No-show_encoded", "No-show"]

def detect_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    return ""

def drop_target_cols(cols):
    return [c for c in cols if c not in TARGET_CANDIDATES]

def clean_xgb_config_string(cfg: str) -> str:
    cfg = re.sub(r'"\[([0-9eE+\-\.]+)\]"', r'"\1"', cfg)
    cfg = re.sub(r'"base_score"\s*:\s*"\[([0-9eE+\-\.]+)\]"', r'"base_score":"\1"', cfg)
    return cfg

@st.cache_resource
def load_model_and_data():
    bundle = joblib.load("models/best_model.joblib")
    if isinstance(bundle, dict):
        model = bundle["model"]
        saved_features = bundle.get("features", None)
    else:
        model = bundle
        saved_features = None

    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    target_col = detect_target_col(df)

    if saved_features:
        feature_cols = saved_features
    else:
        feature_cols = drop_target_cols(df.columns.tolist())

    booster = None
    try:
        booster = model.get_booster()
        cfg = booster.save_config()
        cfg = clean_xgb_config_string(cfg)
        booster.load_config(cfg)
    except Exception:
        booster = None

    return model, booster, df, feature_cols, target_col

@st.cache_resource
def get_expected_features(_model, feature_cols):
    try:
        exp = _model.get_booster().feature_names
        if exp is not None and len(exp) > 0:
            return exp
    except Exception:
        pass
    return feature_cols

# =========================================================
# LOAD MODEL & DATA
# =========================================================
model, booster, data, feature_cols, target_col = load_model_and_data()
if not target_col:
    st.error("❌ Target column not found in dataset.")
    st.stop()

feature_cols = drop_target_cols(feature_cols)
expected_features = get_expected_features(model, feature_cols)
expected_features = [c for c in expected_features if c not in TARGET_CANDIDATES]
background_X = data.reindex(columns=expected_features, fill_value=0)

# =========================================================
# MAPPINGS
# =========================================================
AGE_GROUP_LABELS = {
    "0–1 years old": 1,
    "2–5 years old": 2,
    "6–10 years old": 3,
    "11–15 years old": 4,
    "16–25 years old": 5,
    "26–30 years old": 6,
    "31–35 years old": 7,
    "36–40 years old": 8,
    "41–50 years old": 9,
    "51+ years old": 10
}

AWAITING_TIME_LABELS = {
    "Same day": 1,
    "1–7 days": 2,
    "8–30 days": 3,
    "31–90 days": 4,
    "90+ days": 5
}

CATEGORICAL_MAPPINGS = {
    "Gender": {"Woman": 0, "Man": 1},
    "SMS_received": {"No": 0, "Yes": 1},
    "Hypertension": {"No": 0, "Yes": 1},
    "Diabetes": {"No": 0, "Yes": 1},
    "Alcoholism": {"No": 0, "Yes": 1},
    "Handicap": {"No": 0, "Yes": 1},
    # Internally it's still "Scholarship" but user sees "Medical Coverage"
    "Scholarship": {"No": 0, "Yes": 1},
}

FEATURE_EXPLANATIONS = {
    "Scholarship": "Medical Coverage — patient has government health insurance (Yes/No).",
    "Gender": "Gender of the patient.",
    "SMS_received": "Whether the patient received an SMS reminder.",
    "Hypertension": "Patient has hypertension (Yes/No).",
    "Diabetes": "Patient has diabetes (Yes/No).",
    "Alcoholism": "Patient has a history of alcoholism (Yes/No).",
    "Handicap": "Patient has a physical handicap (Yes/No).",
    "AgeGroup": "Age group of the patient (e.g., 16–25 years old).",
    "AwaitingTimeGroup": "How long between scheduling and appointment.",
    "Appo_Year": "Year of the appointment.",
    "Appo_Month": "Month of the appointment.",
    "Appo_Day": "Day of the appointment.",
    "Sche_Year": "Year the appointment was scheduled.",
    "Sche_Month": "Month the appointment was scheduled.",
    "Sche_Day": "Day the appointment was scheduled."
}

DISPLAY_RENAME = {
    "Scholarship": "Medical Coverage"
}

# =========================================================
# TABS LAYOUT
# =========================================================
tab_predict, tab_explain, tab_performance, tab_about = st.tabs([
    "📊 Prediction",
    "🧠 Explainability",
    "📈 Model Performance",
    "ℹ️ About"
])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab_predict:
    st.subheader("👤 Enter Patient Information")
    user_inputs = {}
    col1, col2 = st.columns(2)

    for col in ["Gender", "SMS_received", "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap"]:
        if col in expected_features:
            display_name = DISPLAY_RENAME.get(col, col)
            user_inputs[col] = col1.selectbox(
                f"{display_name} ❓",
                list(CATEGORICAL_MAPPINGS[col].keys()),
                help=FEATURE_EXPLANATIONS[col]
            )
            user_inputs[col] = CATEGORICAL_MAPPINGS[col][user_inputs[col]]

    if "AgeGroup" in expected_features:
        label = col2.selectbox("Age Group", list(AGE_GROUP_LABELS.keys()), help=FEATURE_EXPLANATIONS["AgeGroup"])
        user_inputs["AgeGroup"] = AGE_GROUP_LABELS[label]

    if "AwaitingTimeGroup" in expected_features:
        label = col2.selectbox("Waiting Time", list(AWAITING_TIME_LABELS.keys()), help=FEATURE_EXPLANATIONS["AwaitingTimeGroup"])
        user_inputs["AwaitingTimeGroup"] = AWAITING_TIME_LABELS[label]

    for col in expected_features:
        if col not in user_inputs:
            user_inputs[col] = int(data[col].median()) if col in data.columns else 0

    user_df = pd.DataFrame([user_inputs])[expected_features]

    if st.button("🚀 Run Prediction"):
        prob = float(model.predict_proba(user_df)[0, 1])
        label_text = "❌ No-Show" if prob >= 0.5 else "✅ Will Attend"
        risk_level = (
            ("🔴 High", "red") if prob >= 0.7 else
            ("🟡 Medium", "orange") if prob >= 0.4 else
            ("🟢 Low", "green")
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label_text)
        c2.metric("No-Show Probability", f"{prob:.1%}")
        c3.metric("Risk Level", risk_level[0])
        st.success(f"✅ Based on the selected information, the predicted risk of no-show is {risk_level[0]}.")

# =========================================================
# TAB 2 — EXPLAINABILITY (with XGBoost fix)
# =========================================================
with tab_explain:
    st.subheader("🧠 SHAP Explainability")

    explainer = None
    shap_sample = background_X.sample(min(50, len(background_X)), random_state=42)

    try:
        booster = model.get_booster()
        booster.feature_names = expected_features  # ✅ critical line
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(shap_sample)
        user_shap = explainer.shap_values(user_df)
    except Exception as e:
        try:
            ke = shap.KernelExplainer(model.predict_proba, shap_sample)
            shap_values = ke.shap_values(shap_sample, nsamples=100)
            user_shap = ke.shap_values(user_df, nsamples=100)
            explainer = ke
        except Exception as e2:
            st.error(f"❌ SHAP failed to initialize: {e2}")
            explainer = None

    if explainer:
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(shap_importance)[::-1][:5]
        top_features = [expected_features[i] for i in top_idx]
        top_impacts = shap_importance[top_idx]

        st.markdown("#### ✨ Top Factors Driving the Prediction:")
        for feat, impact in zip(top_features, top_impacts):
            display_name = DISPLAY_RENAME.get(feat, feat)
            direction = "🔺 increases" if user_shap[0][expected_features.index(feat)] > 0 else "🟩 decreases"
            st.write(f"- **{display_name}** {direction} the likelihood of no-show (impact: {impact:.2f})")

        with st.expander("🔬 Full SHAP Summary Plot"):
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(
                shap_values,
                shap_sample,
                feature_names=[DISPLAY_RENAME.get(f, f) for f in expected_features],
                show=False
            )
            st.pyplot(fig)


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================
with tab_performance:
    st.subheader("📈 Model Evaluation")

    if target_col in data.columns:
        X_all = data.reindex(columns=expected_features, fill_value=0)
        y_all = data[target_col].astype(int)
        y_pred = model.predict(X_all)

        st.text("Classification Report:")
        st.code(classification_report(y_all, y_pred, target_names=["Show", "No-Show"]))

        cm = confusion_matrix(y_all, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["Show", "No-Show"],
                    yticklabels=["Show", "No-Show"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# =========================================================
# TAB 4 — ABOUT
# =========================================================
with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    - **Goal:** Predict whether a patient will **show up** to their medical appointment.
    - **Model:** XGBoost classifier trained on cleaned appointment data.
    - **Explainability:** SHAP (TreeExplainer) provides transparent reasoning.
    - **Medical Coverage:** Refers to whether the patient has **government health insurance**.
    - **Age Group & Waiting Time:** Derived from your preprocessing pipeline.
    """)

