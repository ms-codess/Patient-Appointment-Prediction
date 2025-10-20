from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
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
MODEL_PATH = Path("models/best_model.joblib")
DATA_PATH = Path("data/processed/cleaned_dataset.csv")

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
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model = bundle["model"]
        saved_features = bundle.get("features", None)
    else:
        model = bundle
        saved_features = None

    df = pd.read_csv(DATA_PATH)
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
if not MODEL_PATH.exists():
    st.error(
        "❌ Trained model artifact not found. Please run the training pipeline to create "
        f"`{MODEL_PATH}` before launching the dashboard."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "❌ Processed dataset not found. Expected to load it from "
        f"`{DATA_PATH}`. Please preprocess the data first."
    )
    st.stop()

model, booster, data, feature_cols, target_col = load_model_and_data()
if not target_col:
    st.error("❌ Target column not found in dataset.")
    st.stop()

feature_cols = drop_target_cols(feature_cols)
expected_features = get_expected_features(model, feature_cols)
expected_features = [c for c in expected_features if c not in TARGET_CANDIDATES]
background_X = data.reindex(columns=expected_features, fill_value=0)
feature_index_map = {feat: idx for idx, feat in enumerate(expected_features)}

with st.sidebar:
    st.header("📂 Dataset Overview")
    st.metric("Records", f"{len(data):,}")
    if target_col in data.columns:
        numeric_target = pd.to_numeric(data[target_col], errors="coerce")
        if not numeric_target.dropna().empty:
            no_show_rate = numeric_target.mean()
            st.metric("No-Show Rate", f"{no_show_rate:.1%}")

    st.caption("Preview of processed features")
    preview_cols = [c for c in expected_features[:6] if c in data.columns]
    if target_col and target_col in data.columns:
        preview_cols.append(target_col)
    if preview_cols:
        st.dataframe(data[preview_cols].head(), use_container_width=True)
    else:
        st.info("No feature columns available for preview.")


def ensure_array(values):
    """SHAP sometimes returns a list (multi-class). Convert to a 2D numpy array."""
    if isinstance(values, list):
        if len(values) > 1:
            return np.array(values[1])
        return np.array(values[0])
    return np.array(values)

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

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    with st.form("prediction_form"):
        user_inputs = {}
        col1, col2 = st.columns(2)

        for col in ["Gender", "SMS_received", "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap"]:
            if col in expected_features:
                display_name = DISPLAY_RENAME.get(col, col)
                selection = col1.selectbox(
                    f"{display_name} ❓",
                    list(CATEGORICAL_MAPPINGS[col].keys()),
                    help=FEATURE_EXPLANATIONS[col],
                    key=f"input_{col}"
                )
                user_inputs[col] = CATEGORICAL_MAPPINGS[col][selection]

        if "AgeGroup" in expected_features:
            age_label = col2.selectbox(
                "Age Group",
                list(AGE_GROUP_LABELS.keys()),
                help=FEATURE_EXPLANATIONS["AgeGroup"],
                key="input_age"
            )
            user_inputs["AgeGroup"] = AGE_GROUP_LABELS[age_label]

        if "AwaitingTimeGroup" in expected_features:
            wait_label = col2.selectbox(
                "Waiting Time",
                list(AWAITING_TIME_LABELS.keys()),
                help=FEATURE_EXPLANATIONS["AwaitingTimeGroup"],
                key="input_wait"
            )
            user_inputs["AwaitingTimeGroup"] = AWAITING_TIME_LABELS[wait_label]

        for col in expected_features:
            if col not in user_inputs:
                user_inputs[col] = int(data[col].median()) if col in data.columns else 0

        user_df = pd.DataFrame([user_inputs])[expected_features]

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

    if submitted:
        prob = float(model.predict_proba(user_df)[0, 1])
        label_text = "❌ No-Show" if prob >= 0.5 else "✅ Will Attend"
        risk_level = (
            ("🔴 High", "red") if prob >= 0.7 else
            ("🟡 Medium", "orange") if prob >= 0.4 else
            ("🟢 Low", "green")
        )
        st.session_state.prediction = {
            "prob": prob,
            "label_text": label_text,
            "risk_level": risk_level,
        }
        st.session_state.user_input_df = user_df

    if st.session_state.prediction:
        pred = st.session_state.prediction
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred["label_text"])
        c2.metric("No-Show Probability", f"{pred['prob']:.1%}")
        c3.metric("Risk Level", pred["risk_level"][0])
        st.success(
            "✅ Based on the selected information, the predicted risk of no-show is "
            f"{pred['risk_level'][0]}."
        )

        with st.expander("📋 Patient inputs used for this prediction"):
            display_df = st.session_state.user_input_df.copy()
            display_df.columns = [DISPLAY_RENAME.get(col, col) for col in display_df.columns]
            st.dataframe(display_df)

# =========================================================
# TAB 2 — EXPLAINABILITY (with XGBoost fix)
# =========================================================
with tab_explain:
    st.subheader("🧠 SHAP Explainability")

    if not st.session_state.get("prediction"):
        st.info("Run a prediction from the previous tab to unlock SHAP explanations.")
    else:
        user_df = st.session_state.get("user_input_df")
        if user_df is None:
            st.warning("Couldn't locate the latest patient inputs. Please re-run a prediction.")
            st.stop()

        if background_X.empty:
            st.warning("Background data unavailable – cannot compute SHAP values without reference samples.")
        else:
            explainer = None
            shap_sample = background_X.sample(min(50, len(background_X)), random_state=42)

            try:
                booster = model.get_booster()
                booster.feature_names = expected_features  # ensure alignment for XGBoost
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(shap_sample)
                user_shap = explainer.shap_values(user_df)
            except Exception:
                try:
                    ke = shap.KernelExplainer(model.predict_proba, shap_sample)
                    shap_values = ke.shap_values(shap_sample, nsamples=100)
                    user_shap = ke.shap_values(user_df, nsamples=100)
                    explainer = ke
                except Exception as e2:
                    st.error(f"❌ SHAP failed to initialize: {e2}")
                    explainer = None

            if explainer:
                shap_values = ensure_array(shap_values)
                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(1, -1)

                user_shap = ensure_array(user_shap)
                user_contrib = user_shap[0] if user_shap.ndim > 1 else user_shap

                shap_importance = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(shap_importance)[::-1][:5]
                top_features = [expected_features[i] for i in top_idx]
                top_impacts = shap_importance[top_idx]

                st.markdown("#### ✨ Top Factors Driving the Prediction:")
                for feat, impact in zip(top_features, top_impacts):
                    display_name = DISPLAY_RENAME.get(feat, feat)
                    direction = "🔺 increases" if user_contrib[feature_index_map[feat]] > 0 else "🟩 decreases"
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

