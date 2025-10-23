from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Patient No-Show Prediction", layout="wide")
st.title("Patient No-Show Risk Prediction Dashboard")


# =========================================================
# PATHS & CONSTANTS
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
TARGET_COL = "No_show_label"

MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "train_processed_smote.csv"
TEST_DATA_PATH = ROOT / "data" / "processed" / "test_processed.csv"
THRESHOLD_PATH = MODEL_DIR / "best_threshold_f1.txt"

# Resolve model: prefer the one saved by train.py, else first .pkl in models/
preferred_model = MODEL_DIR / "lightgbm_optimized.pkl"
candidate_pkls = sorted(MODEL_DIR.glob("*.pkl"))
if preferred_model.exists():
    MODEL_PATH = preferred_model
elif candidate_pkls:
    MODEL_PATH = candidate_pkls[0]
else:
    st.error(f"No model artifacts found in {MODEL_DIR}. Run the training pipeline first.")
    raise SystemExit(1)


# =========================================================
# HELPERS
# =========================================================
def detect_target_col(df: pd.DataFrame) -> str:
    return TARGET_COL if TARGET_COL in df.columns else ""


def drop_target_col(cols):
    return [c for c in cols if c != TARGET_COL]


def clean_xgb_config_string(cfg: str) -> str:
    # Make SHAP/XGBoost config robust if needed (harmless for LightGBM/sklearn)
    import re
    cfg = re.sub(r'"\[([0-9eE+\-\.]+)\]"', r'\1', cfg)
    cfg = re.sub(r'"base_score"\s*:\s*([^,}\]]+)', r'"base_score":\1', cfg)
    return cfg


def ensure_array(values):
    # SHAP sometimes returns lists; convert to numpy array consistently
    if isinstance(values, list):
        # For binary classification TreeExplainer, it returns [class0, class1]; take positive class (index 1) if present.
        if len(values) == 2:
            return np.array(values[1])
        return np.array(values[0] if values else [])
    return np.array(values)


def cast_like_training(df_like: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    for col in df_like.columns:
        if col in ref_df.columns:
            try:
                df_like[col] = df_like[col].astype(ref_df[col].dtype)
            except Exception:
                # If casting fails, try numeric coercion then fill
                df_like[col] = pd.to_numeric(df_like[col], errors="coerce").fillna(ref_df[col].median() if pd.api.types.is_numeric_dtype(ref_df[col]) else 0)
                try:
                    df_like[col] = df_like[col].astype(ref_df[col].dtype)
                except Exception:
                    pass
        else:
            # Not present in training; keep as-is (it will be dropped when reindexing)
            pass
    return df_like


@st.cache_resource
def load_model_and_data():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Unable to load model at {MODEL_PATH}: {e}")
        raise SystemExit(1)

    # Load data
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Unable to load data at {DATA_PATH}: {e}")
        raise SystemExit(1)

    target_col = detect_target_col(df)
    feature_cols = drop_target_col(df.columns.tolist())

    # Optional: configure XGBoost booster for SHAP (no-op for LightGBM/sklearn)
    booster = None
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            cfg = booster.save_config()
            cfg = clean_xgb_config_string(cfg)
            booster.load_config(cfg)
            if hasattr(booster, "feature_names"):
                booster.feature_names = feature_cols
    except Exception as e:
        st.warning(f"Could not configure XGBoost booster; SHAP may fall back. Error: {e}")
        booster = None

    return model, booster, df, feature_cols, target_col


def get_expected_features(_model, feature_cols):
    # Try to get feature names from model; fallback to passed feature_cols
    try:
        # XGBoost booster path
        if hasattr(_model, "get_booster"):
            booster_features = _model.get_booster().feature_names
            if booster_features:
                return [f for f in booster_features if f in feature_cols]
    except Exception:
        pass

    # LightGBM/scikit-learn fallback
    return feature_cols


# =========================================================
# LOAD MODEL & DATA
# =========================================================
if not MODEL_PATH.exists():
    st.error(f"Trained model not found at `{MODEL_PATH}`. Please run training first.")
    raise SystemExit(1)

if not DATA_PATH.exists():
    st.error(f"Processed dataset not found at `{DATA_PATH}`. Please run preprocessing first.")
    raise SystemExit(1)

model, booster, data, feature_cols, target_col = load_model_and_data()
if not target_col:
    st.error(f"Target column '{TARGET_COL}' not found in dataset.")
    raise SystemExit(1)

expected_features = [c for c in feature_cols if c != TARGET_COL]
background_X = data.reindex(columns=expected_features, fill_value=0)

# Load the best threshold with a safe fallback, silently
best_threshold = 0.50
if THRESHOLD_PATH.exists():
    try:
        with open(THRESHOLD_PATH, "r") as f:
            best_threshold = float(f.read().strip())
    except Exception:
        pass


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Dataset Overview")
    st.metric("Records", f"{len(data):,}")

    if target_col in data.columns:
        numeric_target = pd.to_numeric(data[target_col], errors="coerce")
        if not numeric_target.dropna().empty:
            no_show_rate = numeric_target.mean()
            st.metric("No-show Rate", f"{no_show_rate:.1%}")

    st.caption("Feature preview")
    preview_cols = [c for c in expected_features[:6] if c in data.columns]
    if target_col and target_col in data.columns:
        preview_cols.append(target_col)
    if preview_cols:
        st.dataframe(data[preview_cols].head(), use_container_width=True)
    else:
        st.info("No feature columns available for preview.")

    st.markdown("---")
    st.header("Model Controls")
    st.write(f"Model Type: {type(model).__name__}")
    st.write(f"Model File: {MODEL_PATH.name}")
    threshold = st.slider("Operating Threshold", min_value=0.0, max_value=1.0, value=float(best_threshold), step=0.01, help="Adjust to explore precision/recall trade-off.")


# =========================================================
# TABS
# =========================================================
tab_predict, tab_explain, tab_performance, tab_about = st.tabs(
    ["Prediction", "Explainability", "Model Performance", "About"]
)


# =========================================================
# TAB: PREDICTION
# =========================================================
with tab_predict:
    st.subheader("Make a Prediction")

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "user_input_df" not in st.session_state:
        st.session_state.user_input_df = None

    # Simple, user-friendly inputs
    day_name_to_int = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    gender_to_num = {"Female": 0.0, "Male": 1.0, "Other/Unknown": 0.5}

    def build_row_from_simple_inputs(df_processed: pd.DataFrame, inputs: dict) -> pd.DataFrame:
        features_df = df_processed.drop(columns=[TARGET_COL])
        baseline = features_df.median(numeric_only=True)
        row = baseline.copy()
        row["Gender"] = gender_to_num[inputs["gender"]]
        row["Scholarship"] = int(inputs["scholarship"])
        row["Hypertension"] = int(inputs["hypertension"])
        row["Diabetes"] = int(inputs["diabetes"])
        row["Alcoholism"] = int(inputs["alcoholism"])
        row["Handicap"] = int(inputs["handicap"])
        row["SMS_received"] = int(inputs["sms"])
        row["ScheduledHour"] = int(inputs["scheduled_hour"])
        row["AppointmentHour"] = int(inputs["appointment_hour"])
        row["AppointmentDayOfWeek"] = day_name_to_int[inputs["appt_dow"]]
        row["IsWeekend"] = 1 if row["AppointmentDayOfWeek"] in [5, 6] else 0
        row["Appointment_Month"] = int(inputs["month"])
        row["Same_Day_Appointment"] = 1 if inputs["same_day"] else 0
        row["Multiple_Conditions"] = 1 if (row["Hypertension"] + row["Diabetes"] + row["Alcoholism"] + row["Handicap"]) > 1 else 0
        row["Age_Scholarship"] = row.get("Age", 0) * row["Scholarship"]
        row["SMS_AwaitingTime"] = row.get("AwaitingTime", 0) * row["SMS_received"]
        row = row.reindex(index=expected_features, fill_value=0)
        return row.to_frame().T

    with st.form("simple_predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", list(gender_to_num.keys()), index=0)
            scholarship = st.checkbox("Scholarship", value=False)
            sms = st.checkbox("SMS received", value=False)
        with col2:
            hypertension = st.checkbox("Hypertension", value=False)
            diabetes = st.checkbox("Diabetes", value=False)
            alcoholism = st.checkbox("Alcoholism", value=False)
        with col3:
            handicap = st.selectbox("Handicap", [0, 1, 2], index=0)
            appt_dow = st.selectbox("Appointment Day", list(day_name_to_int.keys()), index=0)
            month = st.selectbox("Month", list(range(1, 13)), index=0)

        col4, col5, col6 = st.columns(3)
        with col4:
            scheduled_hour = st.slider("Scheduled Hour", 0, 23, 9)
        with col5:
            appointment_hour = st.slider("Appointment Hour", 0, 23, 14)
        with col6:
            same_day = st.checkbox("Same-day appointment", value=False)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        simple_inputs = {
            "gender": gender,
            "scholarship": scholarship,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "alcoholism": alcoholism,
            "handicap": handicap,
            "sms": sms,
            "appt_dow": appt_dow,
            "month": month,
            "scheduled_hour": scheduled_hour,
            "appointment_hour": appointment_hour,
            "same_day": same_day,
        }

        X_row = build_row_from_simple_inputs(data, simple_inputs)
        X_row = cast_like_training(X_row, data.drop(columns=[TARGET_COL]))

        try:
            prob = float(model.predict_proba(X_row)[0, 1])
        except Exception:
            try:
                prob = float(model.decision_function(X_row))
                prob = 1.0 / (1.0 + np.exp(-prob))
            except Exception as e:
                st.error(f"Model cannot produce probability: {e}")
                raise st.stop()

        predicted_class = "No-Show" if prob >= float(threshold) else "Will Attend"
        risk_level = (
            ("High", "red") if prob >= float(threshold) + (1 - float(threshold)) * 0.2
            else ("Medium", "orange") if prob >= float(threshold)
            else ("Low", "green")
        )

        st.session_state.prediction = {"prob": prob, "predicted_class": predicted_class, "risk_level": risk_level}
        st.session_state.user_input_df = X_row.copy()

    if st.session_state.prediction:
        pred = st.session_state.prediction
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred["predicted_class"])
        c2.metric("No-Show Probability", f"{pred['prob']:.1%}")
        c3.metric("Risk Level", pred["risk_level"][0])
        st.success(
            f"Based on the selected information, the predicted risk of no-show is {pred['risk_level'][0]}."
        )

        with st.expander("Input features used for this prediction"):
            st.dataframe(st.session_state.user_input_df)


# =========================================================
# TAB: EXPLAINABILITY
# =========================================================
with tab_explain:
    st.subheader("SHAP Explainability")

    if not st.session_state.get("prediction"):
        st.info("Run a prediction from the 'Prediction' tab to unlock SHAP explanations.")
    else:
        user_df = st.session_state.get("user_input_df")
        if user_df is None or background_X.empty:
            st.warning("Cannot compute SHAP values without the patient's input or background data.")
            if user_df is None:
                st.info("Please run a prediction to capture the latest patient inputs.")
            if background_X.empty:
                st.error("Background data is empty. Check data loading.")
            st.stop()

        st.info("Calculating SHAP values on a background sample (up to 100 rows)...")
        progress_bar = st.progress(0)

        explainer = None
        shap_sample_size = min(100, len(background_X))
        shap_sample = background_X.sample(shap_sample_size, random_state=42)

        try:
            # Tree-based models typically work with TreeExplainer (LightGBM is supported)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_sample)
            user_shap = explainer.shap_values(user_df)
        except Exception:
            # Fallback to KernelExplainer (slower)
            try:
                f = lambda X: model.predict_proba(pd.DataFrame(X, columns=expected_features))[:, 1]
                explainer = shap.KernelExplainer(f, shap_sample)
                shap_values = explainer.shap_values(shap_sample, nsamples=100)
                user_shap = explainer.shap_values(user_df, nsamples=100)
            except Exception as e:
                st.error(f"Could not compute SHAP values: {e}")
                st.stop()

        shap_values = ensure_array(shap_values)
        user_shap = ensure_array(user_shap)
        progress_bar.progress(50)

        # Show top features for the specific prediction
        if user_shap.ndim > 1:
            user_contrib = user_shap[0]
        else:
            user_contrib = user_shap

        if user_contrib.ndim == 1 and len(user_contrib) == len(expected_features):
            shap_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(shap_importance)[::-1]

            st.markdown("#### Top Factors Driving This Prediction")
            num_top = min(5, len(expected_features))
            for i in range(num_top):
                feat_idx = sorted_idx[i]
                feature = expected_features[feat_idx]
                shap_value = user_contrib[feat_idx]
                direction = "increases" if shap_value > 0 else ("decreases" if shap_value < 0 else "has little impact on")
                input_value = user_df.iloc[0, feat_idx] if feature in user_df.columns else "n/a"
                st.write(f"- {feature} (input: {input_value}) {direction} the no-show probability (SHAP: {shap_value:.3f}).")

            with st.expander("SHAP Summary Plot (Overall Feature Importance)"):
                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 5))
                try:
                    shap.summary_plot(shap_values, shap_sample, feature_names=expected_features, show=False, max_display=10)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not render SHAP summary plot: {e}")

        else:
            st.warning("SHAP values could not be computed in the expected format for this prediction.")

        progress_bar.progress(100)


# =========================================================
# TAB: MODEL PERFORMANCE
# =========================================================
with tab_performance:
    st.subheader("Model Evaluation (Test Set)")

    if not TEST_DATA_PATH.exists():
        st.warning(f"Test data not found at {TEST_DATA_PATH}. Cannot display performance metrics.")
    else:
        try:
            test_df = pd.read_csv(TEST_DATA_PATH)
            if TARGET_COL not in test_df.columns:
                st.warning(f"Target column '{TARGET_COL}' not found in test data.")
            else:
                X_test = test_df.drop(columns=[TARGET_COL]).reindex(columns=expected_features, fill_value=0)
                y_test = test_df[TARGET_COL].astype(int)
                X_test = cast_like_training(X_test, data.drop(columns=[TARGET_COL]))

                if not X_test.empty and not y_test.empty:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= float(threshold)).astype(int)

                    # KPI cards
                    acc = float((y_pred == y_test).mean())
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    roc = roc_auc_score(y_test, y_prob)
                    pr_auc = average_precision_score(y_test, y_prob)

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Accuracy", f"{acc:.1%}")
                    k2.metric("F1 Score", f"{f1:.3f}")
                    k3.metric("Precision", f"{prec:.3f}")
                    k4.metric("Recall", f"{rec:.3f}")
                    st.caption(f"ROC AUC: {roc:.3f}  •  PR AUC: {pr_auc:.3f}")

                    # Classification report
                    st.markdown("#### Classification Report")
                    st.code(classification_report(y_test, y_pred, target_names=["Show", "No-Show"]))

                    # Confusion matrix
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Predicted Show", "Predicted No-Show"],
                        yticklabels=["Actual Show", "Actual No-Show"], ax=ax
                    )
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    st.pyplot(fig)

                    # Curves
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### ROC Curve")
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        fig2, ax2 = plt.subplots()
                        ax2.plot(fpr, tpr, label=f"ROC AUC = {roc:.3f}")
                        ax2.plot([0, 1], [0, 1], "k--")
                        ax2.set_xlabel("False Positive Rate")
                        ax2.set_ylabel("True Positive Rate")
                        ax2.legend()
                        st.pyplot(fig2)
                    with c2:
                        st.markdown("#### Precision-Recall Curve")
                        pr, rc, _ = precision_recall_curve(y_test, y_prob)
                        fig3, ax3 = plt.subplots()
                        ax3.plot(rc, pr, label=f"PR AUC = {pr_auc:.3f}")
                        ax3.set_xlabel("Recall")
                        ax3.set_ylabel("Precision")
                        ax3.legend()
                        st.pyplot(fig3)

                    # Score distribution
                    with st.expander("Score Distribution (Predicted Probabilities)"):
                        fig4, ax4 = plt.subplots()
                        sns.histplot(y_prob, bins=30, kde=True, ax=ax4)
                        ax4.axvline(float(threshold), color='red', linestyle='--', label='Threshold')
                        ax4.set_xlabel('No-Show Probability')
                        ax4.legend()
                        st.pyplot(fig4)

                    # Download predictions
                    pred_df = pd.DataFrame({
                        'y_true': y_test.values,
                        'y_prob': y_prob,
                        'y_pred': y_pred,
                    })
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=pred_df.to_csv(index=False).encode('utf-8'),
                        file_name='predictions_test.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("Test data loaded but appears empty or missing target column after processing.")

        except Exception as e:
            st.error(f"Error evaluating model performance on test set: {e}")


# =========================================================
# TAB: ABOUT
# =========================================================
with tab_about:
    st.subheader("About This Project")
    st.markdown(
        """
        - Goal: Predict whether a patient will show up to their medical appointment.
        - Model: Best performing LightGBM model (saved to models/lightgbm_optimized.pkl).
        - Explainability: SHAP to inspect feature impact on predictions.
        - Threshold: Use the sidebar slider to explore precision/recall trade-offs.
        """
    )
    st.markdown("---")
    st.markdown("Built with Streamlit; see preprocessing/training scripts in the repo.")
