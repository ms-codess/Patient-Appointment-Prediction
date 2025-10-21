
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
TARGET_COL = "No_show_label"
MODEL_DIR = Path("models")
# Dynamically find the best model saved with the '_f1.pkl' suffix
try:
    MODEL_PATH = list(MODEL_DIR.glob("*_model_f1.pkl"))[0]
except IndexError:
     st.error("❌ Trained model artifact not found. Please run the training pipeline to create a model file with '_model_f1.pkl' suffix.")
     st.stop()

THRESHOLD_PATH = MODEL_DIR / "best_threshold_f1.txt"
DATA_PATH = Path("data/processed/train_processed_smote.csv") # Use training data for background/feature names

def detect_target_col(df: pd.DataFrame) -> str:
    if TARGET_COL in df.columns:
        return TARGET_COL
    return ""

def drop_target_col(cols):
    return [c for c in cols if c != TARGET_COL]

# Adjusted to handle potential XGBoost config structure
def clean_xgb_config_string(cfg: str) -> str:
    # Example: "base_score": "[0.5]" -> "base_score": 0.5
    cfg = re.sub(r'"\[([0-9eE+\-\.]+)\]"', r'\1', cfg)
    # Handle cases where quotes around the base score were removed but not others
    cfg = re.sub(r'"base_score"\s*:\s*([^,}\]]+)', r'"base_score":\1', cfg)
    return cfg


@st.cache_resource
def load_model_and_data():
    try:
        bundle = joblib.load(MODEL_PATH)
        # Assuming the model is the object itself, not within a dict unless saved explicitly that way
        model = bundle

        df = pd.read_csv(DATA_PATH)
        target_col = detect_target_col(df)

        # Features should be all columns except the target
        feature_cols = drop_target_col(df.columns.tolist())

        booster = None
        # Attempt to get booster and fix config for SHAP
        try:
            booster = model.get_booster()
            cfg = booster.save_config()
            cfg = clean_xgb_config_string(cfg)
            booster.load_config(cfg)
             # Ensure feature names are set on the booster
            booster.feature_names = feature_cols

        except Exception as e:
            st.warning(f"Could not load XGBoost booster with feature names directly. SHAP might fall back to KernelExplainer or use default names if available. Error: {e}")
            booster = None # Ensure booster is None if setting feature_names fails


    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()

    return model, booster, df, feature_cols, target_col

@st.cache_resource
def get_expected_features(_model, feature_cols):
    """Try to get feature names from the model first, fall back to data columns."""
    try:
        # Try getting feature names from the model's booster if available
        if hasattr(_model, 'get_booster'):
             booster_features = _model.get_booster().feature_names
             if booster_features is not None and len(booster_features) > 0:
                 # Ensure they match the data columns, filter out missing ones
                 return [f for f in booster_features if f in feature_cols]
        # Fallback to data columns if model features are not available or usable
        return feature_cols
    except Exception:
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
    st.error(f"❌ Target column '{TARGET_COL}' not found in dataset.")
    st.stop()


expected_features = get_expected_features(model, feature_cols)
expected_features = [c for c in expected_features if c != TARGET_COL] # Ensure target is not in features
background_X = data.reindex(columns=expected_features, fill_value=0)
feature_index_map = {feat: idx for idx, feat in enumerate(expected_features)}

# Load the best threshold from the file
try:
    with open(THRESHOLD_PATH, "r") as f:
        best_threshold = float(f.read().strip())
except FileNotFoundError:
     st.error(f"❌ Threshold file not found at {THRESHOLD_PATH}. Please run the training pipeline.")
     st.stop()
except Exception as e:
     st.error(f"Error loading threshold: {e}")
     st.stop()


with st.sidebar:
    st.header("📂 Dataset Overview")
    st.metric("Records", f"{len(data):,}")
    if target_col in data.columns:
        numeric_target = pd.to_numeric(data[target_col], errors="coerce")
        if not numeric_target.dropna().empty:
            no_show_rate = numeric_target.mean()
            st.metric(f"{target_col.replace('_label', '')} Rate", f"{no_show_rate:.1%}") # Display target name nicely

    st.caption("Preview of processed features")
    preview_cols = [c for c in expected_features[:6] if c in data.columns]
    if target_col and target_col in data.columns:
        preview_cols.append(target_col)
    if preview_cols:
        st.dataframe(data[preview_cols].head(), use_container_width=True)
    else:
        st.info("No feature columns available for preview.")

    st.markdown("---")
    st.header("⚙️ Model Info")
    st.write(f"**Model Type:** {type(model).__name__}")
    st.write(f"**Optimized Metric:** F1 Score")
    st.write(f"**Best Threshold:** {best_threshold:.3f}")
    st.write(f"**Model File:** {MODEL_PATH.name}")


def ensure_array(values):
    """SHAP sometimes returns a list (multi-class). Convert to a 2D numpy array."""
    if isinstance(values, list):
        # For binary classification with TreeExplainer, it returns a list of two arrays [shap_values_class_0, shap_values_class_1]
        # We are interested in the shap values for the positive class (index 1)
        if len(values) == 2:
             return np.array(values[1])
        # Handle potential other list formats, though less common for TreeExplainer binary output
        return np.array(values[0] if values else []) # Fallback

    return np.array(values)

# =========================================================
# MAPPINGS
# =========================================================
# Ensure these keys match the processed feature names after encoding
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

# Map display names to the column names in the processed data
# Ensure keys match the column names after preprocessing (e.g., "Gender", "Scholarship")
CATEGORICAL_MAPPINGS = {
    "Gender": {"Woman": 0, "Man": 1},
    "SMS_received": {"No": 0, "Yes": 1},
    "Hypertension": {"No": 0, "Yes": 1},
    "Diabetes": {"No": 0, "Yes": 1},
    "Alcoholism": {"No": 0, "Yes": 1},
    "Handicap": {"No": 0, "Yes": 1},
    "Scholarship": {"No": 0, "Yes": 1}, # Match processed column name
}

# Map processed column names to user-friendly display names
DISPLAY_RENAME = {
    "Scholarship": "Medical Coverage", # Match processed column name to display name
    "AwaitingTimeGroup": "Waiting Time Group",
    "Neighbourhood_freq": "Neighbourhood Frequency",
    "Appo_Year": "Appointment Year",
    "Appo_Month": "Appointment Month",
    "Appo_Day": "Appointment Day",
    "Appo_DOW": "Appointment Day of Week",
    "Appo_Weekend": "Appointment Weekend",
    "Sche_Year": "Scheduled Year",
    "Sche_Month": "Scheduled Month",
    "Sche_Day": "Scheduled Day",
    "Sche_DOW": "Scheduled Day of Week",
    "Sche_Weekend": "Scheduled Weekend",
}

# Ensure these explanations use the processed column names as keys
FEATURE_EXPLANATIONS = {
    "Scholarship": "Medical Coverage — patient has government health insurance (Yes/No).",
    "Gender": "Gender of the patient.",
    "SMS_received": "Whether the patient received an SMS reminder.",
    "Hypertension": "Patient has hypertension (Yes/No).",
    "Diabetes": "Patient has diabetes (Yes/No).",
    "Alcoholism": "Patient has a history of alcoholism (Yes/No).",
    "Handicap": "Patient has a physical handicap (Yes/No).",
    "AgeGroup": "Age group of the patient (e.g., 16–25 years old).",
    "AwaitingTimeGroup": "How long between scheduling and appointment (grouped).",
    "AwaitingTime": "Number of days between scheduling and appointment.",
    "Neighbourhood_freq": "Frequency of the patient's neighbourhood in the dataset.",
    "Appo_Year": "Year of the appointment.",
    "Appo_Month": "Month of the appointment.",
    "Appo_Day": "Day of the appointment.",
    "Appo_DOW": "Day of the week of the appointment (0=Monday, 6=Sunday).",
    "Appo_Weekend": "Whether the appointment is on a weekend (1=Yes, 0=No).",
    "Sche_Year": "Year the appointment was scheduled.",
    "Sche_Month": "Month the appointment was scheduled.",
    "Sche_Day": "Day the appointment was scheduled.",
    "Sche_DOW": "Day of the week the appointment was scheduled (0=Monday, 6=Sunday).",
    "Sche_Weekend": "Whether the appointment was scheduled on a weekend (1=Yes, 0=No).",
    "PatientID": "Unique identifier for the patient (included as a feature).",
    "AppointmentID": "Unique identifier for the appointment (included as a feature).",
    "Age": "Age of the patient.",

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
        col1, col2, col3 = st.columns(3) # Use more columns for better layout

        # Collect inputs for features expected by the model
        for col in expected_features:
            display_name = DISPLAY_RENAME.get(col, col)
            help_text = FEATURE_EXPLANATIONS.get(col, f"Input for {display_name}")

            if col in CATEGORICAL_MAPPINGS:
                 # Use selectbox for known categorical features
                 selection = col1.selectbox(
                     f"{display_name} ❓",
                     list(CATEGORICAL_MAPPINGS[col].keys()),
                     help=help_text,
                     key=f"input_{col}"
                 )
                 user_inputs[col] = CATEGORICAL_MAPPINGS[col][selection]

            elif col == "AgeGroup" and col in AGE_GROUP_LABELS.values(): # Handle AgeGroup as a selectbox
                 age_label = col2.selectbox(
                     "Age Group",
                     list(AGE_GROUP_LABELS.keys()),
                     help=help_text,
                     key="input_age"
                 )
                 user_inputs[col] = AGE_GROUP_LABELS[age_label]

            elif col == "AwaitingTimeGroup" and col in AWAITING_TIME_LABELS.values(): # Handle AwaitingTimeGroup as a selectbox
                 wait_label = col2.selectbox(
                     "Waiting Time Group",
                     list(AWAITING_TIME_LABELS.keys()),
                     help=help_text,
                     key="input_wait"
                 )
                 user_inputs[col] = AWAITING_TIME_LABELS[wait_label]

            elif pd.api.types.is_numeric_dtype(data[col]):
                # Use number_input for numeric columns
                 if col in ['PatientID', 'AppointmentID']:
                     # Handle large IDs as integers, maybe not ideal as features but included if they are
                     user_inputs[col] = col3.number_input(
                         f"{display_name} ❓",
                         min_value=0, # Or appropriate min value
                         help=help_text,
                         key=f"input_{col}",
                         value= int(data[col].median()) if col in data.columns else 0
                     )
                 else:
                    min_val = float(data[col].min()) if col in data.columns else 0.0
                    max_val = float(data[col].max()) if col in data.columns else 100.0
                    median_val = float(data[col].median()) if col in data.columns else 0.0
                    user_inputs[col] = col3.number_input(
                        f"{display_name} ❓",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        help=help_text,
                        key=f"input_{col}"
                    )
            # Add more input types (e.g., text_input for strings) if needed for other features

        # Ensure all expected features are in user_inputs, fill missing with median/mode from training data
        # This handles features that aren't explicitly added as Streamlit widgets
        for col in expected_features:
             if col not in user_inputs:
                 if col in data.columns:
                     if pd.api.types.is_numeric_dtype(data[col]):
                         user_inputs[col] = data[col].median()
                     else:
                         user_inputs[col] = data[col].mode()[0] # Use mode for non-numeric
                 else:
                     user_inputs[col] = 0 # Default if column not found in data (shouldn't happen if data loaded correctly)


        # Create DataFrame in the correct order
        user_df = pd.DataFrame([user_inputs]).reindex(columns=expected_features, fill_value=0)


        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

    if submitted:
        # Ensure input types match the training data
        for col in expected_features:
            if col in data.columns:
                user_df[col] = user_df[col].astype(data[col].dtype)


        prob = float(model.predict_proba(user_df)[0, 1])
        # Use the best threshold for classification
        predicted_class = "❌ No-Show" if prob >= best_threshold else "✅ Will Attend"

        risk_level = (
            ("🔴 High", "red") if prob >= best_threshold + (1 - best_threshold) * 0.2 else # Example: higher than threshold + 20% of remaining range
            ("🟡 Medium", "orange") if prob >= best_threshold - (best_threshold - 0) * 0.2 else # Example: higher than threshold - 20% of range below threshold
            ("🟢 Low", "green")
        )

        st.session_state.prediction = {
            "prob": prob,
            "predicted_class": predicted_class,
            "risk_level": risk_level,
        }
        st.session_state.user_input_df = user_df # Store the dataframe used for prediction


    if st.session_state.prediction:
        pred = st.session_state.prediction
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", pred["predicted_class"])
        c2.metric("No-Show Probability", f"{pred['prob']:.1%}")
        c3.metric("Risk Level", pred["risk_level"][0])
        st.success(
            f"✅ Based on the selected information, the predicted risk of no-show is "
            f"{pred['risk_level'][0]}."
        )

        with st.expander("📋 Patient inputs used for this prediction"):
            display_df = st.session_state.user_input_df.copy()
            # Rename columns for display
            display_df.columns = [DISPLAY_RENAME.get(col, col) for col in display_df.columns]
            st.dataframe(display_df)

# =========================================================
# TAB 2 — EXPLAINABILITY
# =========================================================
with tab_explain:
    st.subheader("🧠 SHAP Explainability")

    if not st.session_state.get("prediction"):
        st.info("Run a prediction from the 'Prediction' tab to unlock SHAP explanations.")
    else:
        user_df = st.session_state.get("user_input_df")
        if user_df is None or background_X.empty:
            st.warning("Cannot compute SHAP values without the patient's input or background data.")
            if user_df is None:
                 st.info("Please re-run a prediction to capture the latest patient inputs.")
            if background_X.empty:
                 st.error("Background data is empty. Check data loading.")
            st.stop()

        st.info("Calculating SHAP values... This might take a moment depending on the sample size.")
        # Progress bar
        progress_bar = st.progress(0)

        explainer = None
        shap_sample_size = min(100, len(background_X)) # Use up to 100 samples for background
        shap_sample = background_X.sample(shap_sample_size, random_state=42)

        try:
            # Use TreeExplainer if the model is tree-based (like XGBoost) and booster is available
            if booster:
                explainer = shap.TreeExplainer(booster)
                # Compute SHAP values for the background data (for summary plot)
                shap_values = explainer.shap_values(shap_sample)
                 # Compute SHAP values for the single user instance
                user_shap = explainer.shap_values(user_df)
                st.success("✅ Using TreeExplainer for SHAP.")
            else:
                 # Fallback to KernelExplainer for other models or if booster not available
                 # KernelExplainer requires background data and is slower
                 st.warning("TreeExplainer not available or failed. Falling back to KernelExplainer. This may take longer.")
                 explainer = shap.KernelExplainer(model.predict_proba, shap_sample)
                 shap_values = explainer.shap_values(shap_sample, nsamples=100) # nsamples for KernelExplainer
                 user_shap = explainer.shap_values(user_df, nsamples=100)
                 st.success("✅ Using KernelExplainer for SHAP.")

        except Exception as e:
            st.error(f"❌ SHAP failed to compute values: {e}")
            explainer = None


        if explainer is not None:
            # Ensure SHAP values are in a usable format (numpy array)
            shap_values = ensure_array(shap_values)
            # Ensure user_shap is in a usable format and select the values for the positive class
            user_shap = ensure_array(user_shap)

            # Handle case where user_shap might be 2D (e.g., for multiclass, though we expect binary)
            user_contrib = user_shap[0] if user_shap.ndim > 1 else user_shap


            if user_contrib.ndim == 1 and len(user_contrib) == len(expected_features):
                shap_importance = np.abs(shap_values).mean(axis=0)
                # Get indices of features sorted by importance (descending)
                sorted_feature_indices = np.argsort(shap_importance)[::-1]

                st.markdown("#### ✨ Top Factors Driving This Specific Prediction:")

                # Display top contributing features for the specific prediction
                num_top_features = min(5, len(expected_features))
                for i in range(num_top_features):
                    feat_idx = sorted_feature_indices[i]
                    feature = expected_features[feat_idx]
                    shap_value = user_contrib[feat_idx]
                    display_name = DISPLAY_RENAME.get(feature, feature)

                    # Determine if the feature's value increases or decreases the no-show probability
                    # Positive SHAP value increases prediction (closer to 1 for No-Show)
                    # Negative SHAP value decreases prediction (closer to 0 for Show)
                    direction_icon = "🔺" if shap_value > 0 else ("🔽" if shap_value < 0 else "➖")
                    direction_text = "increases" if shap_value > 0 else ("decreases" if shap_value < 0 else "has no significant impact on")


                    # Get the actual input value for this feature from the user_df
                    input_value = user_df[feature].iloc[0]

                    st.write(f"- **{display_name}** (Input value: `{input_value}`) {direction_icon} {direction_text} the predicted no-show probability (SHAP value: `{shap_value:.3f}`).")


                with st.expander("🔬 SHAP Summary Plot (Overall Feature Importance):"):
                    # Clear previous plots
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Ensure feature names match the shap_values columns
                    shap.summary_plot(
                        shap_values,
                        shap_sample, # Use the background sample data
                        feature_names=[DISPLAY_RENAME.get(f, f) for f in expected_features],
                        show=False,
                         max_display=10 # Show top 10 features
                    )
                    st.pyplot(fig)

                with st.expander("🌊 SHAP Force Plot (Detailed Single Prediction Explanation):"):
                     st.info("The force plot shows how each feature contributes to pushing the prediction from the base value (average prediction) to the final output value.")
                     # Clear previous plots
                     plt.clf()
                     # Ensure base_value and expected_value are handled correctly
                     # For TreeExplainer, base_value is often explainer.expected_value
                     expected_value = explainer.expected_value
                     if isinstance(expected_value, list) and len(expected_value) == 2:
                          # For binary classification, we usually care about the positive class (index 1)
                          expected_value = expected_value[1]
                     elif isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                           expected_value = expected_value.flatten()[0] # Take the first element

                     # Ensure user_shap is 2D for force_plot if needed, and take the positive class values
                     user_shap_for_plot = user_shap
                     if user_shap_for_plot.ndim == 1:
                          user_shap_for_plot = user_shap_for_plot.reshape(1, -1) # Reshape to 1 sample, N features
                     if user_shap_for_plot.shape[0] > 1: # If it's multi-output, select the positive class
                           user_shap_for_plot = user_shap_for_plot[1, :].reshape(1, -1)

                     # Ensure user_df has the correct structure (1 row, N features)
                     user_df_for_plot = user_df.head(1)

                     try:
                        # Use the JS visualization directly in Streamlit
                        shap.initjs()
                        # Ensure feature names match
                        feature_names_for_plot = [DISPLAY_RENAME.get(f, f) for f in expected_features]
                        # The force plot expects shap_values, base_value, and feature values
                        st_shap = shap.force_plot(
                            expected_value,
                            user_shap_for_plot,
                            user_df_for_plot,
                            feature_names=feature_names_for_plot,
                            matplotlib=False # Use JS visualization
                        )
                        st.components.v1.html(st_shap.html(), width=1000, height=300) # Embed the HTML

                     except Exception as e:
                         st.warning(f"Could not generate SHAP force plot: {e}")
                         st.info("This might happen with certain model types or SHAP versions in Streamlit.")


            else:
                 st.warning("SHAP values could not be computed in the expected format for this prediction.")

        progress_bar.progress(100) # Complete the progress bar


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================
with tab_performance:
    st.subheader("📈 Model Evaluation")

    # Load test data for performance metrics
    TEST_DATA_PATH = Path("data/processed/test_processed.csv")
    if not TEST_DATA_PATH.exists():
         st.warning(f"Test data not found at {TEST_DATA_PATH}. Cannot display performance metrics.")
    else:
        try:
            test_df_eval = pd.read_csv(TEST_DATA_PATH)
            if TARGET_COL not in test_df_eval.columns:
                 st.warning(f"Target column '{TARGET_COL}' not found in test data.")
            else:
                X_test_eval = test_df_eval.drop(columns=[TARGET_COL]).reindex(columns=expected_features, fill_value=0)
                y_test_eval = test_df_eval[TARGET_COL].astype(int)

                # Ensure test data columns match training data columns and order
                X_test_eval = X_test_eval.reindex(columns=expected_features, fill_value=0)
                for col in expected_features:
                     if col in data.columns and col in X_test_eval.columns:
                         X_test_eval[col] = X_test_eval[col].astype(data[col].dtype)
                     elif col not in X_test_eval.columns:
                          # Add missing columns with a default value (e.g., 0 or median/mode)
                          if col in data.columns:
                             if pd.api.types.is_numeric_dtype(data[col]):
                                 X_test_eval[col] = data[col].median()
                             else:
                                 X_test_eval[col] = data[col].mode()[0]
                             X_test_eval[col] = X_test_eval[col].astype(data[col].dtype)
                          else:
                             X_test_eval[col] = 0 # Fallback


                if not X_test_eval.empty and not y_test_eval.empty:
                    y_prob_test = model.predict_proba(X_test_eval)[:, 1]
                    # Use the best threshold for predictions on the test set
                    y_pred_test = (y_prob_test >= best_threshold).astype(int)


                    st.text("Classification Report (Test Set) - using F1-Optimized Threshold:")
                    # Ensure target names are correct
                    st.code(classification_report(y_test_eval, y_pred_test, target_names=["Show", "No-Show"]))

                    st.text("Confusion Matrix (Test Set):")
                    cm = confusion_matrix(y_test_eval, y_pred_test)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                xticklabels=["Predicted Show", "Predicted No-Show"],
                                yticklabels=["Actual Show", "Actual No-Show"], ax=ax)
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    st.pyplot(fig)

                else:
                     st.warning("Test data loaded but appears empty or missing target column after processing.")

        except Exception as e:
             st.error(f"Error evaluating model performance on test set: {e}")


# =========================================================
# TAB 4 — ABOUT
# =========================================================
with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    - **Goal:** Predict whether a patient will **show up** to their medical appointment.
    - **Model:** The best performing model trained in the notebook (optimized for F1 score on the validation set).
    - **Explainability:** SHAP (TreeExplainer or KernelExplainer) provides insight into feature importance for individual predictions and overall model behavior.
    - **Medical Coverage:** Refers to whether the patient has **government health insurance**.
    - **Age Group & Waiting Time Group:** Features derived during the data preprocessing step.
    - **Optimization Metric:** The model's threshold was selected to maximize the **F1 score** on the validation set, aiming for a balance between precision and recall.
    """)
    st.markdown("---")
    st.markdown("This dashboard is built using Streamlit and leverages the scikit-learn and SHAP libraries for model prediction and explainability.")
    st.markdown("Find the source code and data preprocessing/training steps in the accompanying Google Colab notebook.")
