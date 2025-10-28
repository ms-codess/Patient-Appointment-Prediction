
# Patient Appointment No-Show Prediction

This project builds a **machine learning pipeline** to predict **patient no-shows** for medical appointments.  
The goal is to help healthcare providers **reduce missed appointments**, **optimize scheduling**, and **improve resource utilization**.

**Dataset:** [Medical Appointment No Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
**Live Demo:** [Streamlit App](https://patient-appointment-prediction.streamlit.app/)

## Problem Definition

True No-Show Prediction allows clinics to remind or reschedule, saving time and costs.  
False Negative wastes an appointment slot, which is the most costly scenario.  
False Positive leads to an unnecessary reminder, which is low cost.

**Business Priority:** Minimize missed appointments (high Recall, good F1), accepting some extra reminders.

## Dataset Overview

**Total Records:** 110,527  
**Features:** 14

Feature Categories  
• Demographics: Age, Gender  
• Medical History: Hypertension, Diabetes, Alcoholism, Handicap  
• Appointment Context: ScheduledDay, AppointmentDay, SMS_received  
• Social Factors: Scholarship  
• Target: No-show (Yes/No)

## Model Selection Rationale

| Model | Why It's Used | Strengths |
|-------|---------------|-----------|
| Logistic Regression | Baseline | Fast, interpretable, calibrated probabilities, class weighting |
| Random Forest | Capture non-linear interactions | Robust to outliers, minimal preprocessing, stable ranking |
| LightGBM | Final model | Excellent for tabular data, fast training, imbalance handling, strong performance |

We focus on **Recall and F1** to reflect operational costs of missed appointments.

## Hyperparameter & Threshold Optimization

LightGBM is tuned with Optuna for efficient hyperparameter search.  
Random Forest can be tuned with grid search.  
Threshold tuning optimizes the decision boundary to maximize F1 and recall rather than relying on the default 0.50.

## Project Structure

```
Patient-Appointment-Prediction/
├── models/
├── results/
├── src/
│   ├── data/processed/
│   │   ├── train_processed_smote.csv
│   │   ├── val_processed.csv
│   │   └── test_processed.csv
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
└── MedicalCentre.csv
```

## ML Pipeline Components

### 1. Preprocessing
• Date parsing and temporal feature engineering  
• Handling imbalance with SMOTE  
• Splitting into train, validation, and test sets

### 2. Model Training
• Train Logistic Regression and Random Forest baselines  
• Tune LightGBM with Optuna  
• Optimize threshold on validation set  
• Save best model to models/best_model.pkl

### 3. Evaluation and Explainability
• Metrics: F1, Precision, Recall, ROC AUC, PR AUC, Sensitivity, Specificity  
• SHAP explanations for feature importance and local interpretability

## Setup and Usage

```bash
pip install -r src/requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Streamlit Dashboard

An interactive Streamlit dashboard presents the full ML workflow:

• Problem framing and class imbalance  
• Preprocessing steps  
• Model selection and tuning  
• Threshold optimization  
• Test performance metrics  
• SHAP-based feature explanations

```bash
streamlit run src/app_streamlit.py
```

**Live App:** [https://patient-appointment-prediction.streamlit.app/](https://patient-appointment-prediction.streamlit.app/)

## Performance and Explainability

• Imbalanced data addressed with class weighting and scale_pos_weight  
• Threshold tuning improves recall and aligns with clinic objectives  
• SHAP global and local explanations enhance transparency and trust

## References

• [Kaggle Dataset: Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
• [LightGBM Documentation](https://lightgbm.readthedocs.io)  
• [Optuna Documentation](https://optuna.org/)

## Key Takeaway

A carefully tuned LightGBM model with threshold optimization and explainability provides a practical solution to reduce no-shows and support data-driven healthcare scheduling.
