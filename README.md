# Patient Appointment No-Show Prediction

This project develops a machine learning pipeline to predict patient no-shows for medical appointments. The objective is to help healthcare providers reduce missed appointments, improve scheduling efficiency, and optimize resource allocation.

Dataset source: Medical Appointment No Shows (Kaggle)
https://www.kaggle.com/datasets/joniarroba/noshowappointments


## Problem Definition

If the model predicts a no-show correctly, the clinic can call or remind the patient, reschedule the slot, and reduce wasted time and cost. If the model misses a no-show (false negative), the clinic loses a time slot. If the model flags someone as a no-show incorrectly (false positive), the clinic might do an unnecessary reminder, which is less costly than a missed appointment. Missing a no-show is therefore worse than predicting a no-show that does not happen.


## Dataset Overview

The dataset contains 110,527 appointment records and 14 variables describing patient demographics, medical history, appointment context, and social factors. Feature categories include demographics (Age, Gender), medical history (Hypertension, Diabetes, Alcoholism, Handicap), appointment context (ScheduledDay, AppointmentDay, SMS_received), social factors (Scholarship), and the target (No-show).


## Model Selection Rationale

Why these models fit this problem
Logistic Regression provides a fast, strong baseline for tabular data and works well with engineered features (waiting time, long wait flags, temporal indicators). It is easy to interpret via coefficients, supports regularization to prevent overfitting, produces calibrated probabilities suitable for threshold tuning, and with class_weight balanced it handles skewed classes reasonably. It sets a transparent benchmark for precision and recall.

Random Forest captures non‑linear relationships and feature interactions that are common in behavioral outcomes like no‑shows (for example, long waits interacting with reminder SMS). It is robust to outliers, requires little preprocessing after encoding, offers stable ranking ability (ROC/PR), and with class_weight balanced_subsample it mitigates class imbalance while remaining straightforward to deploy.

LightGBM is a gradient boosting method that consistently performs well on structured/tabular data. It models complex interactions, trains quickly on large datasets, and supports imbalance handling through scale_pos_weight. It is tolerant of missing values and noisy features, and pairs well with target encoding for higher‑cardinality variables like neighborhood. Because our business objective prioritizes recall and F1, we complement LightGBM with validation‑based threshold tuning to align decisions with operational costs.

Why not deep learning here
This is a classic tabular problem with engineered features and moderate dimensionality. Tree‑based methods and linear baselines typically outperform deep models on such data with less compute, simpler deployment, and better explainability.


## Hyperparameter and Threshold Optimization

LightGBM is tuned with Optuna to search the hyperparameter space efficiently and prune weak trials. Random Forest can be tuned with grid search. After model selection, the decision threshold is tuned on the validation set to maximize F1, aligning decisions with the business goal of minimizing missed appointments.


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

Preprocessing (src/preprocess.py) parses dates, engineers temporal and risk features, handles class imbalance with SMOTE on training data, and prepares train, validation, and test splits.

Training (src/train.py) trains Logistic Regression and Random Forest baselines, tunes LightGBM with Optuna, evaluates on validation, and tunes the decision threshold to improve F1. The best model is saved to models/best_model.pkl and a descriptive variant is also saved.

Evaluation (src/evaluate.py) evaluates the saved best model on the test set, reports F1, Precision, Recall, ROC AUC, PR AUC, Sensitivity, and Specificity, and saves plots. SHAP-based explanations are generated when the shap package is available.


## Setup and Usage

Install dependencies
```
pip install -r src/requirements.txt
```

Run preprocessing
```
python src/preprocess.py
```

Train models and tune threshold
```
python src/train.py
```

Evaluate on the test set
```
python src/evaluate.py
```


## Streamlit Dashboard

An interactive Streamlit app presents the full story: Overview, Data, Preprocess, Models, Tuning, Performance, Explain, Business, Technical, and Demo.

Launch the app
```
streamlit run src/app_streamlit.py
```

The app covers problem framing, class imbalance, preprocessing choices, model selection, interactive threshold tuning, validation and test metrics, and SHAP explanations. For clarity, columns are displayed with friendly names in the UI; Scholarship is labeled as Insurance.

Live app
https://patient-appointment-prediction.streamlit.app/


## Notes on Performance and Explainability

The dataset is imbalanced, so a fixed 0.50 threshold can underperform. Class weighting and scale_pos_weight improve learning, and threshold tuning aligns decisions with the operational objective. SHAP summaries provide global importance and directionality, while local explanations help justify individual predictions.


## References

Kaggle dataset: Medical Appointment No Shows
https://www.kaggle.com/datasets/joniarroba/noshowappointments

LightGBM documentation
https://lightgbm.readthedocs.io

Optuna documentation
https://optuna.org
