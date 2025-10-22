**Patient Appointment No-Show Prediction**
Project Overview
This project aims to predict patient no-shows for medical appointments using machine learning. We've chosen LightGBM with Optuna optimization due to its efficiency with tabular data and ability to handle class imbalance.

**Dataset**
The dataset is from the Medical Appointment No Shows Dataset on Kaggle, containing 110,527 medical appointments and 14 associated variables.

**Key Features**
Patient demographics (Age, Gender)
Medical history (Hypertension, Diabetes, Alcoholism, Handicap)
Appointment context (SMS_received, ScheduledDay, AppointmentDay)
Social factors (Scholarship, Neighbourhood)
Model Selection Rationale
Why LightGBM?
Performance

**Efficient handling of large datasets**
Built-in support for categorical features
Leaf-wise tree growth for better accuracy
Class Imbalance Handling

**Native support for imbalanced datasets**
Scale_pos_weight parameter for class weighting
Compatible with SMOTE preprocessing
Interpretability

Feature importance rankings
Tree visualization capabilities
SHAP value integration
Why Optuna?
Hyperparameter Optimization
Efficient Bayesian optimization
Pruning of unpromising trials
Parallel optimization support
## Current Performance Metrics
F1 Score: 0.4632
Precision: 0.3482
Recall: 0.6918
ROC AUC: 0.7560

## Project Structure
```
Patient-Appointment-Prediction/
│
├── data/
│   ├── processed/
│   │   ├── train_processed_smote.csv
│   │   ├── val_processed.csv
│   │   └── test_processed.csv
│   └── MedicalCentre.csv
│
├── models/
│   └── lightgbm_optimized.pkl
│
├── results/
│   └── metrics.csv
│
└── src/
    ├── preprocess.py
    ├── train.py
    └── evaluate.py
```

## Pipeline Components

### 1. Data Preprocessing (`preprocess.py`)
- Date parsing and feature engineering
- Train/validation/test split
- SMOTE for handling class imbalance
- Standard scaling for numerical features
- Target encoding for categorical variables

### 2. Model Training (`train.py`)
- Framework: LightGBM with Optuna optimization
- Hyperparameter tuning:
  ```python
  {
      'lambda_l1': [1e-8, 10.0],
      'lambda_l2': [1e-8, 10.0],
      'num_leaves': [2, 256],
      'feature_fraction': [0.4, 1.0],
      'bagging_fraction': [0.4, 1.0],
      'bagging_freq': [1, 7],
      'min_child_samples': [5, 100],
      'learning_rate': [0.01, 0.1]
  }
  ```
- 50 optimization trials per run
- Optimization metric: F1 Score

### 3. Model Evaluation (`evaluate.py`)
- Metrics:
  - F1 Score
  - Precision
  - Recall
  - ROC AUC
- Confusion matrix analysis
- Performance logging

## Setup and Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Pipeline
1. Preprocessing:
```bash
python src/preprocess.py
```

2. Training:
```bash
python src/train.py
```

3. Evaluation:
```bash
python src/evaluate.py
```

## Current Limitations
- Model performance needs improvement
- Class imbalance affecting predictions
- Limited feature engineering

