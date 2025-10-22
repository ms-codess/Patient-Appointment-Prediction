# 🩺 Patient Appointment No-Show Prediction

This project develops a machine learning pipeline to **predict patient no-shows** for medical appointments.  
The objective is to support healthcare providers in **reducing missed appointments**, improving scheduling efficiency, and optimizing resource allocation.

> 📊 **Dataset Source**: [Medical Appointment No Shows — Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

---

## 📁 Dataset Overview

The dataset contains **110,527 appointment records** and **14 variables** describing patient demographics, medical history, appointment context, and social factors.

### Feature Categories

- **Patient Demographics**: `Age`, `Gender`  
- **Medical History**: `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`  
- **Appointment Context**: `ScheduledDay`, `AppointmentDay`, `SMS_received`  
- **Social Factors**: `Scholarship`, `Neighbourhood`  
- **Target Variable**: `No-show` (Yes/No)

---

## 🧠 Model Selection Rationale

### Why [LightGBM](https://lightgbm.readthedocs.io/)?

LightGBM was chosen for its **high performance on structured/tabular data** and its ability to efficiently handle **imbalanced classification problems**.

#### ⚡ Performance
- Fast and memory-efficient gradient boosting
- Native categorical feature handling
- Leaf-wise tree growth strategy → improved accuracy over depth-wise growth

#### ⚖️ Class Imbalance Handling
- Built-in `scale_pos_weight` parameter for weighting minority class
- Fully compatible with SMOTE preprocessing
- Supports early stopping and regularization

#### 🧭 Interpretability
- Feature importance ranking for explainability
- Tree visualization capabilities
- SHAP integration for model interpretability

---

## 🔍 Hyperparameter Optimization with [Optuna](https://optuna.org/)

Optuna was used for **Bayesian hyperparameter optimization**, enabling:

- **Efficient search** of hyperparameter space  
- **Early pruning** of unpromising trials  
- **Parallel execution** for faster optimization

### Optimization Space Example

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

- 🧪 **Optimization Metric:** F1 Score  
- 🔁 **Trials per run:** 50  
- ⏳ **Iterations:** Multiple runs with different seeds to improve stability and avoid overfitting


---

## 🧪 Project Structure

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

---

## 🧭 ML Pipeline Components

### 1. Preprocessing (`preprocess.py`)
- Parse dates and create temporal features
- Feature engineering (`lead_time`, `weekday`, `is_weekend`)
- Handle class imbalance with SMOTE
- Train/validation/test splitting
- Scaling numerical features and encoding categorical variables

### 2. Model Training (`train.py`)
- LightGBM model with Optuna optimization
- Cross-validation for robust scoring
- Model persisted as `.pkl`

### 3. Evaluation (`evaluate.py`)
- Metrics: F1, Precision, Recall, ROC AUC
- Confusion matrix visualization
- SHAP feature importance plots
- Experiment logs saved to `results/metrics.csv`

---

## ⚙️ Setup & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Preprocess data
python src/preprocess.py

# Train model
python src/train.py

# Evaluate results
python src/evaluate.py
```

---

## 🚧 Current Limitations & Next Steps

- **Class Imbalance:** Recall can be further improved using advanced sampling or focal loss  
- **Feature Engineering:** Temporal and behavioral variables can be extended  
- **Model Generalization:** Consider model calibration and ensembling

### Planned Enhancements
- Evaluate [CatBoost](https://catboost.ai/) and [XGBoost](https://xgboost.readthedocs.io/)  
- Introduce cost-sensitive evaluation  
- Deploy real-time prediction API

---

## 🏷️ References

- [Kaggle Dataset: Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
- [Optuna Documentation](https://optuna.org/)
