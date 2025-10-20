"""
Configuration module for Patient Appointment Prediction project.

This module loads environment variables and provides centralized configuration
for all paths and parameters used throughout the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ==== Data Paths ====
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/MedicalCentre.csv"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed/cleaned_dataset.csv"))

# ==== Model Paths ====
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
BEST_MODEL_PATH = Path(os.getenv("BEST_MODEL_PATH", "models/best_model.joblib"))

# ==== Logs / Outputs ====
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", "figures"))

# ==== Model Parameters ====
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.25"))
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.2"))

# ==== Feature Engineering Parameters ====
AGE_OUTLIER_THRESHOLD = float(os.getenv("AGE_OUTLIER_THRESHOLD", "3"))
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))

# ==== Model Hyperparameters ====
# Random Forest
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "200"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "10"))

# XGBoost
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", "500"))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "5"))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.05"))

# Logistic Regression
LOGREG_MAX_ITER = int(os.getenv("LOGREG_MAX_ITER", "300"))

# ==== Utility Functions ====
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [MODEL_DIR, LOGS_DIR, FIGURES_DIR, 
                  PROCESSED_DATA_PATH.parent, RAW_DATA_PATH.parent]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ All directories created/verified")

def get_config_summary():
    """Return a summary of current configuration."""
    return {
        "raw_data_path": str(RAW_DATA_PATH),
        "processed_data_path": str(PROCESSED_DATA_PATH),
        "model_dir": str(MODEL_DIR),
        "best_model_path": str(BEST_MODEL_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "age_outlier_threshold": AGE_OUTLIER_THRESHOLD,
        "correlation_threshold": CORRELATION_THRESHOLD
    }

if __name__ == "__main__":
    # Test configuration loading
    print("Configuration Summary:")
    print("=" * 40)
    config = get_config_summary()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\nEnsuring directories exist...")
    ensure_directories()
