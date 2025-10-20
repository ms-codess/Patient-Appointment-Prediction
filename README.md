# Patient Appointment Prediction

A machine learning project to predict patient no-show appointments using medical center data.

## Project Overview

This project analyzes medical center appointment data to predict whether patients will show up for their scheduled appointments. The dataset contains information about patients' demographics, medical conditions, appointment scheduling, and historical no-show behavior.

## Project Structure

```
Patient-Appointment-Prediction/
├── .env.example                    # Environment configuration template
├── data/
│   ├── raw/
│   │   └── MedicalCentre.csv       # Original raw dataset
│   └── processed/
│       └── cleaned_dataset.csv     # Cleaned and preprocessed data
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── config.py                   # Configuration management
│   ├── preprocess.py               # Data preprocessing and cleaning
│   ├── train.py                    # Model training and evaluation
│   ├── evaluate.py                 # Model evaluation and visualization
│   ├── utils.py                    # Utility functions and helpers
│   └── app_streamlit.py            # Streamlit web application
├── models/
│   └── best_model.joblib          # Trained model (generated)
├── logs/                           # Log files (generated)
├── figures/                        # Generated plots (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## Dataset Description

The dataset contains information about medical appointments in Brazil and includes:

- **Patient Information**: Age, Gender, Neighbourhood
- **Medical Conditions**: Scholarship, Hypertension, Diabetes, Alcoholism, Handicap
- **Appointment Details**: Scheduled Day, Appointment Day, SMS Received
- **Target Variable**: No-show (Yes/No)

## Key Findings

From exploratory data analysis:

- **Overall No-Show Rate**: ~20%
- **Gender Impact**: Minimal difference between male and female patients
- **Age Impact**: Younger patients tend to have higher no-show rates
- **Waiting Time**: Longer waiting periods correlate with higher no-show rates
- **SMS Reminders**: Patients who receive SMS reminders have lower no-show rates

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Patient-Appointment-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment configuration:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your preferred settings (optional)
# The default values will work for most use cases
```

## Configuration

The project uses environment variables for configuration management. All paths and parameters are defined in the `.env` file (or use defaults from `src/config.py`).

### Key Configuration Variables:
- **Data Paths**: `RAW_DATA_PATH`, `PROCESSED_DATA_PATH`
- **Model Paths**: `MODEL_DIR`, `BEST_MODEL_PATH`
- **Model Parameters**: `RANDOM_STATE`, `TEST_SIZE`
- **Hyperparameters**: `RF_N_ESTIMATORS`, `XGB_N_ESTIMATORS`, etc.

The configuration system ensures:
- ✅ Easy deployment across different environments
- ✅ No hardcoded paths in the code
- ✅ Professional configuration management
- ✅ Easy parameter tuning

## Usage

### 1. Data Preprocessing

Run the preprocessing pipeline to clean and prepare the data:

```bash
python src/preprocess.py
```

This will:
- Load raw data from `data/raw/MedicalCentre.csv`
- Clean missing values, duplicates, and outliers
- Create new features (age groups, waiting time groups)
- Encode categorical variables
- Save processed data to `data/processed/clean_medical.csv`

### 2. Model Training

Train and evaluate multiple machine learning models:

```bash
python src/train.py
```

This will:
- Load processed data
- Train multiple models (Naive Bayes, Decision Tree, Random Forest, XGBoost)
- Evaluate model performance
- Save the best performing model to `models/best_model.joblib`

### 3. Exploratory Data Analysis

Open the Jupyter notebook to explore the data:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Model Evaluation

Evaluate the trained model with comprehensive metrics and visualizations:

```bash
python src/evaluate.py
```

This will:
- Load the best trained model
- Generate ROC curves, precision-recall curves, and confusion matrices
- Create feature importance plots
- Save visualizations to the `figures/` directory

### 5. Optional: Web Application

Run the Streamlit web app for interactive predictions:

```bash
streamlit run src/app_streamlit.py
```

This provides:
- Interactive patient information input
- Real-time no-show risk prediction
- Risk level recommendations
- Model performance metrics display

## Models Used

1. **Logistic Regression**: Linear classifier with balanced class weights
2. **Random Forest**: Ensemble of decision trees with balanced subsampling
3. **XGBoost**: Gradient boosting classifier with optimized hyperparameters

All models use:
- One-hot encoding for categorical features
- Balanced class weights to handle imbalanced data
- Cross-validation for robust evaluation

## Performance Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC

## Features

### Original Features
- Age, Gender, Neighbourhood
- Medical conditions (Scholarship, Hypertension, Diabetes, Alcoholism, Handicap)
- SMS received status
- Scheduled and appointment dates

### Engineered Features
- Age groups (categorical bins)
- Waiting time groups (days between scheduling and appointment)
- Date components (year, month, day)

## Data Preprocessing Steps

1. **Data Cleaning**:
   - Remove duplicate records
   - Handle missing values
   - Remove negative ages

2. **Feature Engineering**:
   - Create age groups
   - Calculate waiting time
   - Extract date components
   - Handle outliers using Z-score

3. **Encoding**:
   - Label encode categorical variables
   - One-hot encode for tree-based models

4. **Feature Selection**:
   - Remove highly correlated features
   - Select relevant features for modeling

## Results

The best performing model typically achieves:
- **Accuracy**: ~80-85%
- **F1-Score**: ~0.4-0.5
- **Precision**: ~0.3-0.4
- **Recall**: ~0.6-0.7

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset from Kaggle: [Medical Appointment No Shows](https://www.kaggle.com/joniarroba/noshowappointments)
- University of Ottawa for project support

## Contact

For questions or suggestions, please open an issue in the repository.
