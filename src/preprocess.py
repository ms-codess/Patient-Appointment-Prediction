# src/preprocess.py
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# --- Load environment variables ---
load_dotenv()
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH"))

# --- Load raw data ---
def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)

# --- Clean & Feature Engineer ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID columns if present
    df.drop(columns=["PatientID", "AppointmentID"], inplace=True, errors="ignore")

    # Fill Age missing values early
    if "Age" in df.columns and df["Age"].isnull().any():
        df["Age"].fillna(df["Age"].median(), inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove negative ages
    if "Age" in df.columns:
        df = df[df["Age"] >= 0]

    # --- AgeGroup ---
    if "Age" in df.columns:
        age_bins = [0, 2, 6, 11, 16, 26, 31, 36, 41, 51, 115]
        age_labels = list(range(1, len(age_bins)))
        df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    # --- Dates ---
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AwaitingTime"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.abs()

    # AwaitingTimeGroup
    awaiting_bins = [0, 1, 8, 31, 91, 180]
    awaiting_labels = list(range(1, len(awaiting_bins)))
    df["AwaitingTimeGroup"] = pd.cut(df["AwaitingTime"], bins=awaiting_bins, labels=awaiting_labels, right=False)

    # Extract date components
    for col in ["AppointmentDay", "ScheduledDay"]:
        prefix = col[:4]
        df[f"{prefix}_Year"] = df[col].dt.year
        df[f"{prefix}_Month"] = df[col].dt.month
        df[f"{prefix}_Day"] = df[col].dt.day

    df.drop(columns=["AppointmentDay", "ScheduledDay"], inplace=True)

    # Encode categorical variables
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype("category").cat.codes
    if "No-show" in df.columns:
        df["No-show"] = df["No-show"].astype("category").cat.codes
    if "Neighbourhood" in df.columns:
        df["Neighbourhood"] = df["Neighbourhood"].astype("category").cat.codes

    # Drop raw Age and AwaitingTime after creating groups
    df.drop(columns=["Age", "AwaitingTime"], inplace=True, errors="ignore")

    # --- Handle any remaining missing values globally ---
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':  # numeric columns
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        else:
            if df[col].mode().shape[0] > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# --- Save processed data ---
def save_data(df: pd.DataFrame):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

# --- Pipeline ---
def preprocess_pipeline():
    df = load_data()
    df = clean_data(df)
    save_data(df)
    return df

if __name__ == "__main__":
    preprocess_pipeline()
