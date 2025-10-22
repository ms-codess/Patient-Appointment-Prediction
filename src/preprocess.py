#Preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

# -------------------------------
# 1. Load and Parse Dates
# -------------------------------
data_path = r"c:\Users\msmirani\Downloads\Patient-Appointment-Prediction\MedicalCentre.csv"
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Convert date columns to datetime
print("Converting dates...")
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

# Calculate waiting time
print("Calculating waiting time...")
df["AwaitingTime"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / (24 * 60 * 60)

# Print column names to verify
print("\nAvailable columns:")
print(df.columns.tolist())

# -------------------------------
# 2. Basic Cleaning
# -------------------------------
print("\nCleaning data...")
df = df.drop_duplicates()
df = df[df["Age"] >= 0]
df.loc[df["Age"] > 115, "Age"] = 115
df["Age"] = df["Age"].fillna(df["Age"].median())

cat_cols = ["Gender", "Neighbourhood", "Scholarship", "Hypertension",
            "Diabetes", "Alcoholism", "Handicap", "SMS_received"]
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -------------------------------
# 3. Enhanced Feature Engineering
# -------------------------------
print("Engineering features...")
# Time-based features
df["ScheduledHour"] = df["ScheduledDay"].dt.hour
df["AppointmentHour"] = df["AppointmentDay"].dt.hour
df["ScheduledDayOfWeek"] = df["ScheduledDay"].dt.dayofweek
df["AppointmentDayOfWeek"] = df["AppointmentDay"].dt.dayofweek
df["ScheduledDayOfMonth"] = df["ScheduledDay"].dt.day
df["IsWeekend"] = df["AppointmentDayOfWeek"].isin([5, 6]).astype(int)
df["IsMonthEnd"] = df["AppointmentDay"].dt.is_month_end.astype(int)
df["IsMonthStart"] = df["AppointmentDay"].dt.is_month_start.astype(int)
df["Appointment_Month"] = df["AppointmentDay"].dt.month
df["Appointment_DayOfYear"] = df["AppointmentDay"].dt.dayofyear

# Health and appointment features
df["Total_Health_Issues"] = df[["Hypertension", "Diabetes", "Alcoholism", "Handicap"]].sum(axis=1)
df["Age_Scholarship"] = df["Age"] * df["Scholarship"].astype(float)
df["SMS_AwaitingTime"] = df["SMS_received"].astype(float) * df["AwaitingTime"]
df["Health_Score"] = df["Total_Health_Issues"]
df["Multiple_Conditions"] = (df["Health_Score"] > 1).astype(int)
df["Same_Day_Appointment"] = (df["AwaitingTime"] == 0).astype(int)
df["Long_Wait"] = (df["AwaitingTime"] > df["AwaitingTime"].median()).astype(int)

# -------------------------------
# 4. Scale Numerical Features
# -------------------------------
print("Scaling numerical features...")
num_features = ["Age", "AwaitingTime", "Total_Health_Issues"]
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# -------------------------------
# 5. Encode Categorical
# -------------------------------
print("Encoding categorical features...")
# Target encoding for Neighbourhood
encoder = TargetEncoder()
df["Neighbourhood_encoded"] = encoder.fit_transform(
    df["Neighbourhood"], df["No-show"].map({"Yes": 1, "No": 0})
)

# Neighbourhood frequency encoding
freq = df["Neighbourhood"].value_counts(normalize=True)
df["Neighbourhood_freq"] = df["Neighbourhood"].map(freq)
df.drop(columns=["Neighbourhood"], inplace=True, errors="ignore")

# Binary encoding
df["Gender"] = df["Gender"].map({"M": 1, "F": 0, "Unknown": 0.5}).astype(float)
binary_cols = ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap", "SMS_received"]
for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# -------------------------------
# 6. Target
# -------------------------------
print("Preparing target variable...")
df["No_show_label"] = df["No-show"].map({"Yes": 1, "No": 0})
df.drop(columns=["No-show", "ScheduledDay", "AppointmentDay"], inplace=True, errors="ignore")

# -------------------------------
# 7. Train / Validation / Test Split
# -------------------------------
print("Splitting data...")
X = df.drop(columns=["No_show_label"])
y = df["No_show_label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\n✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# -------------------------------
# 8. Final Data Check and SMOTE
# -------------------------------
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {X_train.shape}")
print(f"After SMOTE: {X_train_res.shape}")

# -------------------------------
# 9. Save Processed Files
# -------------------------------
print("\nSaving processed files...")
processed_dir = Path(r"c:\Users\msmirani\Downloads\Patient-Appointment-Prediction\data\processed")
processed_dir.mkdir(parents=True, exist_ok=True)

pd.concat([X_train_res, y_train_res], axis=1).to_csv(processed_dir / "train_processed_smote.csv", index=False)
pd.concat([X_val, y_val], axis=1).to_csv(processed_dir / "val_processed.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / "test_processed.csv", index=False)

print("\n✅ Preprocessing completed successfully. Files saved in:", processed_dir)

