#Preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load and Parse Dates
# -------------------------------
df = pd.read_csv("/content/MedicalCentre.csv") #Change with your path to the dataset
for col in ["ScheduledDay", "AppointmentDay"]:
    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)

# -------------------------------
# 2. Basic Cleaning
# -------------------------------
df = df.drop_duplicates()
df = df[df["Age"] >= 0]
df.loc[df["Age"] > 115, "Age"] = 115
df["Age"] = df["Age"].fillna(df["Age"].median())

cat_cols = ["Gender", "Neighbourhood", "Scholarship", "Hypertension",
            "Diabetes", "Alcoholism", "Handicap", "SMS_received"]
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -------------------------------
# 3. Feature Engineering
# -------------------------------
df["AwaitingTime"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(lower=0)

# Awaiting Time Groups
await_bins = [0, 1, 8, 31, 91, 180]
await_labels = [1, 2, 3, 4, 5]
df["AwaitingTimeGroup"] = pd.cut(df["AwaitingTime"], bins=await_bins, labels=await_labels, right=False)

# Age Groups
age_bins = [0, 2, 6, 11, 16, 26, 31, 36, 41, 51, 115]
age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

# Calendar Features
for col, prefix in [("AppointmentDay", "Appo"), ("ScheduledDay", "Sche")]:
    dt = df[col]
    df[f"{prefix}_Year"] = dt.dt.year
    df[f"{prefix}_Month"] = dt.dt.month
    df[f"{prefix}_Day"] = dt.dt.day
    df[f"{prefix}_DOW"] = dt.dt.dayofweek
    df[f"{prefix}_Weekend"] = (dt.dt.dayofweek >= 5).astype(int)

# -------------------------------
# 4. Encode Categorical
# -------------------------------
# Neighbourhood frequency encoding
freq = df["Neighbourhood"].value_counts(normalize=True)
df["Neighbourhood_freq"] = df["Neighbourhood"].map(freq).fillna(0)
df.drop(columns=["Neighbourhood"], inplace=True, errors="ignore")

# Binary encoding
df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)
binary_cols = ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap", "SMS_received"]
for col in binary_cols:
    df[col] = df[col].map({1: 1, 0: 0, "Yes": 1, "No": 0}).fillna(0).astype(int)

# -------------------------------
# 5. Target
# -------------------------------
df["No_show_label"] = df["No-show"].map({"Yes": 1, "No": 0})
df.drop(columns=["No-show", "ScheduledDay", "AppointmentDay"], inplace=True)

# -------------------------------
# 6. Train / Validation / Test Split
# -------------------------------
X = df.drop(columns=["No_show_label"])
y = df["No_show_label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# -------------------------------
# 7. 🧼 Impute Missing Before SMOTE
# -------------------------------
num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_val[col] = X_val[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

for col in cat_cols:
    mode_val = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_val)
    X_val[col] = X_val[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val)

print("\n✅ NaNs before SMOTE (should be 0):")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

# -------------------------------
# 8. SMOTE on Clean Training Data
# -------------------------------
print("\n⚖️ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", X_train.shape)
print("After SMOTE:", X_train_res.shape)

# -------------------------------
# 9. Sanity Check for NaNs after SMOTE
# -------------------------------
if X_train_res.isnull().sum().sum() > 0:
    print("\n⚠️ Found NaNs after SMOTE, imputing...")
    for col in X_train_res.columns:
        if X_train_res[col].dtype.kind in "biufc":
            X_train_res[col] = X_train_res[col].fillna(X_train_res[col].median())
        else:
            X_train_res[col] = X_train_res[col].fillna(X_train_res[col].mode()[0])

print("\n✅ NaNs after SMOTE (should be 0):")
print(X_train_res.isnull().sum()[X_train_res.isnull().sum() > 0])

# -------------------------------
# 10. Save Processed Files
# -------------------------------
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

pd.concat([X_train_res, y_train_res.rename("No_show_label")], axis=1).to_csv(processed_dir / "train_processed_smote.csv", index=False)
pd.concat([X_val, y_val.rename("No_show_label")], axis=1).to_csv(processed_dir / "val_processed.csv", index=False)
pd.concat([X_test, y_test.rename("No_show_label")], axis=1).to_csv(processed_dir / "test_processed.csv", index=False)

print("\n✅ Preprocessing completed successfully. Clean files saved in data/processed/")

