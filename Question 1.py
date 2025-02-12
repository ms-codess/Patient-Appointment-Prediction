#Part 1: Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("MedicalCentre.csv")

# Show initial records
print("Initial records:")
print(df.head())

# Drop irrelevant ID columns
df.drop(columns=["PatientID", "AppointmentID"], inplace=True)
print("\nDataFrame after dropping IDs:")
print(df.head())

# Handle missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)
print("\nMissing values after imputation:")
print(pd.isnull(df).sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle negative ages
negative_age_count = (df["Age"] < 0).sum()
print(f"\nNegative Age count: {negative_age_count}")
df = df[df["Age"] >= 0]

# Create AgeGroup
age_bins = [0, 2, 6, 11, 16, 26, 31, 36, 41, 51, 115]
age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
print("\nDataFrame with AgeGroup:")
print(df.head())

# Detect and visualize outliers
plt.figure()
df.boxplot(column="Age")
plt.title("Age Distribution Before Outlier Removal")
plt.show()

# Rescale Age using Z-score
df['Age_zscore'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
df = df[(df['Age_zscore'].abs() <= 3)]
df.drop(columns=["Age"], inplace=True)

# Create AwaitingTime
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AwaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.abs()

# Create AwaitingTimeGroup
awaiting_bins = [0, 1, 8, 31, 91, 180]
awaiting_labels = [1, 2, 3, 4, 5]
df['AwaitingTimeGroup'] = pd.cut(df['AwaitingTime'], bins=awaiting_bins, labels=awaiting_labels, right=False)
df.drop(columns=["AwaitingTime"], inplace=True)

# Extract date components
for col in ['AppointmentDay', 'ScheduledDay']:
    df[f'{col[:4]}_Year'] = df[col].dt.year
    df[f'{col[:4]}_Month'] = df[col].dt.month
    df[f'{col[:4]}_Day'] = df[col].dt.day
df.drop(columns=['AppointmentDay', 'ScheduledDay'], inplace=True)

# Encode categorical variables
encoders = {
    'Gender': LabelEncoder(),
    'No-show': LabelEncoder(),
    'Neighbourhood': LabelEncoder()
}
for col, encoder in encoders.items():
    df[f'{col}_encoded'] = encoder.fit_transform(df[col])
df.drop(columns=list(encoders.keys()), inplace=True)


# Filter out unnecessary columns for correlation analysis
# Ensure only numeric and relevant features are included
relevant_features = [
    'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 
    'SMS_received', 'Gender_encoded', 'No-show_encoded', 'AgeGroup', 
    'AwaitingTimeGroup'
]


# Subset the DataFrame to only include relevant features
df_relevant = df[relevant_features]

# Ensure all columns are numeric
df_relevant = df_relevant.apply(pd.to_numeric, errors='coerce')

# Check for constant columns (all values the same)
constant_columns = df_relevant.columns[df_relevant.nunique() == 1]
df_relevant = df_relevant.drop(columns=constant_columns)

# Calculate the correlation matrix for relevant features only
correlation_matrix = df_relevant.corr()


    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix (Relevant Features Only)')
    plt.show()

    # Set a correlation threshold for dropping highly correlated variables
    threshold = 0.7

    # Find pairs of highly correlated features (excluding self-correlation)
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Identify features to drop by checking if the correlation exceeds the threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]

    # Drop the highly correlated features from the relevant DataFrame
    df_relevant_cleaned = df_relevant.drop(columns=to_drop)

    # Print the columns that were dropped
    print("Dropped the following highly correlated features:", to_drop)

    # Save the cleaned DataFrame
    df_relevant_cleaned.to_csv("Cleaned_MedicalCentre.csv", index=False)

# Part 2: Model Training and Evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import Binarizer

# Load the cleaned dataset from Part 1
df = pd.read_csv("Cleaned_MedicalCentre.csv")

# Ensure the target variable is binary (0 or 1)
if 'No-show_encoded' not in df.columns:
    raise ValueError("Target variable 'No-show_encoded' not found in the dataset.")

# Separate features (X) and target variable (y)
X = df.drop(columns=['No-show_encoded'])  # Features
y = df['No-show_encoded']  # Target variable

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Binarize continuous features (if any) for BernoulliNB
# Assuming 'AgeGroup' and 'AwaitingTimeGroup' are categorical, no need to binarize
# If there are continuous features, binarize them using a threshold
continuous_features = []  # Add continuous feature names here if any
if continuous_features:
    scaler = Binarizer(threshold=0)  # Binarize any continuous feature values > 0 to 1
    X_train_continuous = scaler.fit_transform(X_train[continuous_features])
    X_test_continuous = scaler.transform(X_test[continuous_features])
else:
    X_train_continuous = np.zeros((X_train.shape[0], 0))  # No continuous features
    X_test_continuous = np.zeros((X_test.shape[0], 0))  # No continuous features

# Encode categorical features using one-hot encoding
X_train_categorical = pd.get_dummies(X_train.drop(columns=continuous_features))
X_test_categorical = pd.get_dummies(X_test.drop(columns=continuous_features))

# Ensure both train and test sets have the same columns after one-hot encoding
X_test_categorical = X_test_categorical.reindex(columns=X_train_categorical.columns, fill_value=0)

# Initialize the Gaussian Naïve Bayes (for continuous data) and Bernoulli Naïve Bayes (for categorical data)
gaussian_nb = GaussianNB()
bernoulli_nb = BernoulliNB()

# Train Gaussian Naïve Bayes on continuous data (if any)
if X_train_continuous.shape[1] > 0:
    gaussian_nb.fit(X_train_continuous, y_train)

# Train Bernoulli Naïve Bayes on categorical data
bernoulli_nb.fit(X_train_categorical, y_train)

# Make predictions on the test set
if X_test_continuous.shape[1] > 0:
    y_pred_continuous = gaussian_nb.predict(X_test_continuous)
else:
    y_pred_continuous = np.zeros_like(y_test)  # No continuous features

y_pred_categorical = bernoulli_nb.predict(X_test_categorical)

# Combine the predictions from both classifiers
# Here, we use a simple majority vote: if both classifiers agree, use their prediction; otherwise, default to BernoulliNB
y_pred_combined = np.where(y_pred_continuous == y_pred_categorical, y_pred_continuous, y_pred_categorical)

# Evaluate the performance of the combined model
accuracy = accuracy_score(y_test, y_pred_combined)
precision = precision_score(y_test, y_pred_combined)
recall = recall_score(y_test, y_pred_combined)
f1 = f1_score(y_test, y_pred_combined)

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_combined))
# Part 3: Model Comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
import xgboost as xgb

# Train a Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_categorical, y_train)
y_pred_dt = dt.predict(X_test_categorical)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_categorical, y_train)
y_pred_rf = rf.predict(X_test_categorical)

# Train an XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train_categorical, y_train)
y_pred_xgb = xgb_clf.predict(X_test_categorical)

# Evaluate all models
models = {
    'Naïve Bayes': y_pred_combined,  # From Part 2
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb
}

results = {}
for model_name, y_pred in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    results[model_name] = [accuracy, precision, recall, specificity, f1]

# Print performance metrics
print("Model Performance Comparison:")
print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    'Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'))
for model, scores in results.items():
    print("{:<15} {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}".format(model, *scores))

# Feature Importance (Random Forest)
feature_importances = rf.feature_importances_
feature_names = X_train_categorical.columns
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx], color='royalblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.show()

# ROC Analysis
plt.figure(figsize=(10, 6))
for model_name, y_pred in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

