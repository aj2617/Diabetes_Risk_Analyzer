import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# Load dataset

df = pd.read_csv("diabetes.csv")
print(df.head())
print("Shape:", df.shape)


# Target and features

TARGET_COL = "Outcome"  

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL].astype(int)


# Replace 0 with NaN (common in this dataset)
# 0 is not valid for these medical measurements

zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in zero_as_missing_cols:
    if c in X.columns:
        X[c] = X[c].replace(0, np.nan)


# Column split
# (This dataset is numeric-only, but kept as template style)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# Preprocessing

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# If there are no categorical columns, we just skip them safely
transformers = []
if len(numeric_features) > 0:
    transformers.append(("num", num_transformer, numeric_features))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# Random Forest Model (Classifier)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # ✅ helps if class imbalance exists
)


# Full Pipeline

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])


# Train-test split (stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train

rf_pipeline.fit(X_train, y_train)


# Evaluation

y_pred = rf_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n✅ Evaluation:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# ROC-AUC (needs probabilities)
try:
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC  : {auc:.4f}")
except Exception as e:
    print("ROC-AUC could not be computed:", e)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model 

with open("diabetes_rf_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("\n✅ Random Forest pipeline saved as diabetes_rf_pipeline.pkl")
