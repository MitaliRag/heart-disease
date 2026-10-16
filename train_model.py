
from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load dataset
# -----------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure data/heart.csv exists."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Original columns:", list(df.columns))


# -----------------------------------------
# 3. Rename dataset columns
# -----------------------------------------

df = df.rename(columns={
    "Age": "age",
    "Sex": "sex",
    "ChestPainType": "cp",
    "RestingBP": "trestbps",
    "Cholesterol": "chol",
    "FastingBS": "fbs",
    "RestingECG": "restecg",
    "MaxHR": "thalach",
    "ExerciseAngina": "exang",
    "Oldpeak": "oldpeak",
    "ST_Slope": "slope",
    "HeartDisease": "target"
})


# -----------------------------------------
# 4. Check required columns
# -----------------------------------------

feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope"
]

required_columns = feature_names + ["target"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# -----------------------------------------
# 5. Convert categorical values to numbers
# -----------------------------------------

category_mappings = {
    "cp": {
        "TA": 0,
        "ATA": 1,
        "NAP": 2,
        "ASY": 3
    },
    "restecg": {
        "Normal": 0,
        "ST": 1,
        "LVH": 2
    },
    "exang": {
        "N": 0,
        "Y": 1
    },
    "slope": {
        "Up": 0,
        "Flat": 1,
        "Down": 2
    }
}

for column, mapping in category_mappings.items():
    df[column] = df[column].replace(mapping)


# -----------------------------------------
# 6. Prepare input features and target
# -----------------------------------------

X = df[feature_names].apply(
    pd.to_numeric, errors="coerce"
)

X = X.replace([float("inf"), float("-inf")], float("nan"))

y = pd.to_numeric(df["target"], errors="coerce")

valid_rows = y.notna()
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        "HeartDisease target must contain only 0 and 1."
    )

y = y.astype(int)

if y.nunique() != 2:
    raise ValueError(
        "Dataset must contain both target classes: 0 and 1."
    )

print("Features:", feature_names)
print("Number of rows:", len(X))
print("Target distribution:")
print(y.value_counts())


# -----------------------------------------
# 7. Split training and testing data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 8. Handle missing values
# -----------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# -----------------------------------------
# 9. Scale the features
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# -----------------------------------------
# 10. Train machine learning model
# -----------------------------------------

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# -----------------------------------------
# 11. Evaluate model
# -----------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# -----------------------------------------
# 12. Save model and preprocessing files
# -----------------------------------------

joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
joblib.dump(feature_names, MODEL_DIR / "features.pkl")

print("\nAll files saved successfully in the model folder:")
print("- model.pkl")
print("- scaler.pkl")
print("- imputer.pkl")
print("- features.pkl")
print("\nTraining process finished successfully!")

