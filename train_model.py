
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.base import clone


# -------------------------------------------------
# 1. File paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# 2. Load and inspect the dataset
# -------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Check your CSV filename and data folder."
    )

df = pd.read_csv(DATA_PATH)

# Normalize accidental whitespace in column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

if "Churn" not in df.columns:
    raise ValueError("The dataset must contain a 'Churn' column.")

# Remove rows without a known target
df["Churn"] = df["Churn"].astype("string").str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()

# Remove identifier; it should not be a predictive feature
df.drop(columns=["customerID"], errors="ignore", inplace=True)

# Convert TotalCharges to numeric when present
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

print("\nChurn distribution:")
print(df["Churn"].value_counts())


# -------------------------------------------------
# 3. Separate features and target
# -------------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)


# -------------------------------------------------
# 4. Split the data
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split part of the training data for model selection.
# The test set remains untouched until final evaluation.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# -------------------------------------------------
# 5. Build preprocessing pipeline
# -------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -------------------------------------------------
# 6. Define candidate models
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# -------------------------------------------------
# 7. Train and compare models on validation data
# -------------------------------------------------

validation_results = {}
trained_candidates = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(estimator))
    ])

    pipeline.fit(X_fit, y_fit)

    probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_val, probabilities)

    validation_results[name] = {
        "roc_auc": float(auc),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        )
    }

    trained_candidates[name] = pipeline

print("\nValidation comparison:")
print(pd.DataFrame(validation_results).T.round(4))

# Select by ROC-AUC, not by training accuracy
best_name = max(
    validation_results,
    key=lambda name: validation_results[name]["roc_auc"]
)

print("\nSelected model:", best_name)


# -------------------------------------------------
# 8. Refit selected model on all training data
# -------------------------------------------------

best_model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", clone(models[best_name]))
])

best_model.fit(X_train, y_train)

# Evaluate once on the held-out test set
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.5).astype(int)

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    )
}

print("\nFinal test metrics:")
print(pd.Series(test_metrics).round(4))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))


# -------------------------------------------------
# 9. Save the complete pipeline and metadata
# -------------------------------------------------

bundle = {
    "model": best_model,
    "model_name": best_name,
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": test_metrics,
    "threshold": 0.5
}

joblib.dump(bundle, MODEL_DIR / "churn_model.joblib")

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump({
        "selected_model": best_name,
        "metrics": test_metrics,
        "validation_results": validation_results
    }, f, indent=4)

print("\nModel saved to models/churn_model.joblib")
print("Metrics saved to models/metrics.json")
print("Training completed successfully!")
