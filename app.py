
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ==================================================
# 1. APP CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "custom_churn_model.joblib"

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Prediction & Retention Analytics")

st.write(
    "Train a machine learning model using your own customer data, "
    "evaluate its performance, and identify customers who may churn."
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def load_saved_model():
    """Load a previously trained custom model."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            st.warning(f"Could not load the saved model: {exc}")

    return None


def build_pipeline(df, features, algorithm):
    """Build preprocessing and classification steps."""

    numeric_features = [
        col
        for col in features
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col
        for col in features
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if algorithm == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return model, numeric_features, categorical_features


def prepare_features(df, feature_columns, numeric_features):
    """Validate and prepare new customer data for prediction."""

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing)
        )

    X = df[feature_columns].copy()

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if col not in numeric_features:
            X[col] = X[col].replace(
                r"^\s*$",
                np.nan,
                regex=True,
            )

    return X


# ==================================================
# 3. LOAD SAVED MODEL
# ==================================================

if "custom_bundle" not in st.session_state:
    st.session_state["custom_bundle"] = load_saved_model()

bundle = st.session_state["custom_bundle"]


# ==================================================
# 4. TRAIN A NEW MODEL
# ==================================================

st.header("1. Train a New Model")

st.write(
    "Upload historical customer data containing the actual churn "
    "outcome. Select the target column, input features, and algorithm."
)

training_file = st.file_uploader(
    "Upload labelled training CSV",
    type=["csv"],
    key="training_upload",
)

if training_file is not None:

    try:
        train_df = pd.read_csv(training_file)
        train_df.columns = train_df.columns.str.strip()

    except Exception as exc:
        st.error(f"Could not read the training CSV: {exc}")
        st.stop()

    if train_df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    if train_df.columns.duplicated().any():
        st.error("Your CSV contains duplicate column names.")
        st.stop()

    if len(train_df.columns) < 2:
        st.error(
            "The dataset needs a target column and at least one feature."
        )
        st.stop()

    st.write(
        f"**Dataset:** {len(train_df):,} rows, "
        f"{len(train_df.columns)} columns"
    )

    st.dataframe(
        train_df.head(10),
        width="stretch",
        hide_index=True,
    )

    target = st.selectbox(
        "Select the actual churn outcome column",
        options=train_df.columns.tolist(),
        key="training_target",
        help=(
            "Choose the column containing known outcomes, "
            "such as Yes/No, Churned/Retained, or 1/0."
        ),
    )

    # Normalize target labels and exclude missing labels.
    target_series = train_df[target].astype("string").str.strip()
    target_series = target_series.replace("", pd.NA)

    class_counts = target_series.dropna().value_counts()

    if len(class_counts) < 2:
        st.error(
            "The target column must contain at least two different classes."
        )
        st.stop()

    st.write("**Target class counts:**")

    st.dataframe(
        class_counts.rename("Rows").rename_axis("Class"),
        width="stretch",
    )

    positive_class = st.selectbox(
        "Which target value means the customer churned?",
        options=class_counts.index.tolist(),
        key="positive_class",
    )

    candidate_features = [
        col
        for col in train_df.columns
        if col != target
    ]

    excluded_names = {
        "customerid",
        "customer_id",
        "id",
        "churn",
        "churn_status",
        "churnlabel",
        "customer_left",
        "customerleft",
    }

    default_features = [
        col
        for col in candidate_features
        if col.strip().lower() not in excluded_names
    ]

    features = st.multiselect(
        "Select input features",
        options=candidate_features,
        default=default_features,
        key="training_features",
        help=(
            "Exclude IDs, the target, and columns that reveal "
            "the outcome after it happened."
        ),
    )

    algorithm = st.selectbox(
        "Choose a machine learning algorithm",
        [
            "Random Forest",
            "Logistic Regression",
        ],
        key="training_algorithm",
    )

    if st.button("Train and Evaluate Model", type="primary"):

        # ------------------------------
        # Validate selected features
        # ------------------------------

        if not features:
            st.error("Select at least one input feature.")
            st.stop()

        # Prepare labelled data.
        usable_df = train_df.copy()
        usable_df[target] = target_series
        usable_df = usable_df.dropna(subset=[target])

        y = usable_df[target].astype(str)
        X = usable_df[features].copy()

        # Clean empty text and infinite numeric values.
        X = X.replace(r"^\s*$", np.nan, regex=True)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Remove constant or entirely empty features.
        constant_features = [
            col
            for col in X.columns
            if X[col].nunique(dropna=True) <= 1
        ]

        if constant_features:
            st.warning(
                "Removing constant or empty features: "
                + ", ".join(constant_features)
            )

            X = X.drop(columns=constant_features)
            features = X.columns.tolist()

        if not features:
            st.error(
                "No usable features remain. Select columns "
                "containing meaningful customer information."
            )
            st.stop()

        counts = y.value_counts()

        if len(counts) < 2 or counts.min() < 2:
            st.error(
                "Each target class must contain at least two labelled "
                "rows. Add more examples before training."
            )
            st.stop()

        if str(positive_class) not in counts.index:
            st.error(
                "The selected churn class is no longer available. "
                "Please select the target class again."
            )
            st.stop()

        # ------------------------------
        # Split the dataset
        # ------------------------------

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

        except ValueError as exc:
            st.error(f"Could not split the dataset: {exc}")
            st.stop()

        # ------------------------------
        # Build and train the model
        # ------------------------------

        model, numeric_features, categorical_features = (
            build_pipeline(X_train, features, algorithm)
        )

        try:
            with st.spinner("Training and evaluating the model..."):

                # Train using training data only.
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)

                classes = list(
                    model.named_steps["classifier"].classes_
                )

                positive_index = classes.index(str(positive_class))

                positive_probabilities = probabilities[
                    :, positive_index
                ]

                actual_positive = (
                    y_test.astype(str) == str(positive_class)
                ).astype(int)

                predicted_positive = (
                    predictions.astype(str) == str(positive_class)
                ).astype(int)

                # Calculate evaluation metrics.
                accuracy = accuracy_score(y_test, predictions)

                precision = precision_score(
                    actual_positive,
                    predicted_positive,
                    zero_division=0,
                )

                recall = recall_score(
                    actual_positive,
                    predicted_positive,
                    zero_division=0,
                )

                f1 = f1_score(
                    actual_positive,
                    predicted_positive,
                    zero_division=0,
                )

                if len(np.unique(actual_positive)) == 2:
                    roc_auc = roc_auc_score(
                        actual_positive,
                        positive_probabilities,
                    )
                else:
                    roc_auc = float("nan")

                # Refit a fresh pipeline on all labelled data.
                final_model, _, _ = build_pipeline(
                    X,
                    features,
                    algorithm,
                )

                final_model.fit(X, y)

                # Save model and its configuration.
                new_bundle = {
                    "model": final_model,
                    "feature_columns": features.copy(),
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "target_column": target,
                    "positive_class": str(positive_class),
                    "algorithm": algorithm,
                    "metrics": {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "roc_auc": float(roc_auc),
                    },
                    "threshold": 0.50,
                }

                joblib.dump(new_bundle, MODEL_PATH)

                st.session_state["custom_bundle"] = new_bundle
                bundle = new_bundle

            st.success(
                "Training completed! Your model has been saved."
            )

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()


# ==================================================
# 5. MODEL PERFORMANCE
# ==================================================

bundle = st.session_state.get("custom_bundle")

if bundle is not None:

    st.divider()
    st.header("2. Model Performance")

    st.write(f"**Algorithm:** {bundle['algorithm']}")

    st.write(
        f"**Target column:** {bundle['target_column']}  \n"
        f"**Churn class:** {bundle['positive_class']}"
    )

    metrics = bundle["metrics"]

    cols = st.columns(5)

    cols[0].metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}",
    )

    cols[1].metric(
        "Precision",
        f"{metrics['precision']:.3f}",
    )

    cols[2].metric(
        "Recall",
        f"{metrics['recall']:.3f}",
    )

    cols[3].metric(
        "F1-score",
        f"{metrics['f1']:.3f}",
    )

    auc = metrics["roc_auc"]

    cols[4].metric(
        "ROC-AUC",
        f"{auc:.3f}" if np.isfinite(auc) else "N/A",
    )

    st.caption(
        "Evaluation metrics are calculated on the held-out test split. "
        "The saved model is retrained on all labelled rows for future "
        "predictions, so the displayed metrics are not an independent "
        "evaluation of that final refitted model."
    )

    st.write("**Features used by this model:**")

    st.code(", ".join(bundle["feature_columns"]))


# ==================================================
# 6. PREDICT CHURN FOR NEW CUSTOMERS
# ==================================================

st.divider()
st.header("3. Predict Churn for New Customers")

if bundle is None:

    st.info(
        "Train a model above before uploading customers for prediction."
    )

else:

    prediction_file = st.file_uploader(
        "Upload new customer CSV",
        type=["csv"],
        key="prediction_upload",
    )

    threshold = st.slider(
        "Churn decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(bundle.get("threshold", 0.50)),
        step=0.05,
        key="prediction_threshold",
        help=(
            "A lower threshold flags more customers as potential churners. "
            "A higher threshold flags fewer customers."
        ),
    )

    if prediction_file is not None:

        try:
            new_df = pd.read_csv(prediction_file)
            new_df.columns = new_df.columns.str.strip()

            if new_df.empty:
                st.error("The prediction dataset is empty.")
                st.stop()

            if new_df.columns.duplicated().any():
                st.error(
                    "The prediction CSV contains duplicate column names."
                )
                st.stop()

            if len(new_df) > 100_000:
                st.error(
                    "This app accepts at most 100,000 rows per upload."
                )
                st.stop()

            # Check and prepare the required feature columns.
            X_new = prepare_features(
                new_df,
                bundle["feature_columns"],
                bundle["numeric_features"],
            )

            # Calculate churn probabilities.
            probabilities = bundle["model"].predict_proba(X_new)

            classes = list(
                bundle["model"]
                .named_steps["classifier"]
                .classes_
            )

            positive_index = classes.index(bundle["positive_class"])

            churn_probability = probabilities[:, positive_index]

            # Add prediction results to the original data.
            results = new_df.copy()

            results["Churn Probability (%)"] = (
                churn_probability * 100
            ).round(2)

            results["Predicted Churn"] = np.where(
                churn_probability >= threshold,
                bundle["positive_class"],
                "Not " + bundle["positive_class"],
            )

            results["Risk Level"] = np.select(
                [
                    churn_probability >= 0.70,
                    churn_probability >= 0.40,
                ],
                [
                    "High",
                    "Medium",
                ],
                default="Low",
            )

            # ------------------------------
            # Customer analytics
            # ------------------------------

            st.subheader("Customer Analytics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Customers analyzed",
                f"{len(results):,}",
            )

            col2.metric(
                "Predicted churners",
                f"{int((churn_probability >= threshold).sum()):,}",
            )

            col3.metric(
                "High-risk customers",
                f"{int((churn_probability >= 0.70).sum()):,}",
            )

            col4.metric(
                "Average churn probability",
                f"{churn_probability.mean() * 100:.1f}%",
            )

            # ------------------------------
            # Charts
            # ------------------------------

            left, right = st.columns(2)

            with left:

                risk_counts = (
                    results["Risk Level"]
                    .value_counts()
                    .reindex(
                        ["Low", "Medium", "High"],
                        fill_value=0,
                    )
                    .rename_axis("Risk Level")
                    .reset_index(name="Customers")
                )

                fig_risk = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Customers",
                    color="Risk Level",
                    title="Customer Risk Distribution",
                    category_orders={
                        "Risk Level": ["Low", "Medium", "High"]
                    },
                )

                st.plotly_chart(
                    fig_risk,
                    width="stretch",
                )

            with right:

                fig_probability = px.histogram(
                    results,
                    x="Churn Probability (%)",
                    nbins=20,
                    title="Churn Probability Distribution",
                )

                st.plotly_chart(
                    fig_probability,
                    width="stretch",
                )

            # ------------------------------
            # Filter and display results
            # ------------------------------

            st.subheader("Customer-Level Predictions")

            risk_filter = st.multiselect(
                "Filter by risk level",
                ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"],
            )

            filtered_results = results[
                results["Risk Level"].isin(risk_filter)
            ]

            st.dataframe(
                filtered_results,
                width="stretch",
                hide_index=True,
            )

            # ------------------------------
            # Download results
            # ------------------------------

            csv_data = filtered_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(
                "Prediction failed. Check that your uploaded CSV contains "
                "all feature columns used during training and compatible "
                f"data. Details: {exc}"
            )
