"""
feature_engineering.py
----------------------
Feature engineering for the Telco Customer Churn project.

This module:
1. Creates a few interpretable, business-motivated derived features.
2. Builds a scikit-learn ColumnTransformer that imputes, one-hot encodes
   categorical features and scales numeric features. Wrapping preprocessing in a
   ColumnTransformer keeps the pipeline leak-free (imputers/scalers/encoders are
   fit on training folds only) and makes the model trivial to deploy.

Why a Pipeline + ColumnTransformer (interview talking point)
------------------------------------------------------------
All preprocessing is fit *only* on the training data inside each CV fold. This
prevents data leakage (e.g. computing scaling statistics or imputation values
using the test set) which would otherwise inflate the reported metrics.

Author: Naveen
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add business-motivated derived features.

    New features
    ------------
    - tenure_group : binned customer lifetime (0-12, 12-24, 24-48, 48-60, 60-72
      months). Churn is strongly concentrated in early tenure, so a coarse
      grouping is interpretable for stakeholders.
    - avg_monthly_spend : TotalCharges / tenure. A normalized spend measure that
      smooths out the noise in a single month's bill (guards against divide-by-zero
      for tenure == 0 customers).
    - has_streaming : 1 if the customer subscribes to streaming TV or movies.
    - num_addon_services : count of optional protective/support add-ons the
      customer subscribes to (online security, online backup, device protection,
      tech support). Fewer add-ons => "stickier-but-unprotected" customers who
      tend to churn more.

    Parameters
    ----------
    X : pd.DataFrame
        Cleaned feature matrix (no target).

    Returns
    -------
    pd.DataFrame
        Feature matrix with new columns appended.
    """
    X = X.copy()

    # Tenure buckets (months). Right-open bins.
    X["tenure_group"] = pd.cut(
        X["tenure"],
        bins=[-0.1, 12, 24, 48, 60, 72],
        labels=["0-12", "12-24", "24-48", "48-60", "60-72"],
    ).astype(str)

    # Normalized average monthly spend. Guard tenure == 0 to avoid divide-by-zero.
    safe_tenure = X["tenure"].replace(0, 1)
    X["avg_monthly_spend"] = (X["TotalCharges"] / safe_tenure).round(2)

    # Streaming flag derived from two service columns.
    streaming_tv = X["StreamingTV"] == "Yes"
    streaming_movies = X["StreamingMovies"] == "Yes"
    X["has_streaming"] = (streaming_tv | streaming_movies).astype(int)

    # Count of optional protective add-on services.
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    present = [c for c in addon_cols if c in X.columns]
    if present:
        X["num_addon_services"] = (X[present] == "Yes").sum(axis=1)

    return X


def split_column_types(X: pd.DataFrame):
    """Identify numeric and categorical columns for the preprocessor.

    Returns
    -------
    tuple(list, list)
        (numeric_columns, categorical_columns)
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer with imputation + scaling + one-hot encoding.

    Numeric branch:  median imputation -> standard scaling.
    Categorical branch: most-frequent imputation -> one-hot encoding.

    Median/most-frequent imputation makes the pipeline robust to unexpected
    missing values at inference time (a production concern), even though the
    cleaned training data has none left.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used only to infer column types/names).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer to embed in a Pipeline.
    """
    numeric_cols, categorical_cols = split_column_types(X)

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
