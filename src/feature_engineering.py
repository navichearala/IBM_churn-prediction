"""
feature_engineering.py
----------------------
Feature engineering for the Telco Customer Churn project.

This module:
1. Creates a few interpretable, business-motivated derived features.
2. Builds a scikit-learn ColumnTransformer that one-hot encodes categorical
   features and scales numeric features. Wrapping preprocessing in a
   ColumnTransformer keeps the pipeline leak-free (the scaler/encoder are fit
   on training folds only) and makes the model easy to deploy.

Author: Naveen
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
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
    """Build a ColumnTransformer that scales numerics and one-hot encodes categoricals.

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor
