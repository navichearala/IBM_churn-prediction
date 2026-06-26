"""
data_preprocessing.py
---------------------
Reusable functions for loading and cleaning the Telco Customer Churn dataset.

The raw dataset is the public IBM "Telco Customer Churn" sample (7,043 rows, 21
columns). The main data-quality issue is that the `TotalCharges` column is stored
as text and contains 11 blank values for brand-new customers (tenure = 0).

Author: Naveen
"""

from __future__ import annotations

import pandas as pd


# Columns that should be treated as numeric but are sometimes parsed as object.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# The unique identifier carries no predictive signal and is dropped before modeling.
ID_COLUMN = "customerID"

# Target column in the raw data ("Yes"/"No").
TARGET_COLUMN = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw churn DataFrame.

    Steps
    -----
    1. Strip the `customerID` column (no predictive value).
    2. Convert `TotalCharges` from text to numeric, coercing blanks to NaN.
    3. Impute the small number of missing `TotalCharges` (tenure = 0 customers)
       with 0, since they have not been billed yet.
    4. Encode the target `Churn` from "Yes"/"No" to 1/0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    df = df.copy()

    # 1. Drop the identifier column if present.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # 2. TotalCharges is read as object because of blank strings -> coerce to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Blanks correspond to tenure == 0 (never billed). Impute with 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 4. Encode target as integer 0/1 (robust to object or string dtype).
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    return df


def get_feature_target(df: pd.DataFrame):
    """Split a cleaned DataFrame into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    tuple(pd.DataFrame, pd.Series)
        (X, y) where y is the binary churn target.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    # Quick manual smoke test when run directly.
    raw = load_data("data/raw/Telco-Customer-Churn.csv")
    cleaned = clean_data(raw)
    X, y = get_feature_target(cleaned)
    print(f"Rows: {len(cleaned)} | Features: {X.shape[1]} | Churn rate: {y.mean():.3f}")
