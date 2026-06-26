"""
model_training.py
-----------------
Train and evaluate churn classification models.

Three models are compared inside a single, leak-free scikit-learn Pipeline:
  - Logistic Regression (interpretable linear baseline)
  - Random Forest (non-linear, robust)
  - Gradient Boosting (typically strongest on tabular data)

Because churn is imbalanced (~26.5% positive), we evaluate with ROC-AUC,
precision, recall and F1 in addition to accuracy, and use stratified splitting.

Author: Naveen
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feature_engineering import build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass
class ModelResult:
    """Container for a single model's evaluation metrics."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def get_models() -> dict:
    """Return the candidate estimators keyed by display name."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split that preserves the churn ratio."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> ModelResult:
    """Compute classification metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return ModelResult(
        name=name,
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        precision=round(precision_score(y_test, y_pred), 4),
        recall=round(recall_score(y_test, y_pred), 4),
        f1=round(f1_score(y_test, y_pred), 4),
        roc_auc=round(roc_auc_score(y_test, y_proba), 4),
    )


def train_and_compare(X: pd.DataFrame, y: pd.Series):
    """Train all candidate models and return results + fitted pipelines.

    Returns
    -------
    tuple(list[ModelResult], dict[str, Pipeline], tuple)
        (results, fitted_pipelines, (X_train, X_test, y_train, y_test))
    """
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))
        fitted[name] = pipeline

    return results, fitted, (X_train, X_test, y_train, y_test)


def best_model(results: list[ModelResult]) -> str:
    """Pick the best model by ROC-AUC (primary metric for imbalanced churn)."""
    return max(results, key=lambda r: r.roc_auc).name


def save_artifacts(
    results: list[ModelResult],
    fitted: dict,
    out_dir: str = "models",
    metrics_path: str = "reports/metrics.json",
):
    """Persist the best pipeline and a metrics JSON for the README/report."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    best_name = best_model(results)
    joblib.dump(fitted[best_name], os.path.join(out_dir, "churn_model.joblib"))

    payload = {
        "best_model": best_name,
        "results": [asdict(r) for r in results],
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    return best_name, payload


def text_report(pipeline: Pipeline, X_test, y_test) -> str:
    """Return a sklearn classification report + confusion matrix as text."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Churn"])
    cm = confusion_matrix(y_test, y_pred)
    return f"{report}\nConfusion matrix [rows=actual, cols=pred]:\n{cm}"
