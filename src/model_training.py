"""
model_training.py
-----------------
Train, tune and evaluate churn classification models.

Five models are compared inside a single, leak-free scikit-learn Pipeline:
  - Logistic Regression   (interpretable linear BASELINE)
  - Random Forest         (non-linear, robust bagging ensemble)
  - Gradient Boosting     (sklearn boosting)
  - XGBoost               (regularized gradient boosting, strong on tabular)
  - LightGBM              (fast histogram-based boosting)

Methodology
-----------
* Stratified 80/20 hold-out split (preserves the ~26.5% churn ratio).
* Stratified 5-fold cross-validation on the TRAIN set to estimate ROC-AUC with
  a stability band (mean +/- std) -> guards against a lucky single split.
* Class imbalance handled per-model: `class_weight="balanced"` for LR/RF and
  `scale_pos_weight = (neg/pos)` for XGBoost/LightGBM.
* Hyperparameter tuning via RandomizedSearchCV (ROC-AUC scoring) for the
  boosting models, which carry the most tunable capacity.
* Decision-threshold tuning: instead of the default 0.5, pick the probability
  cutoff that maximizes F1 on the train set, then report metrics at that
  threshold on the untouched test set. This directly targets the
  precision/recall trade-off that matters for a churn-retention budget.

Primary selection metric: ROC-AUC (threshold-independent, appropriate for an
imbalanced ranking problem).

Author: Naveen
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from feature_engineering import build_preprocessor

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LGBM = False

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 15  # RandomizedSearchCV sampling budget per tuned model


@dataclass
class ModelResult:
    """Container for a single model's evaluation metrics (at tuned threshold)."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    threshold: float
    tuned: bool = False
    best_params: dict = field(default_factory=dict)


def _scale_pos_weight(y: pd.Series) -> float:
    """Imbalance ratio (negatives / positives) for boosting models."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return round(neg / max(pos, 1), 4)


def get_models(y_train: pd.Series) -> dict:
    """Return the candidate estimators keyed by display name.

    Class imbalance is addressed directly inside each estimator:
    * LR / RF        -> class_weight="balanced"
    * XGBoost/LightGBM -> scale_pos_weight = neg/pos
    """
    spw = _scale_pos_weight(y_train)

    models: dict = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )

    return models


def get_search_spaces() -> dict:
    """RandomizedSearchCV parameter distributions for the tunable models.

    Keys are prefixed with ``model__`` because the estimator is the ``model``
    step inside the Pipeline.
    """
    spaces: dict = {
        "Random Forest": {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [8, 12, 16, 24],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [150, 250, 400],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 0.9, 1.0],
        },
    }
    if _HAS_XGB:
        spaces["XGBoost"] = {
            "model__n_estimators": [300, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
        }
    if _HAS_LGBM:
        spaces["LightGBM"] = {
            "model__n_estimators": [300, 400, 600],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 6, 10],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    return spaces


# Models we actually run RandomizedSearchCV on (most capacity to gain from tuning).
TUNED_MODELS = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split that preserves the churn ratio."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def best_threshold_by_f1(y_true, y_proba) -> float:
    """Find the probability threshold that maximizes F1 on the given data.

    Returns the cutoff in (0, 1). Used on the TRAIN set only, then applied to
    the test set to avoid tuning on the evaluation data.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have length n_thresholds + 1; align by dropping last point.
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    f1 = f1[:-1]
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def evaluate_at_threshold(
    name: str,
    pipeline: Pipeline,
    X_test,
    y_test,
    threshold: float,
    cv_mean: float,
    cv_std: float,
    tuned: bool,
    best_params: dict,
) -> ModelResult:
    """Compute classification metrics for a fitted pipeline at a given threshold."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return ModelResult(
        name=name,
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_test, y_pred), 4),
        f1=round(f1_score(y_test, y_pred), 4),
        roc_auc=round(roc_auc_score(y_test, y_proba), 4),
        cv_roc_auc_mean=round(cv_mean, 4),
        cv_roc_auc_std=round(cv_std, 4),
        threshold=round(threshold, 3),
        tuned=tuned,
        best_params=best_params,
    )


def train_and_compare(X: pd.DataFrame, y: pd.Series, tune: bool = True):
    """Train, cross-validate, optionally tune, and evaluate all candidate models.

    For each model:
      1. Build a leak-free Pipeline (preprocess -> model).
      2. Estimate ROC-AUC via stratified 5-fold CV on the TRAIN set.
      3. If `tune` and the model is tunable, run RandomizedSearchCV (ROC-AUC).
      4. Fit the (best) pipeline on the full TRAIN set.
      5. Tune the decision threshold for F1 on TRAIN, evaluate on TEST.

    Returns
    -------
    tuple(list[ModelResult], dict[str, Pipeline], tuple)
        (results, fitted_pipelines, (X_train, X_test, y_train, y_test))
    """
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = get_search_spaces()

    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in get_models(y_train).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        # ---- 5-fold cross-validated ROC-AUC on TRAIN (stability estimate) ----
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

        # ---- Optional hyperparameter tuning (boosting/forest models) ---------
        tuned = False
        best_params: dict = {}
        if tune and name in TUNED_MODELS and name in search_spaces:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=search_spaces[name],
                n_iter=N_ITER_SEARCH,
                scoring="roc_auc",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = {
                k.replace("model__", ""): v for k, v in search.best_params_.items()
            }
            cv_mean, cv_std = float(search.best_score_), cv_std
            tuned = True
        else:
            pipeline.fit(X_train, y_train)

        # ---- Decision-threshold tuning on TRAIN, evaluate on TEST ------------
        train_proba = pipeline.predict_proba(X_train)[:, 1]
        thr = best_threshold_by_f1(y_train, train_proba)

        results.append(
            evaluate_at_threshold(
                name, pipeline, X_test, y_test, thr, cv_mean, cv_std, tuned, best_params
            )
        )
        fitted[name] = pipeline

    # Sort results best-first by ROC-AUC for nicer reporting.
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    return results, fitted, (X_train, X_test, y_train, y_test)


def best_model(results: list[ModelResult]) -> str:
    """Pick the best model by test ROC-AUC (primary metric for imbalanced churn)."""
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
    best_res = next(r for r in results if r.name == best_name)

    joblib.dump(fitted[best_name], os.path.join(out_dir, "churn_model.joblib"))

    payload = {
        "best_model": best_name,
        "best_threshold": best_res.threshold,
        "primary_metric": "roc_auc",
        "cv_folds": CV_FOLDS,
        "results": [asdict(r) for r in results],
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    return best_name, payload


def get_best_threshold(results: list[ModelResult], name: str) -> float:
    """Return the tuned decision threshold for a named model."""
    for r in results:
        if r.name == name:
            return r.threshold
    return 0.5


def text_report(pipeline: Pipeline, X_test, y_test, threshold: float = 0.5) -> str:
    """Return a sklearn classification report + confusion matrix at `threshold`."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Churn"])
    cm = confusion_matrix(y_test, y_pred)
    return (
        f"{report}\nConfusion matrix [rows=actual, cols=pred] @ threshold="
        f"{threshold:.3f}:\n{cm}"
    )
