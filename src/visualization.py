"""
visualization.py
----------------
Plotting helpers for the Telco Customer Churn project. All figures are saved to
the reports/figures/ directory so they can be embedded in the README and report.

Author: Naveen
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend for script/CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

sns.set_theme(style="whitegrid")
FIG_DIR = "reports/figures"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_churn_balance(df: pd.DataFrame, target: str = "Churn"):
    """Bar chart of the class balance (churn vs stay)."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts().sort_index()
    labels = ["Stay (0)", "Churn (1)"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("Customer Churn Class Balance")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "churn_balance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_churn_by_contract(df: pd.DataFrame):
    """Churn rate by contract type — a key business driver."""
    _ensure_dir()
    rate = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rate.index, rate.values * 100, color="#C44E52")
    for i, v in enumerate(rate.values):
        ax.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn rate (%)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "churn_by_contract.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_tenure_distribution(df: pd.DataFrame):
    """Tenure distribution split by churn status."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        subset = df[df["Churn"] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.6, label="Churn" if label else "Stay", color=color)
    ax.set_title("Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of customers")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "tenure_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    _ensure_dir()
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(pipeline, X_test, y_test, model_name: str):
    """Confusion matrix for the best model."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(pipeline, X_test, y_test, model_name: str):
    """ROC curve for the best model."""
    _ensure_dir()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "roc_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(pipeline, model_name: str, top_n: int = 15):
    """Plot top feature importances/coefficients from the fitted pipeline.

    Works for tree models (feature_importances_) and linear models (coef_).
    Returns None if the model exposes neither.
    """
    _ensure_dir()
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Top {top_n} Feature Importances — {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"Top {top_n} Coefficient Magnitudes — {model_name}"
    else:
        return None

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#4C72B0",
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
