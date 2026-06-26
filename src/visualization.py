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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

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


def plot_confusion_matrix(pipeline, X_test, y_test, model_name: str, threshold: float = 0.5):
    """Confusion matrix for the best model at the tuned decision threshold."""
    _ensure_dir()
    from sklearn.metrics import confusion_matrix

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"]).plot(
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold:.2f})")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(results):
    """Grouped bar chart comparing all models across the key metrics.

    `results` is a list of ModelResult dataclasses (or objects with the
    matching attributes).
    """
    _ensure_dir()
    names = [r.name for r in results]
    metrics = {
        "Accuracy": [r.accuracy for r in results],
        "Precision": [r.precision for r in results],
        "Recall": [r.recall for r in results],
        "F1": [r.f1 for r in results],
        "ROC-AUC": [r.roc_auc for r in results],
    }
    x = np.arange(len(names))
    width = 0.15
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + (i - 2) * width, vals, width, label=metric, color=palette[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set (metrics at tuned threshold)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "model_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall_curve(pipeline, X_test, y_test, model_name: str):
    """Precision-recall curve — more informative than ROC under imbalance."""
    _ensure_dir()
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline = float((y_test == 1).mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#8172B3", lw=2, label=model_name)
    ax.axhline(baseline, color="grey", lw=1, linestyle="--",
               label=f"No-skill ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "precision_recall_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_threshold_tuning(pipeline, X_test, y_test, chosen_threshold, model_name: str):
    """Show precision, recall and F1 as a function of the decision threshold.

    The vertical line marks the F1-optimal threshold selected on the train set.
    """
    _ensure_dir()
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    prec, rec, f1s = [], [], []
    for t in grid:
        pred = (y_proba >= t).astype(int)
        prec.append(precision_score(y_test, pred, zero_division=0))
        rec.append(recall_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, prec, label="Precision", color="#4C72B0")
    ax.plot(grid, rec, label="Recall", color="#DD8452")
    ax.plot(grid, f1s, label="F1", color="#C44E52", lw=2)
    ax.axvline(chosen_threshold, color="black", linestyle="--",
               label=f"Chosen = {chosen_threshold:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", label="Default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning — {model_name}")
    ax.legend(loc="center right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "threshold_tuning.png")
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
