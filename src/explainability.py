"""
explainability.py
------------------
Model interpretability for the Telco Customer Churn project.

Two complementary, model-agnostic techniques are provided:

1. Permutation importance (sklearn)
   - Shuffles one feature at a time on the held-out TEST set and measures the
     drop in ROC-AUC. A large drop => the model genuinely relies on that feature.
   - Computed on transformed feature names so results are human-readable.
   - Leak-free: uses only the test set and the already-fitted pipeline.

2. SHAP values
   - Game-theoretic per-prediction attributions. For tree models we use the fast
     exact TreeExplainer; for other models we fall back to a sampled explainer.
   - Produces a global summary (mean |SHAP|) bar plot.

Both answer the recruiter/interview question: "Which variables drive churn, and
can you defend the model's decisions?"

Author: Naveen
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

FIG_DIR = "reports/figures"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _feature_names(pipeline) -> np.ndarray:
    """Get transformed feature names out of the fitted preprocessor."""
    preprocess = pipeline.named_steps["preprocess"]
    return preprocess.get_feature_names_out()


def permutation_importance_plot(
    pipeline, X_test, y_test, model_name: str, top_n: int = 15
):
    """Permutation importance on the test set, scored by ROC-AUC.

    Returns the saved figure path (or None if it cannot be computed).
    """
    _ensure_dir()
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # Permutation importance is on the *input* columns of the pipeline.
    feat_names = np.array(X_test.columns)
    importances = result.importances_mean
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        feat_names[order][::-1],
        importances[order][::-1],
        xerr=result.importances_std[order][::-1],
        color="#55A868",
    )
    ax.set_title(f"Permutation Importance (ROC-AUC drop) — {model_name}")
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "permutation_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def shap_summary_plot(pipeline, X_train, X_test, model_name: str, max_samples: int = 500):
    """Global SHAP summary (mean |SHAP|) for the best model.

    Transforms the data through the fitted preprocessor, then explains the bare
    estimator. Uses TreeExplainer for tree/boosting models and a sampled
    permutation explainer otherwise. Returns the saved figure path or None.
    """
    try:
        import shap
    except Exception:
        return None

    _ensure_dir()
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feat_names = _feature_names(pipeline)

    # Transform to the numeric matrix the model actually sees.
    X_test_t = preprocess.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()
    X_test_t = np.asarray(X_test_t, dtype=float)

    # Subsample for speed/clarity.
    if X_test_t.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_test_t.shape[0], max_samples, replace=False)
        X_sample = X_test_t[idx]
    else:
        X_sample = X_test_t

    is_tree = model.__class__.__name__ in {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    }

    try:
        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            bg = shap.sample(X_sample, min(100, len(X_sample)), random_state=42)
            explainer = shap.Explainer(model.predict_proba, bg)
            shap_values = explainer(X_sample).values
    except Exception:
        return None

    # Normalize multi-output SHAP (some explainers return per-class arrays).
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[-1]  # positive (churn) class
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]  # (samples, features, classes) -> churn class

    fig = plt.figure(figsize=(8, 6))
    try:
        shap.summary_plot(
            sv,
            features=X_sample,
            feature_names=list(feat_names),
            plot_type="bar",
            show=False,
            max_display=15,
        )
    except Exception:
        plt.close(fig)
        return None
    plt.title(f"SHAP Global Importance (mean |SHAP|) — {model_name}")
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "shap_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def top_drivers_table(pipeline, X_test, y_test, top_n: int = 10) -> pd.DataFrame:
    """Return a tidy DataFrame of the top permutation-importance drivers."""
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).head(top_n)
    return df.reset_index(drop=True)
