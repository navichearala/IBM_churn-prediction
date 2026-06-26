"""
main.py
-------
End-to-end pipeline runner for the Telco Customer Churn project.

Run from the project root:
    python src/main.py            # full run WITH hyperparameter tuning
    python src/main.py --fast     # skip RandomizedSearchCV (quick smoke run)

This will:
  1. Load and clean the raw data.
  2. Engineer features.
  3. Generate EDA figures.
  4. Train, cross-validate, tune and compare five models.
  5. Tune the decision threshold for the best model.
  6. Save the best model + metrics + evaluation + explainability figures.

Author: Naveen
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# LightGBM emits a benign feature-name warning during permutation importance
# (sklearn passes a NumPy view of a named DataFrame). It does not affect results.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Allow running both as `python src/main.py` and from inside src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data, clean_data, get_feature_target
from feature_engineering import add_engineered_features
from model_training import (
    train_and_compare,
    save_artifacts,
    get_best_threshold,
    text_report,
)
import visualization as viz
import explainability as expl

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"


def main(tune: bool = True):
    print("=" * 64)
    print("Telco Customer Churn — End-to-End Pipeline")
    print(f"(hyperparameter tuning: {'ON' if tune else 'OFF'})")
    print("=" * 64)

    # 1. Load + clean -------------------------------------------------------
    print("\n[1/6] Loading and cleaning data ...")
    raw = load_data(RAW_PATH)
    clean = clean_data(raw)
    print(f"    Rows: {len(clean):,} | Churn rate: {clean['Churn'].mean():.3f}")

    # 2. EDA figures (use cleaned data with original categorical columns) ---
    print("\n[2/6] Generating EDA figures ...")
    viz.plot_churn_balance(clean)
    viz.plot_churn_by_contract(clean)
    viz.plot_tenure_distribution(clean)
    viz.plot_correlation_heatmap(clean)
    print("    Saved EDA figures to reports/figures/")

    # 3. Feature engineering ------------------------------------------------
    print("\n[3/6] Engineering features ...")
    X, y = get_feature_target(clean)
    X = add_engineered_features(X)
    print(f"    Feature count after engineering: {X.shape[1]}")

    # 4. Train + cross-validate + tune + compare ----------------------------
    print("\n[4/6] Training, cross-validating and comparing models ...")
    if tune:
        print("    (RandomizedSearchCV tuning enabled — this may take a few minutes)")
    results, fitted, (X_train, X_test, y_train, y_test) = train_and_compare(X, y, tune=tune)

    header = (
        f"\n    {'Model':<22}{'Acc':>7}{'Prec':>8}{'Rec':>8}"
        f"{'F1':>8}{'AUC':>8}{'CV-AUC':>9}{'Thr':>7}"
    )
    print(header)
    for r in results:
        cv = f"{r.cv_roc_auc_mean:.3f}±{r.cv_roc_auc_std:.3f}"
        print(
            f"    {r.name:<22}{r.accuracy:>7}{r.precision:>8}"
            f"{r.recall:>8}{r.f1:>8}{r.roc_auc:>8}{cv:>9}{r.threshold:>7}"
        )

    # 5. Persist artifacts + evaluation figures -----------------------------
    print("\n[5/6] Saving best model and evaluation figures ...")
    best_name, _ = save_artifacts(results, fitted)
    best_pipe = fitted[best_name]
    thr = get_best_threshold(results, best_name)

    viz.plot_model_comparison(results)
    viz.plot_confusion_matrix(best_pipe, X_test, y_test, best_name, threshold=thr)
    viz.plot_roc_curve(best_pipe, X_test, y_test, best_name)
    viz.plot_precision_recall_curve(best_pipe, X_test, y_test, best_name)
    viz.plot_threshold_tuning(best_pipe, X_test, y_test, thr, best_name)
    viz.plot_feature_importance(best_pipe, best_name)

    # 6. Explainability (SHAP + permutation importance) ---------------------
    print("\n[6/6] Computing explainability (SHAP + permutation importance) ...")
    expl.permutation_importance_plot(best_pipe, X_test, y_test, best_name)
    shap_path = expl.shap_summary_plot(best_pipe, X_train, X_test, best_name)
    if shap_path is None:
        print("    SHAP plot skipped (explainer unavailable for this model).")
    else:
        print(f"    SHAP summary saved to {shap_path}")

    print(f"\n    Best model: {best_name}  (decision threshold = {thr:.3f})")
    print("\n" + text_report(best_pipe, X_test, y_test, threshold=thr))

    print("\nTop churn drivers (permutation importance):")
    drivers = expl.top_drivers_table(best_pipe, X_test, y_test, top_n=8)
    print(drivers.to_string(index=False))

    print("\nDone. Model -> models/churn_model.joblib | Metrics -> reports/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco churn end-to-end pipeline")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip RandomizedSearchCV hyperparameter tuning for a quick run.",
    )
    args = parser.parse_args()
    main(tune=not args.fast)
