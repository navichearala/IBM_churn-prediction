"""
main.py
-------
End-to-end pipeline runner for the Telco Customer Churn project.

Run from the project root:
    python src/main.py

This will:
  1. Load and clean the raw data.
  2. Engineer features.
  3. Generate EDA figures.
  4. Train and compare three models.
  5. Save the best model + metrics + evaluation figures.

Author: Naveen
"""

from __future__ import annotations

import os
import sys

# Allow running both as `python src/main.py` and from inside src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data, clean_data, get_feature_target
from feature_engineering import add_engineered_features
from model_training import train_and_compare, save_artifacts, best_model, text_report
import visualization as viz

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"


def main():
    print("=" * 60)
    print("Telco Customer Churn — End-to-End Pipeline")
    print("=" * 60)

    # 1. Load + clean -------------------------------------------------------
    print("\n[1/5] Loading and cleaning data ...")
    raw = load_data(RAW_PATH)
    clean = clean_data(raw)
    print(f"    Rows: {len(clean):,} | Churn rate: {clean['Churn'].mean():.3f}")

    # 2. EDA figures (use cleaned data with original categorical columns) ---
    print("\n[2/5] Generating EDA figures ...")
    viz.plot_churn_balance(clean)
    viz.plot_churn_by_contract(clean)
    viz.plot_tenure_distribution(clean)
    viz.plot_correlation_heatmap(clean)
    print("    Saved EDA figures to reports/figures/")

    # 3. Feature engineering ------------------------------------------------
    print("\n[3/5] Engineering features ...")
    X, y = get_feature_target(clean)
    X = add_engineered_features(X)
    print(f"    Feature count after engineering: {X.shape[1]}")

    # 4. Train + compare models --------------------------------------------
    print("\n[4/5] Training and comparing models ...")
    results, fitted, (X_train, X_test, y_train, y_test) = train_and_compare(X, y)
    print(f"\n    {'Model':<22}{'Acc':>7}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
    for r in results:
        print(
            f"    {r.name:<22}{r.accuracy:>7}{r.precision:>8}"
            f"{r.recall:>8}{r.f1:>8}{r.roc_auc:>8}"
        )

    # 5. Persist artifacts + evaluation figures -----------------------------
    print("\n[5/5] Saving best model and evaluation figures ...")
    best_name, _ = save_artifacts(results, fitted)
    best_pipe = fitted[best_name]
    viz.plot_confusion_matrix(best_pipe, X_test, y_test, best_name)
    viz.plot_roc_curve(best_pipe, X_test, y_test, best_name)
    viz.plot_feature_importance(best_pipe, best_name)

    print(f"\n    Best model: {best_name}")
    print("\n" + text_report(best_pipe, X_test, y_test))
    print("\nDone. Model -> models/churn_model.joblib | Metrics -> reports/metrics.json")


if __name__ == "__main__":
    main()
