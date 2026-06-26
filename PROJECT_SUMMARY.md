# Project Summary — Telco Customer Churn Prediction

**One-line:** An end-to-end ML project that predicts telecom customer churn, compares
five tuned models with cross-validation, explains its decisions with SHAP, and turns the
model into a prioritized retention action list.

| | |
|---|---|
| **Domain** | Telecom / Business Analytics |
| **Problem type** | Binary classification (imbalanced, ~26.5% churn) |
| **Dataset** | IBM Telco Customer Churn — 7,043 rows × 21 cols (public) |
| **Models** | Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost, LightGBM |
| **Best model** | Gradient Boosting — ROC-AUC **0.847**, Recall **0.74** (at tuned threshold 0.32) |
| **Validation** | Stratified 80/20 hold-out + 5-fold cross-validation |
| **Tuning** | RandomizedSearchCV (ROC-AUC) + decision-threshold optimization (F1) |
| **Explainability** | SHAP global importance + permutation importance |
| **Skills shown** | Data cleaning, EDA, feature engineering, multi-model comparison, CV, hyperparameter tuning, imbalance handling, threshold tuning, explainability, business interpretation, clean/modular code, documentation |
| **Stack** | Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn, Jupyter |

## What I did
1. Cleaned the raw data (fixed `TotalCharges` text/blanks, encoded target).
2. Ran EDA: class balance, churn by contract, tenure, correlations.
3. Engineered features (`tenure_group`, `avg_monthly_spend`, `has_streaming`,
   `num_addon_services`) inside a leak-free impute/scale/encode pipeline.
4. Trained and compared five models with 5-fold cross-validation.
5. Tuned the boosting/forest models with RandomizedSearchCV; handled imbalance with
   class weights / `scale_pos_weight`.
6. Tuned the decision threshold to maximize F1 (boosting recall 0.49 → 0.74).
7. Explained the best model with SHAP + permutation importance.
8. Translated churn drivers into concrete retention recommendations.

## Headline result
Five models land within ~0.84–0.85 ROC-AUC (overlapping CV bands). Gradient Boosting is
selected on test ROC-AUC and, after threshold tuning, catches **~74% of churners** — the
business-relevant outcome — while Logistic Regression remains a strong, interpretable
baseline.

## Top churn drivers (SHAP + permutation importance agree)
Short tenure · month-to-month contracts · fiber-optic internet · electronic-check
payments · missing tech-support/security add-ons.

## Interview talking points
- Why ROC-AUC over accuracy for imbalanced churn; why CV *and* a hold-out test set.
- Why preprocessing lives inside a `ColumnTransformer`/`Pipeline` (no leakage).
- The precision/recall trade-off and how threshold tuning maps to business cost.
- Why SHAP and permutation importance together build trust in the explanation.
- How the model output becomes a monthly, ranked retention campaign.
