# Project Summary — Telco Customer Churn Prediction

**One-line:** An end-to-end ML project that predicts telecom customer churn and turns
the model into a prioritized retention action list.

| | |
|---|---|
| **Domain** | Telecom / Business Analytics |
| **Problem type** | Binary classification (imbalanced, ~26.5% churn) |
| **Dataset** | IBM Telco Customer Churn — 7,043 rows × 21 cols (public) |
| **Best model** | Gradient Boosting — ROC-AUC **0.843**, Accuracy **0.797** |
| **Skills shown** | Data cleaning, EDA, feature engineering, model comparison, evaluation, business interpretation, clean/modular code, documentation |
| **Stack** | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter |

## What I did
1. Cleaned the raw data (fixed `TotalCharges` text/blanks, encoded target).
2. Ran EDA: class balance, churn by contract, tenure, correlations.
3. Engineered features (`tenure_group`, `avg_monthly_spend`, `has_streaming`) and built
   a leak-free preprocessing pipeline.
4. Trained and compared Logistic Regression, Random Forest, and Gradient Boosting.
5. Evaluated with ROC-AUC, precision, recall, F1; saved the best pipeline + figures.
6. Translated churn drivers into concrete retention recommendations.

## Top churn drivers
Short tenure · month-to-month contracts · fiber-optic internet · electronic-check
payments · missing tech-support/security add-ons.

## Interview talking points
- Why ROC-AUC over accuracy for imbalanced churn.
- Why preprocessing lives inside a `ColumnTransformer`/`Pipeline` (no leakage).
- The precision/recall trade-off and how the operating threshold maps to business cost.
- How the model output becomes a monthly, ranked retention campaign.
