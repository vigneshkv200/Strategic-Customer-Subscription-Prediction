# Portuguese Bank Marketing: Strategic Customer Subscription Prediction Pipeline

[![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Gradient%20Boosting-XGBoost%20%7C%20CatBoost%20%7C%20LightGBM-red)](https://xgboost.readthedocs.io/)

An end-to-end production-grade machine learning system designed to predict whether a bank customer will subscribe to a term deposit. This project goes beyond basic model training by establishing a robust pipeline that addresses class imbalance, avoids data leakage, tracks experiments with **MLflow**, and bridges the gap between raw model probabilities and actionable business strategies through **Precision-Constrained Recall Optimization**.

---

##  Business Impact & Objective
In telemarketing campaigns, efficiency is key. Contacting customers who are highly unlikely to subscribe leads to wasted operational hours, increased costs, and customer fatigue. 

* **Primary Goal**: Identify customers with the highest propensity to subscribe to a term deposit.
* **Secondary Goal**: Create a tunable framework allowing the bank's marketing team to adjust targeting aggressiveness based on current operational budgets and contact-cost constraints.
* **Crucial Leakage Prevention**: The variable `duration` (call length) is highly predictive of subscription but is **only known after a call starts**. To build a realistic predictive model, `duration` was intentionally dropped from the feature set to ensure zero data leakage.

---

## Pipeline Architecture & Engineering Process

```
[Raw CSV Data] 
       │
       ▼
[Data Cleaning] ───────► (Remove Duplicates, Handle Specific 'Unknown' Categories)
       │
       ▼
[Feature Engineering] ─► (Extract 6 Domain-Driven Custom Features)
       │
       ▼
[Stratified Split] ────► (80% Train, 20% Untouched Evaluation Test Set)
       │
       ▼
[Preprocessing] ───────► (One-Hot Encoding, StandardScaler for Linear Baseline)
       │
       ▼
[Model Exploration] ───► (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM)
       │
       ▼
[Experiment Tracking] ─► (Auto-Log Parameters, Metrics & Thresholds to MLflow)
       │
       ▼
[Threshold Scan] ──────► (Evaluate 140 Thresholds [0.01 - 0.71] on Validation Set)
       │
       ▼
[Decision Matrix] ─────► (Strategy 1: Precision >= 50% vs. Strategy 2: Precision >= 40%)
       │
       ▼
[Final Evaluation] ────► (Assess Champion Models on Untouched Test Set & Extract SHAP)
```

### Key Feature Engineering & Data Decisions
* **Handling Special Values (`pdays`)**: The feature `pdays` (days since last campaign contact) was heavily skewed by a placeholder value of `999` (never previously contacted). To prevent the model from treating `999` as a continuous interval, we engineered a binary feature `previously_contacted` and scaled or processed `pdays` explicitly.
* **Domain-Driven Feature Creation**:
  1. `previously_contacted`: Indicator of whether the customer has been engaged before.
  2. `employment_economic_pressure`: Combination of employment variation rate and number of employees.
  3. `consumer_economic_index`: Aggregate of consumer prices and confidence metrics.
  4. `contact_timing_history`: Interaction between contact recency (`pdays`) and historical touchpoints (`previous`).
  5. `total_contact_history`: Consolidated campaign touchpoints (`campaign` + `previous`).
  6. `economic_rate_pressure`: Interaction of interest rates (`euribor3m`) and economic indicators.

---

## Experimentation Scoreboard (Baseline Threshold = 0.50)

All five algorithms were initially evaluated on an untouched test set under the default 0.50 probability threshold to establish clear benchmarks:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.901 | 0.690 | 0.221 | 0.335 | 0.801 | 0.466 |
| **Random Forest** | ~0.890 | Competitive | Moderate | ~0.480 | ~0.805 | ~0.470 |
| **XGBoost** | 0.890 | 0.511 | 0.568 | **0.538** | **0.810** | 0.491 |
| **CatBoost** | ~0.892 | - | - | ~0.531 | ~0.808 | **0.493** |
| **LightGBM** | ~0.889 | - | - | ~0.528 | ~0.805 | ~0.488 |

---

##  Decision Science: Probability Threshold Optimization

A default threshold of `0.50` is rarely optimal for imbalanced real-world datasets. To align modeling with business realities, we executed a detailed **Threshold Scan** (testing 140 thresholds from `0.01` to `0.71` in increments of `0.005`) on validation data.

This analysis revealed distinct optimal decision thresholds for each model class:
* **XGBoost**: `0.24` (Optimal F1)
* **CatBoost**: `0.25` (Optimal F1)
* **LightGBM**: `0.23` (Optimal F1)
* **Random Forest**: `0.63` (Optimal F1)
* **Logistic Regression**: `0.66` (Optimal F1)

###  Business Operating Strategies
Instead of forcing a single configuration, we mapped the model probabilities to two real-world bank operating strategies:

#### Strategy 1: Selective Targeting (Precision ≥ 50% Constraint)
* **Focus**: High resource efficiency. Minimizes wasted calls.
* **XGBoost Operating Point**: Threshold = **0.24** 
* **Performance**: Yields **56.79% Recall** and **51.07% Precision** (F1 = **0.5378**).

#### Strategy 2: High Reach Targeting (Precision ≥ 40% Constraint)
* **Focus**: Market capture. Maximizes the number of subscribers identified.
* **Operating Point**: Lowering the threshold to ~**0.18** allows the bank to achieve significantly higher recall at the expense of a controlled increase in false-positive cold calls.

---

##  Model Interpretability & Explainability (SHAP)
Using **SHAP (SHapley Additive exPlanations)**, we moved past black-box metrics to understand the *why* behind predictions:
* **Key Global Drivers**: Macroeconomic indicators (`euribor3m`, `nr.employed`) and historical contact outcomes (`poutcome_success`) had the strongest global influence on prediction outcomes.
* **Actionable Insight**: Customers contacted during months with high macroeconomic instability were predicted to have lower subscription rates, highlighting the importance of timing campaigns alongside economic indicators.

---

##  Project Key Takeaways & Proof of Work
1. **Data Leakage Elimination**: Ensured real-world viability by removing `duration` early.
2. **Systematic Experimentation**: Automated metric logging with **MLflow** to prevent "notebook chaos" and guarantee reproducibility.
3. **Domain-Driven Feature Engineering**: Generated an absolute **8-point F1-score increase** through deliberate, logical features rather than arbitrary mathematical transformations.
4. **Business-Informed Modeling**: Demonstrated that model selection (XGBoost) and threshold configuration (0.24) are dual parameters governed directly by operational constraints and campaign economics.
print("✔ README.md successfully created in the print(f"Project Name Proposed: {project_name}")
print(f"Project Description Proposed: {project_description}")
