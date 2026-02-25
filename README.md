# Credit Card Fraud Detection

## Project Overview
This project studies a binary classification problem: detecting fraudulent credit card transactions.

The work is presented in the notebook [fraud-detection.ipynb](./fraud-detection.ipynb).

## Project Objectives
The main goals are:
- show clear problem framing,
- apply reproducible machine learning methods,
- evaluate models with suitable metrics for imbalanced data.

## Dataset
- Source: Kaggle credit card fraud dataset (loaded from the Kaggle input directory in the notebook runtime)
- Rows: `284,807`
- Columns: `31`
- Target column: `Class` (`0` = non-fraud, `1` = fraud)
- Class imbalance: fraud cases are very rare (`492` rows, about `0.1727%`)

## Method
The notebook follows a full machine learning pipeline:

1. **Setup and reproducibility**
   - fixed random seed,
   - clear global configuration.
2. **Data loading**
   - automatic CSV path detection for Kaggle.
3. **Exploratory data analysis (EDA)**
   - shape, missing values, target ratio, duplicates, and data types.
4. **Data quality checks**
   - leakage checks,
   - duplicate removal.
5. **Preprocessing**
   - median imputation + scaling for numeric features,
   - one-hot encoding for categorical features.
6. **Model training with cross-validation**
   - baseline model: Logistic Regression (`class_weight="balanced"`),
   - stronger model: CatBoost (`auto_class_weights="Balanced"`),
   - split strategy: Stratified 5-fold CV.
7. **Evaluation and model selection**
   - ROC-AUC,
   - PR-AUC (primary metric),
   - F1 score,
   - threshold optimization using out-of-fold predictions.

## Results (from saved notebook outputs)

### Baseline (Logistic Regression, CV mean)
- ROC-AUC: `0.9796`
- PR-AUC: `0.7189`
- F1@0.5: `0.1163`

### Strong Model (CatBoost, CV mean)
- ROC-AUC: `0.9827`
- PR-AUC: `0.7384`
- F1@0.5: `0.4789`

### Out-of-Fold Comparison
- Baseline: ROC-AUC `0.9792`, PR-AUC `0.7165`, F1 `0.1162`
- Strong model: ROC-AUC `0.9643`, PR-AUC `0.7513`, F1 `0.3855`

Selected model in notebook: **strong model** (based on PR-AUC).  
Best F1 threshold on out-of-fold predictions: **0.91**.

## Key Learning Points
- In highly imbalanced problems, PR-AUC is often more informative than accuracy.
- Threshold tuning can strongly change F1 performance.
- Comparing models with several metrics gives a more complete view than using one metric only.

## Limitations
- The notebook currently keeps stratified folds even when time-based validation is requested in settings.
- No final model export or submission file generation step is included yet.

## Reproducibility
The notebook is designed for Kaggle runtime:
- Python 3 environment,
- input data read from `/kaggle/input/...`,
- output files should be written to `/kaggle/working/...`.

Run all cells from top to bottom to reproduce the main results.

## Repository Structure
- [fraud-detection.ipynb](./fraud-detection.ipynb): full analysis and modeling workflow
- [README.md](./README.md): project summary
