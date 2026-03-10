# Polish Company Bankruptcy Prediction

**MSIN0097 — Predictive Analytics | Individual Coursework**
**UCL MSc Business Analytics | March 2026**

---

## Project Overview

An end-to-end machine learning pipeline predicting corporate bankruptcy for Polish companies using 64 named financial ratio features (mapped from UCI dataset documentation). The project follows six structured steps: problem framing, EDA, data preparation, model exploration, fine-tuning & evaluation, and final solution presentation.

**Final model:** XGBoost (tuned via RandomizedSearchCV)
**Test set performance:** AUC-ROC = 0.9569 | F1-Score = 0.7454 | Precision = 88.0% | Recall = 64.6%

---

## Repository Structure

```
polish-bankruptcy-prediction/
├── 1year.arff          # Raw data — Year 1 (cohort)
├── 2year.arff          # Raw data — Year 2
├── 3year.arff          # Raw data — Year 3
├── 4year.arff          # Raw data — Year 4
├── 5year.arff          # Raw data — Year 5
├── main_analysis.ipynb # Full pipeline — Steps 1–6
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

All output files (plots, saved models, preprocessed arrays) are generated automatically when the notebook is executed.

---

## Data

The dataset is the **Polish Companies Bankruptcy** dataset by Zieba, Tomczak & Tomczak (2016), sourced from the UCI Machine Learning Repository.

- 43,405 firm-year observations across 5 annual cohorts (years 1–5)
- 64 named financial ratio features (liquidity, leverage, profitability, turnover)
- Binary target: `class` — 1 = bankrupt, 0 = survived
- Class imbalance: ~96:4 (survived:bankrupt)

The five `.arff` files are included in this repository. No additional download is required.

---

## Environment Setup

**Python version:** 3.13

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install all dependencies
pip install -r requirements.txt
```

**macOS note — XGBoost / libomp:**
XGBoost requires the OpenMP runtime. If you encounter a `libxgboost.dylib` error:
```bash
brew install libomp   # requires Homebrew: https://brew.sh
```
If Homebrew is unavailable, `libomp.dylib` is bundled with scikit-learn at `<site-packages>/sklearn/.dylibs/libomp.dylib` — copy or symlink it to XGBoost's `lib/` directory.

---

## How to Run

The entire pipeline is contained in a single notebook. Run it from the project root:

```bash
jupyter nbconvert --to notebook --execute main_analysis.ipynb \
  --output main_analysis_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

Or open interactively:
```bash
jupyter notebook main_analysis.ipynb
```

**Expected runtime:** ~10–15 minutes (RandomizedSearchCV with 40 configurations dominates).

The notebook will generate all output plots and save the tuned model automatically.

---

## Pipeline Summary

| Step | Description |
|---|---|
| 1 | Load all 5 ARFF files, define target, set evaluation metrics, enforce temporal split |
| 2 | EDA — class imbalance, missingness, feature distributions, outliers, correlations |
| 3 | Preprocessing — winsorise, median impute, SMOTE (training only), StandardScaler |
| 4 | Train Logistic Regression, Random Forest, Neural Network, XGBoost; shortlist top 2 |
| 5 | Tune XGBoost via RandomizedSearchCV; sweep threshold; error analysis; calibration check |
| 6 | Model selection rationale; feature importance (gain-based); SHAP analysis; model card |

---

## Reproducibility

All random seeds are set to `42` throughout (NumPy, scikit-learn, XGBoost, TensorFlow/Keras). Results may vary slightly across hardware due to floating-point non-determinism in parallel tree construction.

---

## Reference

Zieba, M., Tomczak, S.K. and Tomczak, J.M. (2016) 'Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction', *Expert Systems with Applications*, 58, pp. 93–101.
