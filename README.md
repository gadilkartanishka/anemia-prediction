# Predicting Anemia in Indian Women (NFHS-5)

A machine-learning analysis of anemia among Indian women aged 15–49, using data from the **National Family Health Survey (NFHS-5, 2019–21)** — covering ~690,000 women across all 36 states and union territories.

## What This Project Does

1. **Trains a logistic regression model** to predict anemia level (none / mild / moderate / severe) from demographic, socioeconomic, and nutritional factors.
2. **Computes adjusted odds ratios** with 95% confidence intervals for every predictor — answering *"which factors are most strongly linked to anemia?"*
3. **Tests whether nutritional factors** (BMI + 9 food-frequency scores) add predictive power beyond demographics alone.

## Data

The project uses the **NFHS-5 Individual Women's Recode** (`IAIR7EFL.DTA`), available from [dhsprogram.com](https://dhsprogram.com) after registration. The raw file is ~5 GB and is not included in this repo.

To generate the cleaned dataset:

```bash
python3 prepare_data.py /path/to/IAIR7EFL.DTA
```

This produces `anemia_clean.csv` (~690k rows × 23 columns), which is used by the other two scripts.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate the clean CSV (requires the raw DHS file)
python3 prepare_data.py /path/to/IAIR7EFL.DTA

# Step 2: Train and evaluate the model
python3 train.py

# Step 3: Compute odds ratios and compare feature sets
python3 interpret.py
```

## Pipeline

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `prepare_data.py` | Clean & recode the raw DHS data | `IAIR7EFL.DTA` | `anemia_clean.csv` |
| `train.py` | Train logistic regression, evaluate on held-out test set | `anemia_clean.csv` | Metrics + `figures/fig5_confusion.png` |
| `interpret.py` | Odds ratios (hand-coded Newton's method) + feature-set comparison | `anemia_clean.csv` | `odds_ratios.csv` + `figures/fig6_odds_ratios.png` |

## Features Used

### Demographic & Socioeconomic
- Age, urban/rural residence, education level, wealth quintile
- Religion, state, household size

### Reproductive & Body
- Children ever born, currently pregnant, currently breastfeeding, BMI

### Nutritional (NFHS-5 food-frequency questions)
- 9 food items scored 0 (never) to 3 (daily): milk/curd, pulses, green leafy vegetables, fruits, eggs, fish, meat, fried food, aerated drinks

## Target Variable

Anemia level based on altitude-adjusted hemoglobin (WHO cut-offs):

| Level | Code | Hemoglobin |
|-------|------|------------|
| Not anemic | 0 | ≥ 12.0 g/dL (≥ 11.0 if pregnant) |
| Mild | 1 | 11.0–11.9 (10.0–10.9 if pregnant) |
| Moderate | 2 | 8.0–10.9 (7.0–9.9 if pregnant) |
| Severe | 3 | < 8.0 (< 7.0 if pregnant) |

## Why Logistic Regression?

The goal is **interpretability**, not maximum prediction accuracy. Logistic regression produces **odds ratios** — a clinically meaningful way to say *"women with higher education have 0.87× the odds of anemia, holding everything else constant."* A black-box model (random forest, XGBoost) might score a few AUC points higher but wouldn't produce these interpretable, publishable effect sizes.

The odds ratios are computed using a **hand-coded Newton's method** (not scikit-learn) because scikit-learn's solver doesn't expose the Hessian matrix needed for standard errors and confidence intervals.

## Dependencies

- Python 3.8+
- numpy, pandas, matplotlib, scikit-learn, scipy

```bash
pip install -r requirements.txt
```
