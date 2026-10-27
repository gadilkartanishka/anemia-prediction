# Predicting Anemia in Indian Women (NFHS-5)

A mini ML project that predicts whether a woman is **anemic or not** based on her demographic, socioeconomic, and nutritional factors — using survey data from ~690,000 Indian women.

## Problem Statement

**Anemia risk prediction using nutritional and demographic factors.**

Over half of Indian women aged 15–49 are anemic. Using data from the National Family Health Survey (NFHS-5, 2019–21), this project trains a logistic regression model to:
1. Predict anemia risk from a woman's profile (age, wealth, education, diet, BMI, etc.)
2. Identify which factors are most strongly linked to anemia
3. Test whether nutritional factors add predictive value beyond demographics

## Data

The project uses the **NFHS-5 Individual Women's Recode** from [dhsprogram.com](https://dhsprogram.com). The raw file (~5 GB) is not included — run `prepare_data.py` to generate the cleaned dataset.

**Features used:** age, urban/rural, education, wealth index, religion, state, household size, children ever born, pregnancy status, breastfeeding status, BMI, and 9 food-frequency scores (milk, pulses, green vegetables, fruits, eggs, fish, meat, fried food, aerated drinks).

## Quick Start

```bash
pip install -r requirements.txt

# Generate clean CSV (requires raw DHS file):
python3 prepare_data.py /path/to/IAIR7EFL.DTA

# Train and evaluate the model:
python3 model.py
```

## Pipeline

| Script | What it does | Output |
|--------|-------------|--------|
| `prepare_data.py` | Cleans raw survey data → CSV | `anemia_clean.csv` |
| `model.py` | Trains logistic regression, evaluates, shows risk factors | Metrics + 2 figures |

## Output

`model.py` produces:
- **Test-set metrics**: accuracy, AUC, precision, recall, F1
- **`figures/confusion_matrix.png`**: where the model gets it right/wrong
- **`figures/feature_importance.png`**: odds ratios showing which factors increase or decrease anemia risk
- **Feature-set comparison**: demographics only vs nutrition only vs both combined

## Dependencies

```bash
pip install -r requirements.txt
```

Python 3.8+ · numpy · pandas · matplotlib · scikit-learn
