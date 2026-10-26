"""
train.py - Step 4: train and evaluate logistic regression on the 4 anemia levels.

WHAT  Learn a rule that takes a woman's age, wealth, diet, BMI etc. and outputs the
      probability of each anemia level (0 none, 1 mild, 2 moderate, 3 severe).
WHY   To measure how well these factors predict anemia, and (next step) which factors matter.
HOW   1. Split the women into a TRAIN set (80%) and a TEST set (20%) that the model never sees
         while learning. Test performance is the honest number.
      2. Prepare the columns (scale numbers, one-hot the categories).
      3. Fit logistic regression on the train set.
      4. Score it on the test set and compare with a "dumb" baseline.

Usage:   python3 train.py anemia_clean.csv
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)

CSV = sys.argv[1] if len(sys.argv) > 1 else "anemia_clean.csv"
SEED = 42
LEVELS = ["Not anemic", "Mild", "Moderate", "Severe"]

DIET = ["milk_curd", "pulses", "green_veg", "fruits", "eggs", "fish", "meat", "fried_food", "aerated_drinks"]
NUMERIC = ["age", "household_size", "children_ever_born", "bmi", "education", "wealth_index"] + DIET
BINARY = ["urban", "pregnant", "breastfeeding"]
CATEGORICAL = ["state", "religion"]          # no natural order -> one-hot
TARGET = "anemia_level"
FEATURES = NUMERIC + BINARY + CATEGORICAL    # NOTE: 'anemic', 'anemia_level' and 'weight' are NOT features

df = pd.read_csv(CSV)
n0 = len(df)
df = df.dropna(subset=FEATURES + [TARGET])
print(f"Rows: {n0:,} -> {len(df):,} after dropping rows with a missing feature value")

X, y = df[FEATURES], df[TARGET].astype(int)

# ---- 1. Train / test split. stratify=y keeps the same level mix in both parts. ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {len(X_train):,} women   Test: {len(X_test):,} women")

# ---- 2. Column preparation (learned from the TRAIN set only, then applied to test) ----
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

models = {
    "Baseline (always guess the most common level)": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=500),
    "Logistic regression (balanced classes)": LogisticRegression(max_iter=500, class_weight="balanced"),
}

rows, fitted = [], {}
for name, clf in models.items():
    t0 = time.time()
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)                                  # 3. learn from the train set
    proba = pipe.predict_proba(X_test)
    pred = proba.argmax(axis=1)                                  # most probable level
    p_anemic = 1 - proba[:, 0]                                   # P(any anemia) = 1 - P(not anemic)
    y_bin = (y_test > 0).astype(int)
    rows.append({
        "model": name,
        "accuracy (4 levels)": accuracy_score(y_test, pred),
        "macro F1 (4 levels)": f1_score(y_test, pred, average="macro"),
        "accuracy (anemic yes/no)": accuracy_score(y_bin, (p_anemic >= 0.5).astype(int)),
        "AUC (anemic yes/no)": roc_auc_score(y_bin, p_anemic) if p_anemic.std() > 0 else 0.5,
    })
    fitted[name] = (pipe, pred)
    print(f"  trained '{name}' in {time.time() - t0:.0f}s", flush=True)

# ---- 4. Results ----
results = pd.DataFrame(rows).set_index("model").round(3)
pd.set_option("display.width", 200)
print("\n=== Test-set results ===")
print(results.to_string())
share_anemic = (y_test > 0).mean()
print(f"\nNOTE: the baseline row guesses 'Not anemic' (the most common of the 4 levels), so its yes/no "
      f"accuracy is misleadingly low. The fair yes/no baseline is 'always say anemic': "
      f"{max(share_anemic, 1 - share_anemic):.3f} accuracy.")

for name in list(models)[1:]:
    print(f"\n=== Per-level results: {name} ===")
    print(classification_report(y_test, fitted[name][1], target_names=LEVELS, digits=3, zero_division=0))

# Confusion matrices: each row = the true level; each cell = % of those women given that prediction
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, name in zip(axes, list(models)[1:]):
    cm = confusion_matrix(y_test, fitted[name][1], labels=[0, 1, 2, 3], normalize="true") * 100
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > 55 else "#0b0b0b")
    ax.set_xticks(range(4), LEVELS, rotation=30); ax.set_yticks(range(4), LEVELS)
    ax.set_xlabel("Predicted level"); ax.set_ylabel("True level")
    ax.set_title(name.replace("Logistic regression", "LogReg"), fontsize=10, loc="left", fontweight="bold")
fig.suptitle("Of the women at each TRUE level (row), % given each predicted level", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig5_confusion.png", dpi=200); plt.close(fig)
print("\nSaved figures/fig5_confusion.png")