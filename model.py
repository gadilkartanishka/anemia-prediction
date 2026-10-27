"""
model.py - Train and interpret a logistic regression model for anemia risk prediction.

Predicts whether a woman is anemic (yes/no) based on her demographic, socioeconomic,
and nutritional factors using data from the NFHS-5 survey (~690,000 women).

Steps:
  1. Load the cleaned dataset and split into 80% train / 20% test.
  2. Preprocess features (scale numeric, one-hot encode categorical).
  3. Train a logistic regression model (scikit-learn).
  4. Evaluate on the test set: accuracy, AUC, confusion matrix.
  5. Show which factors increase or decrease anemia risk (odds ratios).
  6. Compare feature sets: do nutritional factors improve prediction?

Usage:   python3 model.py
"""
import os
import sys
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ---- Configuration ----
CSV = sys.argv[1] if len(sys.argv) > 1 else "anemia_clean.csv"
SEED = 42
os.makedirs("figures", exist_ok=True)

# ---- Feature groups ----
DIET = ["milk_curd", "pulses", "green_veg", "fruits", "eggs", "fish",
        "meat", "fried_food", "aerated_drinks"]
NUMERIC = ["age", "household_size", "children_ever_born", "bmi",
           "education", "wealth_index"] + DIET
BINARY = ["urban", "pregnant", "breastfeeding"]
CATEGORICAL = ["state", "religion"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# ---- Chart styling ----
BLUE, RED = "#2a78d6", "#e6553a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

# =====================================================================
# LOAD DATA
# =====================================================================
df = pd.read_csv(CSV)
df = df.dropna(subset=FEATURES + ["anemic"])
X = df[FEATURES]
y = df["anemic"].astype(int)
print(f"Dataset: {len(df):,} women")
print(f"  Anemic:     {y.sum():,} ({y.mean():.1%})")
print(f"  Not anemic: {(1 - y).sum():,} ({(1 - y).mean():.1%})")

# =====================================================================
# STEP 1: Train / test split
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain set: {len(X_train):,} women")
print(f"Test set:  {len(X_test):,} women")


def make_pipeline(num_cols, bin_cols, cat_cols):
    """Create a preprocessing + logistic regression pipeline."""
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if bin_cols:
        transformers.append(("bin", "passthrough", bin_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
    ])


# =====================================================================
# STEP 2: Train the model
# =====================================================================
model = make_pipeline(NUMERIC, BINARY, CATEGORICAL)
model.fit(X_train, y_train)
print("\nModel trained.")

# =====================================================================
# STEP 3: Evaluate on the test set
# =====================================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'=' * 55}")
print(f" TEST SET RESULTS")
print(f"{'=' * 55}")
print(f" Baseline (always guess majority class): {baseline_acc:.3f} accuracy")
print(f" Logistic Regression:                    {acc:.3f} accuracy")
print(f" AUC (area under ROC curve):             {auc:.3f}")
print(f"{'=' * 55}")
print(classification_report(y_test, y_pred,
                            target_names=["Not anemic", "Anemic"], digits=3))

# ---- Figure 1: Confusion matrix ----
cm = confusion_matrix(y_test, y_pred, normalize="true") * 100
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
labels = ["Not anemic", "Anemic"]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:.1f}%", ha="center", va="center",
                color="white" if cm[i, j] > 55 else INK, fontsize=16, fontweight="bold")
ax.set_xticks(range(2), labels)
ax.set_yticks(range(2), labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig("figures/confusion_matrix.png", dpi=200)
plt.close(fig)
print("Saved figures/confusion_matrix.png")

# =====================================================================
# STEP 4: Which factors are linked to anemia? (odds ratios)
# =====================================================================
# Get feature names and coefficients from the trained model
prep = model.named_steps["prep"]
names = list(prep.get_feature_names_out())
names = [n.replace("num__", "").replace("bin__", "").replace("cat__", "") for n in names]
coefs = model.named_steps["clf"].coef_[0]
odds_ratios = np.exp(coefs)

# Human-readable labels for the main features (skip state dummies)
NICE = {
    "age": "Age", "bmi": "BMI", "urban": "Urban (vs rural)",
    "education": "Education level", "wealth_index": "Wealth index",
    "household_size": "Household size", "children_ever_born": "Children ever born",
    "pregnant": "Currently pregnant", "breastfeeding": "Currently breastfeeding",
    "milk_curd": "Milk / curd", "pulses": "Pulses / lentils",
    "green_veg": "Green leafy vegetables", "fruits": "Fruits",
    "eggs": "Eggs", "fish": "Fish", "meat": "Meat / chicken",
    "fried_food": "Fried food", "aerated_drinks": "Aerated drinks",
}

or_df = pd.DataFrame({"feature": names, "odds_ratio": odds_ratios})
show = or_df[or_df["feature"].isin(NICE)].copy()
show["label"] = show["feature"].map(NICE)
show = show.sort_values("odds_ratio")

# ---- Figure 2: Odds ratio chart ----
fig, ax = plt.subplots(figsize=(8, 7))
colors = [BLUE if v >= 1 else RED for v in show["odds_ratio"]]
bars = ax.barh(show["label"], show["odds_ratio"], color=colors, height=0.6)
ax.axvline(1, color=MUTED, linewidth=1.2, linestyle="--")
for i, (_, row) in enumerate(show.iterrows()):
    offset = 0.01 if row["odds_ratio"] >= 1 else -0.04
    ax.text(row["odds_ratio"] + offset, i, f"{row['odds_ratio']:.2f}",
            va="center", fontsize=9, color=INK)
ax.set_xlabel("Odds Ratio  (blue >1 = higher risk · red <1 = lower risk)")
ax.set_title("Factors linked to anemia risk (adjusted for all other factors)",
             loc="left", fontweight="bold")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/feature_importance.png", dpi=200)
plt.close(fig)
print("Saved figures/feature_importance.png")

# Print the top factors
print(f"\n{'=' * 55}")
print(f" ODDS RATIOS (key factors)")
print(f"{'=' * 55}")
print(" OR > 1 means higher anemia risk; OR < 1 means lower risk.")
print(" (Numeric features are standardised: OR is per 1 SD change.)\n")
for _, row in show.sort_values("odds_ratio", ascending=False).iterrows():
    direction = "↑" if row["odds_ratio"] > 1 else "↓"
    print(f"  {direction}  {row['label']:30s}  OR = {row['odds_ratio']:.3f}")

# =====================================================================
# STEP 5: Does adding nutritional factors improve prediction?
# =====================================================================
DEMO_NUM = ["age", "household_size", "children_ever_born", "education", "wealth_index"]
NUTRI_NUM = ["bmi"] + DIET

groups = {
    "Demographics only":          (DEMO_NUM, BINARY, CATEGORICAL),
    "Nutrition only (BMI + diet)": (NUTRI_NUM, [],     []),
    "Both combined":              (DEMO_NUM + NUTRI_NUM, BINARY, CATEGORICAL),
}

print(f"\n{'=' * 55}")
print(f" FEATURE SET COMPARISON (anemic yes/no)")
print(f"{'=' * 55}")
for name, (num, bin_, cat) in groups.items():
    cols = num + bin_ + cat
    pipe = make_pipeline(num or [], bin_, cat)
    pipe.fit(X_train[cols], y_train)
    p = pipe.predict_proba(X_test[cols])[:, 1]
    a = accuracy_score(y_test, (p >= 0.5).astype(int))
    auc_val = roc_auc_score(y_test, p)
    print(f"  {name:35s}  Accuracy: {a:.3f}   AUC: {auc_val:.3f}")
print(f"  {'Baseline (guess majority)':35s}  Accuracy: {baseline_acc:.3f}   AUC: 0.500")
print()
