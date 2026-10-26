"""
interpret.py - Step 5: which factors are linked to anemia, and do nutritional factors add anything?

PART A  Odds ratios.  A yes/no logistic regression (anemic vs not) fitted on ALL women.
        An odds ratio (OR) says how the ODDS of anemia change when a factor changes, with all
        the other factors held fixed:  OR = 1  no link,  OR > 1  higher odds,  OR < 1  lower odds.
        We also give a 95% confidence interval (CI): a range of values the true OR plausibly lies in.
        If the CI contains 1, we cannot tell the factor apart from "no effect".

PART B  Do nutritional factors help prediction?  Same model, trained on the same 80% train split as
        train.py, using (1) demographic/reproductive factors only, (2) nutritional factors only
        (BMI + 9 food-frequency scores), (3) both.  Judged by AUC on the 20% test set.

Usage:   python3 interpret.py anemia_clean.csv
Output:  odds_ratios.csv, figures/fig6_odds_ratios.png, and tables printed on screen.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

CSV = sys.argv[1] if len(sys.argv) > 1 else "anemia_clean.csv"
os.makedirs("figures", exist_ok=True)

DIET = ["milk_curd", "pulses", "green_veg", "fruits", "eggs", "fish", "meat", "fried_food", "aerated_drinks"]
ALL_FEATURES = ["age", "household_size", "children_ever_born", "bmi", "education", "wealth_index",
                "urban", "pregnant", "breastfeeding", "state", "religion"] + DIET

df = pd.read_csv(CSV).dropna(subset=ALL_FEATURES + ["anemia_level"])      # same rows as train.py
y = (df["anemia_level"] > 0).astype(int).to_numpy()                       # 1 = any anemia
print(f"Women used: {len(df):,}   share anemic: {y.mean():.1%}")

# =====================================================================
# PART A - odds ratios from a logistic regression (fitted by Newton's method / IRLS)
# =====================================================================
X = pd.DataFrame(index=df.index)
X["age_per5"] = df["age"] / 5
X["urban"] = df["urban"]
X["pregnant"] = df["pregnant"]
X["breastfeeding"] = df["breastfeeding"]
X["children_ever_born"] = df["children_ever_born"]
X["household_size"] = df["household_size"]
X["bmi_per5"] = df["bmi"] / 5
for col in DIET:
    X[col] = df[col]                                    # per one step more frequent (0 never ... 3 daily)
for code, name in {1: "primary", 2: "secondary", 3: "higher"}.items():
    X[f"edu_{name}"] = (df["education"] == code).astype(int)          # reference: no education
for code, name in {2: "poorer", 3: "middle", 4: "richer", 5: "richest"}.items():
    X[f"wealth_{name}"] = (df["wealth_index"] == code).astype(int)    # reference: poorest
rel = df["religion"].map({2: "muslim", 3: "christian", 4: "sikh", 5: "buddhist"})
rel = rel.where(df["religion"] == 1, rel).fillna("other")              # Hindu stays NaN -> filled below
rel = np.where(df["religion"] == 1, "hindu", rel)
for name in ["muslim", "christian", "sikh", "buddhist", "other"]:      # reference: Hindu
    X[f"rel_{name}"] = (rel == name).astype(int)
states = pd.get_dummies(df["state"], prefix="state", drop_first=True).astype(int)   # reference: first state
X = pd.concat([X, states], axis=1)
X = X.loc[:, X.std() > 0]                                # drop any column with no variation
X.insert(0, "intercept", 1.0)
names = list(X.columns)
A = X.to_numpy(dtype=np.float64)


def fit_logit(A, y, max_iter=30, chunk=100_000):
    """Maximum-likelihood logistic regression by Newton's method. Returns coefficients and covariance."""
    n, k = A.shape
    beta = np.zeros(k)
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    for it in range(max_iter):
        grad, H = np.zeros(k), np.zeros((k, k))
        for s in range(0, n, chunk):
            a, t = A[s:s + chunk], y[s:s + chunk]
            p = 1 / (1 + np.exp(-(a @ beta)))
            w = p * (1 - p)
            grad += a.T @ (t - p)
            H += a.T @ (a * w[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(k), grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta, np.linalg.inv(H), it + 1


beta, cov, iters = fit_logit(A, y)
se = np.sqrt(np.diag(cov))
z = beta / se
res = pd.DataFrame({
    "variable": names, "coef": beta, "odds_ratio": np.exp(beta),
    "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se),
    "p_value": 2 * norm.sf(np.abs(z)),
}).set_index("variable")
res.to_csv("odds_ratios.csv")
print(f"Logistic regression converged in {iters} iterations; {len(names) - 1} predictors (incl. state dummies).")

LABELS = {
    "wealth_poorer": "Wealth: poorer (vs poorest)", "wealth_middle": "Wealth: middle (vs poorest)",
    "wealth_richer": "Wealth: richer (vs poorest)", "wealth_richest": "Wealth: richest (vs poorest)",
    "edu_primary": "Education: primary (vs none)", "edu_secondary": "Education: secondary (vs none)",
    "edu_higher": "Education: higher (vs none)",
    "urban": "Urban (vs rural)", "age_per5": "Age, per 5 years", "bmi_per5": "BMI, per 5 units",
    "pregnant": "Currently pregnant", "breastfeeding": "Currently breastfeeding",
    "children_ever_born": "Children ever born, per child", "household_size": "Household size, per person",
    "rel_muslim": "Religion: Muslim (vs Hindu)", "rel_christian": "Religion: Christian (vs Hindu)",
    "rel_sikh": "Religion: Sikh (vs Hindu)", "rel_buddhist": "Religion: Buddhist (vs Hindu)",
    "rel_other": "Religion: other (vs Hindu)",
    "milk_curd": "Milk or curd, per step more often", "pulses": "Pulses or beans, per step more often",
    "green_veg": "Dark green leafy veg, per step more often", "fruits": "Fruits, per step more often",
    "eggs": "Eggs, per step more often", "fish": "Fish, per step more often",
    "meat": "Chicken or meat, per step more often", "fried_food": "Fried food, per step more often",
    "aerated_drinks": "Aerated drinks, per step more often",
}
shown = [v for v in LABELS if v in res.index]
table = res.loc[shown, ["odds_ratio", "ci_low", "ci_high", "p_value"]].copy()
table.index = [LABELS[v] for v in shown]
fmt = table.copy()
fmt["odds_ratio"] = table["odds_ratio"].map("{:.2f}".format)
fmt["95% CI"] = [f"{lo:.2f} - {hi:.2f}" for lo, hi in zip(table["ci_low"], table["ci_high"])]
fmt["p_value"] = table["p_value"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
print("\n=== PART A: odds ratios for ANY anemia (adjusted for all factors and for state) ===")
print(fmt[["odds_ratio", "95% CI", "p_value"]].to_string())
print("\n(State effects are in odds_ratios.csv.)")

# ---- Forest plot ----
BLUE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
n = len(table)
fig, ax = plt.subplots(figsize=(9, 0.32 * n + 1.6))
ypos = np.arange(n)[::-1]
ax.hlines(ypos, table["ci_low"], table["ci_high"], color=BLUE, linewidth=2)
ax.plot(table["odds_ratio"], ypos, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
ax.set_xscale("log")
lo, hi = table["ci_low"].min() * 0.9, table["ci_high"].max() * 1.1
ax.set_xlim(lo, hi * 1.9)
for yp, (_, r) in zip(ypos, table.iterrows()):
    ax.text(hi * 1.12, yp, f"{r.odds_ratio:.2f}  ({r.ci_low:.2f}-{r.ci_high:.2f})", va="center", color=INK, fontsize=8)
ax.set_yticks(ypos, table.index)
ax.set_xticks([0.6, 0.8, 1, 1.25, 1.5], ["0.6", "0.8", "1", "1.25", "1.5"]) if lo < 0.8 and hi > 1.3 else None
ax.set_xlabel("Odds ratio for any anemia (log scale). Right of the dashed line = higher odds, left = lower odds.")
ax.set_title("Factors linked to anemia, adjusted for each other and for state", loc="left", fontweight="bold")
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig6_odds_ratios.png", dpi=200); plt.close(fig)
print("Saved figures/fig6_odds_ratios.png and odds_ratios.csv")

# =====================================================================
# PART B - does adding nutritional factors improve prediction?
# =====================================================================
X_all = df[ALL_FEATURES]
y4 = df["anemia_level"].astype(int)
# Same split as train.py (same rows, stratified on the 4 levels, seed 42)
idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=y4, random_state=42)
yb_train, yb_test = (df.loc[idx_train, "anemia_level"] > 0).astype(int), (df.loc[idx_test, "anemia_level"] > 0).astype(int)

groups = {
    "Demographic & reproductive only": dict(
        num=["age", "household_size", "children_ever_born", "education", "wealth_index"],
        bin=["urban", "pregnant", "breastfeeding"], cat=["state", "religion"]),
    "Nutritional only (BMI + 9 foods)": dict(num=["bmi"] + DIET, bin=[], cat=[]),
    "Both": dict(
        num=["age", "household_size", "children_ever_born", "education", "wealth_index", "bmi"] + DIET,
        bin=["urban", "pregnant", "breastfeeding"], cat=["state", "religion"]),
}
rows = []
fair_baseline = max(yb_test.mean(), 1 - yb_test.mean())
for name, g in groups.items():
    parts = []
    if g["num"]: parts.append(("num", StandardScaler(), g["num"]))
    if g["bin"]: parts.append(("bin", "passthrough", g["bin"]))
    if g["cat"]: parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), g["cat"]))
    pipe = Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(max_iter=500))])
    cols = g["num"] + g["bin"] + g["cat"]
    pipe.fit(X_all.loc[idx_train, cols], yb_train)
    p = pipe.predict_proba(X_all.loc[idx_test, cols])[:, 1]
    rows.append({"feature set": name, "n features": len(cols),
                 "accuracy": accuracy_score(yb_test, p >= 0.5), "AUC": roc_auc_score(yb_test, p)})
print("\n=== PART B: test-set performance for anemic yes/no, by feature set ===")
print(pd.DataFrame(rows).set_index("feature set").round(3).to_string())
print(f"(Fair baseline accuracy, always guess 'anemic': {fair_baseline:.3f}; AUC of guessing: 0.500)")
print("With ~138,000 test women, an AUC gap of about 0.01 or more is a real difference, not noise.")