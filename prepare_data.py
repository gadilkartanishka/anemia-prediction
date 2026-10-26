"""
prepare_data.py - Step 2: turn the huge NFHS-5 women's file into a small clean table.

WHAT it does
  1. Reads only the ~25 columns we need from the 5,972 in the file (in chunks, because the file is 5 GB).
  2. Builds the target:  anemic = 1 if v457 is severe/moderate/mild, 0 if not anemic.
  3. Recodes DHS special codes into proper numbers (BMI x100, food-frequency scale, etc.).
  4. Prints sanity checks so we can see whether the data looks right.
  5. Saves anemia_clean.csv.

Usage:
    python3 prepare_data.py /full/path/to/IAIR7EFL.DTA
"""
import sys
import time
import numpy as np
import pandas as pd

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "synthetic_IAIR7E.dta"
OUT_PATH = "anemia_clean.csv"
CHUNK_ROWS = 2000

# ---- Columns we need (all names confirmed against the real file with inspect_columns.py) ----
FEATURE_COLS = [
    "v012",   # age
    "v025",   # urban / rural
    "v106",   # education level
    "v190",   # wealth index
    "v130",   # religion
    "v024",   # state
    "v136",   # household size
    "v201",   # children ever born
    "v213",   # currently pregnant
    "v404",   # currently breastfeeding
    "v445",   # BMI x 100
]
DIET_COLS = [  # NFHS-5 food frequency: 0 never, 1 daily, 2 weekly, 3 occasionally
    "s731a", "s731b", "s731c", "s731d", "s731e", "s731f", "s731g", "s731h", "s731i",
]
OTHER_COLS = ["v005"]          # survey weight (used later as sample_weight, NOT as a feature)
LABEL_COL = ["v457"]           # anemia level -> used ONLY to make the label, then dropped
WANTED = FEATURE_COLS + DIET_COLS + OTHER_COLS + LABEL_COL

# ---- Find the real (case-sensitive) column names in the file header ----
header = pd.read_stata(DATA_PATH, iterator=True, convert_categoricals=False)
actual_names = {name.lower(): name for name in header.variable_labels().keys()}
if hasattr(header, "close"):      # older pandas versions have no close()
    header.close()
missing = [c for c in WANTED if c not in actual_names]
if missing:
    sys.exit(f"These columns are not in the file: {missing}. Tell Claude which ones.")
to_read = [actual_names[c] for c in WANTED]

# ---- Read in chunks, keeping only our columns ----
t0 = time.time()
pieces = []
reader = pd.read_stata(DATA_PATH, columns=to_read, convert_categoricals=False,
                       chunksize=CHUNK_ROWS, iterator=True)
for i, chunk in enumerate(reader, start=1):
    chunk.columns = [c.lower() for c in chunk.columns]
    pieces.append(chunk[WANTED])
    if i % 50 == 0:
        print(f"  ...read {i * CHUNK_ROWS:,} rows ({time.time() - t0:.0f}s)", flush=True)
df = pd.concat(pieces, ignore_index=True)
print(f"\nLoaded {len(df):,} women, {df.shape[1]} columns in {time.time() - t0:.0f}s")
print("(The survey documentation says 724,115 women were interviewed.)")

# ---- Target ----
n_total = len(df)
df = df[df["v457"].isin([1, 2, 3, 4])].copy()
print(f"\nWomen with a hemoglobin result (v457): {len(df):,}  "
      f"(dropped {n_total - len(df):,} with no result)")
df["anemic"] = df["v457"].isin([1, 2, 3]).astype(int)      # yes/no version
# Level version, ordered by severity: 0 not anemic, 1 mild, 2 moderate, 3 severe
# (DHS codes them the other way round: 1 severe ... 4 not anemic)
df["anemia_level"] = df["v457"].map({4: 0, 3: 1, 2: 2, 1: 3})
df = df.drop(columns=["v457"])      # label source removed so it cannot leak into features

# ---- Recodes ----
out = pd.DataFrame({
    "age": df["v012"],
    "urban": (df["v025"] == 1).astype(int),
    "education": df["v106"],                       # 0 none, 1 primary, 2 secondary, 3 higher
    "wealth_index": df["v190"],                    # 1 poorest ... 5 richest
    "religion": df["v130"],
    "state": df["v024"],
    "household_size": df["v136"],
    "children_ever_born": df["v201"],
    "pregnant": df["v213"].map({0: 0, 1: 1}),
    # breastfeeding is only asked of women with a recent birth; blank means "no"
    "breastfeeding": df["v404"].map({0: 0, 1: 1}).fillna(0),
    # BMI is stored x100 (2250 = 22.5); 9998 means flagged/no valid BMI -> missing
    "bmi": df["v445"].where(df["v445"] < 9000) / 100,
})

# Food frequency: raw codes are NOT in order of how often (1 daily, 2 weekly, 3 occasionally).
# Recode to an ordered score: never 0, occasionally 1, weekly 2, daily 3.
diet_names = {
    "s731a": "milk_curd", "s731b": "pulses", "s731c": "green_veg", "s731d": "fruits",
    "s731e": "eggs", "s731f": "fish", "s731g": "meat", "s731h": "fried_food",
    "s731i": "aerated_drinks",
}
for raw, new in diet_names.items():
    out[new] = df[raw].map({0: 0, 3: 1, 2: 2, 1: 3})

out["weight"] = df["v005"] / 1_000_000
out["anemic"] = df["anemic"]
out["anemia_level"] = df["anemia_level"]       # TARGET (4 levels). 'anemic' is the same info collapsed.

# ---- Sanity checks (read these!) ----
print("\n=== Missing values per column ===")
miss = out.isna().sum()
print(miss[miss > 0].to_string() if miss.any() else "none")

print("\n=== Missing BMI, split by pregnancy ===")
print(out.groupby("pregnant")["bmi"].apply(lambda s: f"{s.isna().mean():.1%} missing").to_string())

print(f"\n=== Anemia prevalence ===")
print(f"Unweighted: {out['anemic'].mean():.1%}")
print(f"Weighted (survey weights): {np.average(out['anemic'], weights=out['weight']):.1%}")

print("\n=== Share of women at each anemia level (0 none, 1 mild, 2 moderate, 3 severe) ===")
print(out["anemia_level"].value_counts(normalize=True).sort_index().round(4).to_string())

print("\n=== Prevalence by wealth index (expect it to fall as wealth rises) ===")
print(out.groupby("wealth_index")["anemic"].mean().round(3).to_string())
print("\n=== Prevalence by education ===")
print(out.groupby("education")["anemic"].mean().round(3).to_string())
print("\n=== Prevalence by pregnancy ===")
print(out.groupby("pregnant")["anemic"].mean().round(3).to_string())

assert not {"v453", "v455", "v456", "v457", "v458"} & set(out.columns), "Label leakage!"
out.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}: {out.shape[0]:,} rows x {out.shape[1]} columns")