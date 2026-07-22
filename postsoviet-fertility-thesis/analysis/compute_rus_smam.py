"""
compute_rus_smam.py
-------------------
Compute female SMAM for Russia from the 2021 Census (VPN-2020)
Table 5: Population by age, sex, and marital status.

Uses the standard Hajnal (1953) formula on census counts of women
who have NEVER been married (никогда не состоявшие в браке, column 6)
as a proportion of those who stated marital status (column 2).

Note: the Russian census covers ages 16+, so the 15-19 age group is
approximated by combining the 16-17 and 18-19 rows. See "AGE-15 SENSITIVITY"
below for an explicit, falsifiable arithmetic bound on the resulting error
(replacing an earlier unsupported "<0.1 year" assertion).

AGE-15 SENSITIVITY BOUND (derived, not asserted):
  Hajnal's A = 15 + 5 * sum(p_g for g in the seven age groups). Only the
  15-19 term is affected by the age-16+ census coverage, and only through
  replacing the true 5-value average (p15, p16, p17, p18, p19) with the
  4-value average p1619 = mean(p16..p19) actually observed. The resulting
  error in that term is (1/5) * (p15 - p1619), so the error in A is
  (p15 - p1619), and since B, C, D (the 45-49 terms) are untouched, the
  resulting error in SMAM = (A - D) / C is approximately (p15 - p1619) / C.

  ASSUMPTION: |p15 - p1619| <= 0.02 (the proportion of 15-year-old women
  never married differs from the observed 16-19 average by at most 2
  percentage points). This is a conservative bound given Russia's legal
  minimum marriage age (18, or 16 by regional exception) makes marriage at
  15 a legal impossibility in nearly all cases, so both proportions should
  sit well above 0.95 with a gap far smaller than 2 points in practice.

  Under this assumption, with C = 1 - B typically in the 0.93-0.98 range for
  a SMAM near 24-25 (B is printed at runtime below, so this bound can be
  checked against the actual run), the resulting bound on the SMAM error is
  0.02 / C =~ 0.020-0.022 years -- roughly five times smaller than the
  previously claimed (but undemonstrated) 0.1-year threshold.

Source: Rosstat, Всероссийская перепись населения 2020 (conducted Oct 2021),
        Tom 2, Table 5 (Tоm2_tab5_VPN-2020.xlsx)
Result: Russia female SMAM = 24.86

Run from repo root:  python analysis/compute_rus_smam.py
Reads:               data/raw/rus_census2021_tab5.xlsx
Updates:             data/manual/cultural_vars.csv (Russia row only)
"""

import os

import pandas as pd

df = pd.read_excel("data/raw/rus_census2021_tab5.xlsx",
                   sheet_name="таб 5", header=None)

# Women section starts at row 50 (Женщины в возрасте 16 лет и более)
# Age data rows: 52 (16-17), 53 (18-19), 54 (20-24), ..., 59 (45-49)
# Column 2: stated marital status (denominator)
# Column 6: never married (никогда не состоявшие в браке)
rows = {
    "16-17": 52, "18-19": 53, "20-24": 54, "25-29": 55,
    "30-34": 56, "35-39": 57, "40-44": 58, "45-49": 59
}

# --- Workbook-label validation: verify the expected age-range text is still
# present near each hardcoded row position before trusting df.iloc[r, 2] /
# df.iloc[r, 6]. A reformatted or reordered workbook (inserted/deleted rows,
# a different sheet layout) would otherwise silently return numbers read
# from the wrong row. Scans the first two columns of each row for the
# row's own age-range label (e.g. "16-17"), since a fixed single column
# index for the label was not assumed. ---
def _row_has_age_label(row_idx, label):
    for col in (0, 1):
        cell = df.iloc[row_idx, col]
        if pd.notna(cell) and label in str(cell):
            return True
    return False

label_mismatches = [(label, r) for label, r in rows.items()
                     if not _row_has_age_label(r, label)]
if label_mismatches:
    raise SystemExit(
        "Workbook row layout does not match the expected age-group labels -- "
        f"the sheet may have been reformatted. Mismatches (expected label, "
        f"row index): {label_mismatches}. Update the `rows` map in this "
        "script after confirming the new layout against the source workbook."
    )

data = {}
for label, r in rows.items():
    stated = float(df.iloc[r, 2])
    never  = float(df.iloc[r, 6])
    data[label] = {"stated": stated, "never": never, "prop_single": never / stated}

# Combine 16-17 and 18-19 into 15-19 proxy
prop_15_19 = ((data["16-17"]["never"] + data["18-19"]["never"]) /
              (data["16-17"]["stated"] + data["18-19"]["stated"]))

print("=== Russian Census 2021 (VPN-2020) — Women, marital status ===\n")
print(f"  {'Age group':12s}  {'Stated':>10s}  {'Never married':>14s}  {'Prop single':>12s}")
print(f"  {'15-19*':12s}  {'combined':>10s}  {'combined':>14s}  {prop_15_19:12.4f}")
for label in ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]:
    d = data[label]
    print(f"  {label:12s}  {d['stated']:10.0f}  {d['never']:14.0f}  {d['prop_single']:12.4f}")
print("\n  * 15-19 approximated from 16-17 + 18-19 (census covers 16+)")

# Hajnal SMAM
props = {"15-19": prop_15_19}
for label in ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]:
    props[label] = data[label]["prop_single"]

age_groups = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
missing = [g for g in age_groups if g not in props]
if missing:
    raise SystemExit(f"Missing age groups: {missing}")

A = 15 + 5 * sum(props[g] for g in age_groups)
B = props["45-49"]
C = 1 - B
D = 50 * B
SMAM = (A - D) / C

print(f"\n=== Hajnal SMAM ===")
print(f"  A = {A:.3f}")
print(f"  B (prop never-married at 45-49) = {B:.4f}")
print(f"  C = {C:.4f}")
print(f"  D = {D:.3f}")
print(f"  SMAM = {SMAM:.2f}")
print(f"\n>>> Russia female SMAM (Census VPN-2020, conducted 2021): {SMAM:.2f}")
print(f"    Previous (UN WMD 2019): 24.4 (year 2010)")

# --- Age-15 sensitivity bound, evaluated at the actual C from this run (see
# "AGE-15 SENSITIVITY BOUND" in the module docstring for the derivation) ---
age15_bound = 0.02 / C
print(f"\n=== Age-15 sensitivity bound (see module docstring) ===")
print(f"  Under |p15 - p1619| <= 0.02: |SMAM error| <= 0.02 / C = {age15_bound:.4f} years")

# Update cultural_vars.csv
cv = pd.read_csv("data/manual/cultural_vars.csv")
mask = cv["country"] == "Russia"
assert mask.sum() == 1, (
    f"Expected exactly one Russia row in cultural_vars.csv, found {mask.sum()}"
)
old = cv.loc[mask, "smam_female"].values[0]
cv.loc[mask, "smam_female"] = round(SMAM, 2)

# Atomic write: write to a temp file then rename, so an interrupted write
# cannot leave cultural_vars.csv truncated or corrupted.
tmp_path = "data/manual/cultural_vars.csv.tmp"
cv.to_csv(tmp_path, index=False)
os.replace(tmp_path, "data/manual/cultural_vars.csv")
print(f"\nUpdated cultural_vars.csv: Russia SMAM {old} -> {round(SMAM, 2)}")