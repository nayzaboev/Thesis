"""
compute_uzb_smam.py
-------------------
Compute female SMAM for Uzbekistan from MICS6 2021-2022 women's file (wm.sav)
using the standard Hajnal (1953) formula.

CRITICAL: uses MSTATUS (recoded marital-status variable), NOT MA1.
  MSTATUS coding (MICS6 standard):
    1 = Currently married/in union
    2 = Formerly married/in union    <- widowed/divorced/separated
    3 = Never married/in union       <- this is what Hajnal SMAM needs
  MA1 alone conflates never-married with formerly-married, biasing SMAM downward.

Hajnal SMAM = (A - D) / C
  A = 15 + 5 * sum of proportions never-married across age groups 15-19..45-49
  B = proportion never-married at 45-49 (proxy for proportion never marrying by 50)
  C = 1 - B
  D = 50 * B

Run from repo root:  python analysis/compute_uzb_smam.py
Reads:               data/raw/wm.sav
Updates:             data/manual/cultural_vars.csv (Uzbekistan row only)
"""

import os

import pandas as pd
import pyreadstat

df, meta = pyreadstat.read_sav("data/raw/wm.sav")
print(f"Loaded wm.sav: {df.shape[0]} rows")

d = df[["WB4","MSTATUS","wmweight"]].dropna().copy()
d = d.rename(columns={"WB4":"age","MSTATUS":"marital","wmweight":"weight"})
d["age"] = d["age"].astype(int)
d = d[(d["age"] >= 15) & (d["age"] <= 49)]

print("\nMSTATUS distribution (weighted):")
print(d.groupby("marital")["weight"].sum().round(1).to_string())

d["single"] = (d["marital"] == 3).astype(int)
d["age_group"] = d["age"].apply(lambda a: f"{5*(a//5)}-{5*(a//5)+4}")

grouped = d.groupby("age_group", group_keys=False).apply(
    lambda g: pd.Series({
        "n_weighted": g["weight"].sum(),
        "prop_single": (g["single"]*g["weight"]).sum() / g["weight"].sum()
    })
).reset_index()

print("\nProportion NEVER-married by 5-year age group (weighted):")
print(grouped.to_string(index=False))

age_groups = ["15-19","20-24","25-29","30-34","35-39","40-44","45-49"]
prop = {r["age_group"]: r["prop_single"] for _, r in grouped.iterrows()}
missing = [g for g in age_groups if g not in prop]
if missing:
    raise SystemExit(f"Missing age groups: {missing}")

A = 15 + 5 * sum(prop[g] for g in age_groups)
B = prop["45-49"]
C = 1 - B
D = 50 * B
SMAM = (A - D) / C

print(f"\n=== Hajnal SMAM ===")
print(f"  A = {A:.3f}")
print(f"  B (prop never-married at 45-49) = {B:.4f}")
print(f"  C = {C:.4f}")
print(f"  D = {D:.3f}")
print(f"  SMAM = {SMAM:.2f}")
print(f"\n>>> Uzbekistan female SMAM (MICS6 2021-2022): {SMAM:.2f}")

# Update cultural_vars.csv
cv = pd.read_csv("data/manual/cultural_vars.csv")
mask = cv["country"] == "Uzbekistan"
assert mask.sum() == 1, (
    f"Expected exactly one Uzbekistan row in cultural_vars.csv, found {mask.sum()}"
)
old = cv.loc[mask, "smam_female"].values[0]
cv.loc[mask, "smam_female"] = round(SMAM, 2)

# Atomic write: write to a temp file then rename, so an interrupted write
# cannot leave cultural_vars.csv truncated or corrupted.
tmp_path = "data/manual/cultural_vars.csv.tmp"
cv.to_csv(tmp_path, index=False)
os.replace(tmp_path, "data/manual/cultural_vars.csv")
print(f"\nUpdated cultural_vars.csv: Uzbekistan SMAM {old} -> {round(SMAM,2)}")