"""
compute_blr_smam.py
-------------------
Compute female SMAM for Belarus from MICS6 2019 women's file (blrwm.sav)
using the standard Hajnal (1953) formula.

Uses MSTATUS (recoded marital-status variable):
    1 = Currently married/in union
    2 = Formerly married/in union
    3 = Never married/in union  <- what Hajnal SMAM requires

Source: UNICEF MICS6 Belarus 2019, women's questionnaire microdata
Result: Belarus female SMAM = 23.08

Run from repo root:  python analysis/compute_blr_smam.py
Reads:               data/raw/blrwm.sav
Updates:             data/manual/cultural_vars.csv (Belarus row only)
"""

import pandas as pd
import pyreadstat

df, meta = pyreadstat.read_sav("data/raw/blrwm.sav")
print(f"Loaded blrwm.sav: {df.shape[0]} rows")

d = df[["WB4", "MSTATUS", "wmweight"]].dropna().copy()
d = d.rename(columns={"WB4": "age", "MSTATUS": "marital", "wmweight": "weight"})
d["age"] = d["age"].astype(int)
d = d[(d["age"] >= 15) & (d["age"] <= 49)]

print("\nMSTATUS distribution (weighted):")
print(d.groupby("marital")["weight"].sum().round(1).to_string())

d["single"] = (d["marital"] == 3).astype(int)
d["age_group"] = d["age"].apply(lambda a: f"{5*(a//5)}-{5*(a//5)+4}")

grouped = d.groupby("age_group", group_keys=False).apply(
    lambda g: pd.Series({
        "n_weighted": g["weight"].sum(),
        "prop_single": (g["single"] * g["weight"]).sum() / g["weight"].sum()
    })
).reset_index()

print("\nProportion NEVER-married by 5-year age group (weighted):")
print(grouped.to_string(index=False))

age_groups = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
prop = {r["age_group"]: r["prop_single"] for _, r in grouped.iterrows()}

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
print(f"\n>>> Belarus female SMAM (MICS6 2019): {SMAM:.2f}")
print(f"    Previous (UN WMD 2019): 22.5 (year 2012)")

# Update cultural_vars.csv
cv = pd.read_csv("data/manual/cultural_vars.csv")
old = cv.loc[cv["country"] == "Belarus", "smam_female"].values[0]
cv.loc[cv["country"] == "Belarus", "smam_female"] = round(SMAM, 2)
cv.to_csv("data/manual/cultural_vars.csv", index=False)
print(f"\nUpdated cultural_vars.csv: Belarus SMAM {old} -> {round(SMAM, 2)}")