"""
compute_rus_smam.py
-------------------
Compute female SMAM for Russia from the 2021 Census (VPN-2020)
Table 5: Population by age, sex, and marital status.

Uses the standard Hajnal (1953) formula on census counts of women
who have NEVER been married (никогда не состоявшие в браке, column 6)
as a proportion of those who stated marital status (column 2).

Note: the Russian census covers ages 16+, so the 15-19 age group is
approximated by combining the 16-17 and 18-19 rows. Since essentially
all 15-year-olds are single, the effect on SMAM is negligible (<0.1 year).

Source: Rosstat, Всероссийская перепись населения 2020 (conducted Oct 2021),
        Tom 2, Table 5 (Tоm2_tab5_VPN-2020.xlsx)
Result: Russia female SMAM = 24.86

Run from repo root:  python analysis/compute_rus_smam.py
Reads:               data/raw/rus_census2021_tab5.xlsx
Updates:             data/manual/cultural_vars.csv (Russia row only)
"""

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

# Update cultural_vars.csv
cv = pd.read_csv("data/manual/cultural_vars.csv")
old = cv.loc[cv["country"] == "Russia", "smam_female"].values[0]
cv.loc[cv["country"] == "Russia", "smam_female"] = round(SMAM, 2)
cv.to_csv("data/manual/cultural_vars.csv", index=False)
print(f"\nUpdated cultural_vars.csv: Russia SMAM {old} -> {round(SMAM, 2)}")