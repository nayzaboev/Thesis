"""Part 2 scaffold."""
# Build n=14 country-average cross-section + cultural variables.
"""
13_build_crosssection.py
------------------------
Collapse the annual panel (panel.csv) to 14 country-level averages,
merge with cultural variables (cultural_vars.csv), and produce the
cross-section dataset used in script 18.

Output: data/processed/crosssection.csv (14 rows)
Run from repo root:  python analysis/13_build_crosssection.py
"""

import os
import pandas as pd

p = pd.read_csv("data/processed/panel.csv")
cv = pd.read_csv("data/manual/cultural_vars.csv")

# Country-level means of TFR and economic variables
agg = p.groupby("country").agg(
    mean_tfr=("tfr", "mean"),
    mean_log_gdp_ppp=("log_gdp_ppp", "mean"),
    mean_urban=("urban_pop_pct", "mean"),
    mean_remittances=("remittances_gdp_pct", "mean"),
    mean_under5_mort=("under5_mortality", "mean"),
    bloc=("bloc", "first"),
).reset_index()
agg["ca"] = (agg["bloc"] == "Central Asia").astype(int)
agg["mean_tfr"] = agg["mean_tfr"].round(3)

# Merge cultural variables
cs = agg.merge(cv, on="country", how="left")

# Verify
assert len(cs) == 14, f"Expected 14 rows, got {len(cs)}"
assert cs.isnull().sum().sum() == 0, f"Missing values:\n{cs.isnull().sum()}"

os.makedirs("data/processed", exist_ok=True)
cs.to_csv("data/processed/crosssection.csv", index=False)

print(f"Cross-section built: {len(cs)} countries, {cs.shape[1]} columns")
print(cs[["country", "bloc", "ca", "mean_tfr", "smam_female",
          "muslim_share", "female_mean_schooling"]].to_string(index=False))
print("\nSaved -> data/processed/crosssection.csv")