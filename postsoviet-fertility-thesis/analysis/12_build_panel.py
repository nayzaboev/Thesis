
# Merge fertility + WDI covariates, add 1yr lags, flag missingness (NO interpolation/extrapolation).
"""
12_build_panel.py
-----------------
Build the analytical panel for Part 2.

Inputs:
  data/processed/master_tfr.csv         (TFR + subgroup + bloc, from Part 1)
  data/raw/raw_wb_gdp_per_capita_ppp.csv
  data/raw/raw_wb_urban_pop_pct.csv
  data/raw/raw_wb_remittances_gdp_pct.csv
  data/raw/raw_wb_under5_mortality.csv

Output:
  data/processed/panel.csv

Design rules (do NOT relax):
  - No interpolation. No extrapolation. Gaps stay as NaN; the panel is unbalanced.
  - Missingness is flagged per variable (boolean *_missing column) for transparency.
  - 1-year lags are created for the four annual covariates (used in the FE models).
  - log(GDP-PPP) is precomputed for convenience.
  - A Central Asia dummy is added.

Run from repo root:  python analysis/12_build_panel.py
"""

import os
import numpy as np
import pandas as pd

from _assertions import assert_year_continuity

ANNUAL_VARS = [
    "gdp_per_capita_ppp",
    "urban_pop_pct",
    "remittances_gdp_pct",
    "under5_mortality",
]

# --- Load TFR (already has country, year, tfr, subgroup, bloc from Part 1) ---
tfr = pd.read_csv("data/processed/master_tfr.csv")

# --- Load and merge the four annual covariates ---
panel = tfr.copy()
for var in ANNUAL_VARS:
    cov = pd.read_csv(f"data/raw/raw_wb_{var}.csv")
    panel = panel.merge(cov[["country", "year", var]], on=["country", "year"],
                        how="left", validate="many_to_one")

# --- Sort so lag/groupby operations are well-defined ---
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

# --- Key-integrity guard: the panel must stay one row per (country, year) ---
assert not panel.duplicated(["country", "year"]).any(), (
    "Duplicate (country, year) rows in panel — a covariate merge introduced "
    "a fan-out; check the raw_wb_*.csv files for repeated country-year rows."
)

# --- Missingness flags (BEFORE creating lags, so flags reflect raw availability) ---
for var in ANNUAL_VARS:
    panel[f"{var}_missing"] = panel[var].isna()

# --- Year-continuity guard: shift(1) below silently treats a multi-year gap
# as a one-year change. The panel is built directly from master_tfr.csv,
# which is asserted rectangular (14 countries x 24 years, no gaps) in
# 01_clean_data.py, so this should never fire; it guards against a future
# change to that assumption. ---
assert_year_continuity(panel)

# --- 1-year lags (within country) ---
# linearmodels and statsmodels both prefer the user to supply lags explicitly.
for var in ANNUAL_VARS:
    panel[f"{var}_lag1"] = panel.groupby("country")[var].shift(1)

# --- Log GDP-PPP and its lag (precompute for convenience) ---
panel["log_gdp_ppp"] = np.log(panel["gdp_per_capita_ppp"])
panel["log_gdp_ppp_lag1"] = panel.groupby("country")["log_gdp_ppp"].shift(1)

# --- Central Asia dummy (1 = Central Asia, 0 = rest) ---
panel["ca"] = (panel["bloc"] == "Central Asia").astype(int)

# --- Save ---
os.makedirs("data/processed", exist_ok=True)
panel.to_csv("data/processed/panel.csv", index=False)

# --- Report ---
print(f"Panel saved -> data/processed/panel.csv")
print(f"Shape: {panel.shape}   Countries: {panel['country'].nunique()}   Years: {panel['year'].min()}-{panel['year'].max()}")
print()

print("=== Non-missing counts per variable (raw, after merge) ===")
for var in ANNUAL_VARS + ["tfr"]:
    print(f"  {var:24s}: {int(panel[var].notna().sum())}/{len(panel)}")

print("\n=== Non-missing counts per LAGGED variable (1 year always lost per country at t=2000) ===")
for var in ANNUAL_VARS:
    col = f"{var}_lag1"
    print(f"  {col:30s}: {int(panel[col].notna().sum())}/{len(panel)}")

print("\n=== Effective sample for the FE model (all four LAGGED covariates + TFR present) ===")
lag_cols = [f"{v}_lag1" for v in ANNUAL_VARS]
complete = panel[["tfr"] + lag_cols].notna().all(axis=1)
print(f"  Complete-case rows: {int(complete.sum())}/{len(panel)}")
print(f"  (the 14 cells lost at year=2000 are the lag drop; the rest reflect the remittances gaps)")

print("\n=== CA dummy distribution ===")
print(panel.groupby("bloc")["ca"].first().to_string())