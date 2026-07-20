"""
01_clean_data.py
----------------
Extract the Total Fertility Rate (Median variant) for the 14 study countries
from the UN Population Division Data Portal export, 2000–2023.

Input:  data/raw/unpopulation_dataportal_*.csv (WPP 2024)
Output: data/processed/master_tfr.csv (336 rows: 14 countries x 24 years)
"""

import pandas as pd

# --- Load raw WPP export ---
df = pd.read_csv("data/raw/unpopulation_dataportal_20260616151558.csv")

# --- Filter to the indicator/variant/sex we need ---
df = df[df["IndicatorName"] == "Total fertility rate"]
df = df[df["Variant"] == "Median"]
df = df[df["Sex"] == "Both sexes"]

# --- Rename & normalise country names ---
df = df[["Location", "Time", "Value"]].copy()
df.columns = ["country", "year", "tfr"]
df["country"] = df["country"].replace({
    "Russian Federation": "Russia",
    "Republic of Moldova": "Moldova",
})

# --- Define the 14-country sample and subgroups ---
central_asia = ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan"]
eastern_eu   = ["Russia", "Ukraine", "Belarus", "Moldova"]
baltic       = ["Estonia", "Latvia", "Lithuania"]
caucasus     = ["Armenia", "Georgia", "Azerbaijan"]
COUNTRIES    = central_asia + eastern_eu + baltic + caucasus  # 14 countries

# --- Filter to sample countries and study years ---
df = df[df["country"].isin(COUNTRIES)]
df = df[df["year"].between(2000, 2023)]

# --- Sample-integrity guarantees (fail fast if raw file changes shape) ---
assert df["country"].nunique() == 14, (
    f"Expected 14 countries, got {df['country'].nunique()}. "
    f"Missing: {set(COUNTRIES) - set(df['country'].unique())}"
)
assert df["year"].min() == 2000, f"Year range starts at {df['year'].min()}, expected 2000"
assert df["year"].max() == 2023, f"Year range ends at {df['year'].max()}, expected 2023"
assert len(df) == 336, f"Expected 336 rows (14 countries x 24 years), got {len(df)}"
assert df.groupby("country")["year"].nunique().eq(24).all(), \
    "Not all countries have complete 2000-2023 coverage"

# --- Assign subgroup and bloc ---
def subgroup(c):
    if c in central_asia: return "Central Asia"
    if c in eastern_eu:   return "Eastern European"
    if c in baltic:       return "Baltic"
    if c in caucasus:     return "Caucasus"
    return "Unassigned"

df["subgroup"] = df["country"].apply(subgroup)
df["bloc"] = df["subgroup"].apply(
    lambda s: "Central Asia" if s == "Central Asia" else "Rest of post-Soviet"
)

# --- Save ---
df = df.sort_values(["subgroup", "country", "year"])
df.to_csv("data/processed/master_tfr.csv", index=False)
print(f"Saved. Rows: {len(df)}, Countries: {df['country'].nunique()}, "
      f"Years: {df['year'].min()}–{df['year'].max()}")