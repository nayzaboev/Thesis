"""Part 2 scaffold."""
# Country x year x variable coverage grid + heatmap.
"""
11_coverage_map.py
------------------
Build a per-country x per-variable data-coverage map for the four annual WDI
series, BEFORE any merging or modelling.

Why this runs first: wbgapi returns a full rectangular grid (one row per
country-year) even when the underlying value is missing, so a row count of 336
does NOT mean 336 real observations. This script counts genuine (non-missing)
values, prints a coverage table, lists every missing country-year, and saves a
heatmap. This is the evidence base for deciding how to handle gaps — and the
answer to any "is your data real?" question at the defence.

Run from repo root:  python analysis/11_coverage_map.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

YEARS = list(range(2000, 2024))
N_YEARS = len(YEARS)

# raw file -> tidy variable name (must match 10_fetch_wdi.py outputs)
FILES = {
    "data/raw/raw_wb_gdp_per_capita_ppp.csv": "gdp_per_capita_ppp",
    "data/raw/raw_wb_urban_pop_pct.csv":      "urban_pop_pct",
    "data/raw/raw_wb_remittances_gdp_pct.csv":"remittances_gdp_pct",
    "data/raw/raw_wb_under5_mortality.csv":   "under5_mortality",
}

# --- Load and merge the four series on (country, year) ---
merged = None
for path, var in FILES.items():
    df = pd.read_csv(path)
    # the value column is whatever isn't country/year
    val_col = [c for c in df.columns if c not in ("country", "year")][0]
    df = df.rename(columns={val_col: var})[["country", "year", var]]
    merged = df if merged is None else merged.merge(df, on=["country", "year"], how="outer")

countries = sorted(merged["country"].unique())
varnames = list(FILES.values())

print(f"Countries: {len(countries)}   Years: {YEARS[0]}-{YEARS[-1]}   Variables: {len(varnames)}")
print(f"Full rectangular grid would be {len(countries)} x {N_YEARS} = {len(countries)*N_YEARS} country-years per variable.\n")

# --- Coverage table: non-missing count per country per variable ---
print("=== Non-missing values per country (out of %d years) ===" % N_YEARS)
header = "country".ljust(13) + "".join(v[:11].rjust(13) for v in varnames)
print(header)
print("-" * len(header))

coverage = pd.DataFrame(index=countries, columns=varnames, dtype=int)
for c in countries:
    sub = merged[merged["country"] == c]
    row = c.ljust(13)
    for v in varnames:
        n = int(sub[v].notna().sum())
        coverage.loc[c, v] = n
        flag = "" if n == N_YEARS else " *"
        row += (f"{n}/{N_YEARS}{flag}").rjust(13)
    print(row)

# --- Per-variable totals ---
print("\n=== Per-variable coverage (all countries) ===")
total_cells = len(countries) * N_YEARS
for v in varnames:
    got = int(merged[v].notna().sum())
    pct = 100 * got / total_cells
    print(f"  {v:24s}: {got}/{total_cells}  ({pct:.1f}%)")

# --- List every missing country-year explicitly ---
print("\n=== Missing country-years (explicit) ===")
any_missing = False
for v in varnames:
    miss = merged[merged[v].isna()][["country", "year"]].sort_values(["country", "year"])
    if len(miss):
        any_missing = True
        pairs = ", ".join(f"{r.country} {int(r.year)}" for r in miss.itertuples())
        print(f"  {v}: {len(miss)} missing -> {pairs}")
if not any_missing:
    print("  None — every country-year has a real value for all four variables.")

# --- Save coverage table ---
os.makedirs("data/processed", exist_ok=True)
coverage.to_csv("data/processed/coverage_table.csv")
print("\nSaved coverage table -> data/processed/coverage_table.csv")

# --- Heatmap: full picture of present (1) vs missing (0), country x year, per variable ---
fig, axes = plt.subplots(1, len(varnames), figsize=(4.2 * len(varnames), 5.2), sharey=True)
cmap = mcolors.ListedColormap(["#c1121f", "#2a9d8f"])  # red=missing, teal=present
for ax, v in zip(axes, varnames):
    grid = (merged.pivot(index="country", columns="year", values=v)
                  .reindex(index=countries, columns=YEARS))
    present = grid.notna().astype(int).values
    ax.imshow(present, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(v, fontsize=9, weight="bold")
    ax.set_xticks(range(0, N_YEARS, 4))
    ax.set_xticklabels(YEARS[::4], fontsize=7, rotation=45)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=7)
    ax.set_xlabel("year", fontsize=8)
fig.suptitle("Data coverage map (teal = present, red = missing)", fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/coverage_map.png", bbox_inches="tight", dpi=120)
print("Saved heatmap -> figures/coverage_map.png")