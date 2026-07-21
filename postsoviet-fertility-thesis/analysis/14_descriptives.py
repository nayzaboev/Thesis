# Summary statistics + coverage appendix table.
"""
14_descriptives.py
------------------
Descriptive statistics and coverage documentation for Part 2.

Produces:
  1. A summary-statistics table (mean, sd, min, median, max, N) for TFR and the
     four annual covariates, OVERALL and split Central Asia vs Rest.
  2. A between-bloc means comparison for the dependent and each covariate,
     reported BOTH unweighted (country average) AND population-weighted.
     Unweighted answers "how do average countries differ?"; population-weighted
     answers "how do average PEOPLE differ?" — the two can diverge when small
     countries (e.g. the Baltics) or very large ones (Russia, Uzbekistan)
     pull the mean in different directions.
  3. The coverage / missingness appendix table (per variable: N present, N missing,
     and which country-years are missing) — the audit trail for the unbalanced panel.

Outputs:
  data/processed/summary_stats.csv
  data/processed/summary_by_bloc.csv
  data/processed/summary_by_bloc_pop_weighted.csv
  data/processed/missingness_appendix.csv

Run from repo root:  python analysis/14_descriptives.py
"""

import os
import numpy as np
import pandas as pd

p = pd.read_csv("data/processed/panel.csv")

DV = "tfr"
COVS = ["gdp_per_capita_ppp", "urban_pop_pct", "remittances_gdp_pct", "under5_mortality"]
ALL_VARS = [DV] + COVS

LABELS = {
    "tfr": "Total fertility rate",
    "gdp_per_capita_ppp": "GDP per capita, PPP (const 2021 intl $)",
    "urban_pop_pct": "Urban population (%)",
    "remittances_gdp_pct": "Remittances (% of GDP)",
    "under5_mortality": "Under-5 mortality (per 1,000)",
}

os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Overall summary statistics
# ---------------------------------------------------------------------------
def describe(df, cols):
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "variable": LABELS[c],
            "N": int(s.shape[0]),
            "mean": round(s.mean(), 2),
            "sd": round(s.std(), 2),
            "min": round(s.min(), 2),
            "median": round(s.median(), 2),
            "max": round(s.max(), 2),
        })
    return pd.DataFrame(rows)

summary = describe(p, ALL_VARS)
summary.to_csv("data/processed/summary_stats.csv", index=False)
print("=== Summary statistics (all 14 countries, 2000-2023) ===")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 2a. By bloc — UNWEIGHTED (country-level average within each bloc)
# ---------------------------------------------------------------------------
print("\n=== Means by bloc — UNWEIGHTED (country average) ===")
rows = []
for c in ALL_VARS:
    ca_mean = p.loc[p["bloc"] == "Central Asia", c].mean()
    rest_mean = p.loc[p["bloc"] == "Rest of post-Soviet", c].mean()
    rows.append({
        "variable": LABELS[c],
        "Central Asia": round(ca_mean, 2),
        "Rest": round(rest_mean, 2),
        "difference": round(ca_mean - rest_mean, 2),
    })
by_bloc = pd.DataFrame(rows)
by_bloc.to_csv("data/processed/summary_by_bloc.csv", index=False)
print(by_bloc.to_string(index=False))

# ---------------------------------------------------------------------------
# 2b. By bloc — POPULATION-WEIGHTED (each country weighted by its population)
# ---------------------------------------------------------------------------
# We fetch total population from the World Bank (SP.POP.TOTL) on the fly rather
# than baking it into panel.csv — it is not used in any regression, only in this
# descriptive comparison. If the fetch fails (no internet), we skip gracefully.
print("\n=== Means by bloc — POPULATION-WEIGHTED ===")
try:
    import wbgapi as wb
    ISO = {"Armenia":"ARM","Azerbaijan":"AZE","Belarus":"BLR","Estonia":"EST",
           "Georgia":"GEO","Kazakhstan":"KAZ","Kyrgyzstan":"KGZ","Latvia":"LVA",
           "Lithuania":"LTU","Moldova":"MDA","Russia":"RUS","Tajikistan":"TJK",
           "Ukraine":"UKR","Uzbekistan":"UZB"}
    pop_df = wb.data.DataFrame("SP.POP.TOTL", economy=list(ISO.values()),
                                time=range(2000, 2024), labels=False)
    pop_df = pop_df.reset_index().melt(id_vars="economy",
                                        var_name="year", value_name="population")
    pop_df["year"] = pop_df["year"].str.replace("YR", "", regex=False).astype(int)
    iso_to_name = {v: k for k, v in ISO.items()}
    pop_df["country"] = pop_df["economy"].map(iso_to_name)
    pop_df = pop_df[["country", "year", "population"]]

    pw = p.merge(pop_df, on=["country", "year"], how="left")

    def wmean(sub, col):
        s = sub[[col, "population"]].dropna()
        if s.empty: return float("nan")
        return (s[col] * s["population"]).sum() / s["population"].sum()

    rows = []
    for c in ALL_VARS:
        ca_w = wmean(pw[pw["bloc"] == "Central Asia"], c)
        rest_w = wmean(pw[pw["bloc"] == "Rest of post-Soviet"], c)
        rows.append({
            "variable": LABELS[c],
            "Central Asia (pop-weighted)": round(ca_w, 2),
            "Rest (pop-weighted)": round(rest_w, 2),
            "difference": round(ca_w - rest_w, 2),
        })
    by_bloc_pw = pd.DataFrame(rows)
    by_bloc_pw.to_csv("data/processed/summary_by_bloc_pop_weighted.csv", index=False)
    print(by_bloc_pw.to_string(index=False))
    print("\n  Note: unweighted means treat each country as one observation;")
    print("  population-weighted means give large countries proportional influence.")
    print("  Divergences between the two flag countries whose demographic size is")
    print("  atypical for their bloc (Russia in the Rest; Uzbekistan in CA).")
except Exception as e:
    print(f"  Population fetch failed ({type(e).__name__}: {e});")
    print("  skipping population-weighted table. Run with internet access to include it.")
    # Prevent a SILENTLY STALE output: if the live fetch fails, remove any
    # previously-written population-weighted CSV so the repo never ships a
    # pop-weighted table that does not correspond to the current run.
    _pw_path = "data/processed/summary_by_bloc_pop_weighted.csv"
    if os.path.exists(_pw_path):
        os.remove(_pw_path)
        print(f"  Removed stale {_pw_path} (was from an earlier run).")

# ---------------------------------------------------------------------------
# 3. Missingness appendix (the audit trail)
# ---------------------------------------------------------------------------
print("\n=== Missingness appendix (unbalanced-panel audit trail) ===")
rows = []
for c in COVS:
    miss_mask = p[c].isna()
    miss_pairs = p.loc[miss_mask, ["country", "year"]].sort_values(["country", "year"])
    pairs_str = "; ".join(f"{r.country} {int(r.year)}" for r in miss_pairs.itertuples()) or "—"
    rows.append({
        "variable": LABELS[c],
        "N_present": int((~miss_mask).sum()),
        "N_missing": int(miss_mask.sum()),
        "missing_country_years": pairs_str,
    })
missingness = pd.DataFrame(rows)
missingness.to_csv("data/processed/missingness_appendix.csv", index=False)
# print without the long string column for console readability
print(missingness[["variable", "N_present", "N_missing"]].to_string(index=False))
print("\nMissing country-years detail:")
for r in missingness.itertuples():
    print(f"  {r.variable}: {r.missing_country_years}")

print("\nSaved -> summary_stats.csv, summary_by_bloc.csv, missingness_appendix.csv")