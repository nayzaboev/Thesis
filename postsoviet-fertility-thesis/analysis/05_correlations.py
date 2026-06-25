import pandas as pd
p = pd.read_csv("data/processed/panel.csv")

cov_cols = ["gdp_per_capita_usd", "urban_pop_pct", "flfp_pct",
            "remittances_gdp_pct", "female_tert_educ_pct"]
print("=== Pooled correlations with TFR (all 336 country-years) ===")
for col in cov_cols:
    r = p["tfr"].corr(p[col])
    print(f"  {col}: r = {r:.3f}")
    country_means = p.groupby("country")[["tfr"] + cov_cols].mean()
print("\n=== Between-country correlations (country averages, n=14) ===")
for col in cov_cols:
    r = country_means["tfr"].corr(country_means[col])
    print(f"  {col}: r = {r:.3f}")
    print("\n=== Correlations AMONG covariates (watch for high values) ===")
print(country_means[cov_cols].corr().round(2))
country_means.corr().round(3).to_csv("data/processed/correlations.csv")
print("\nSaved -> data/processed/correlations.csv")