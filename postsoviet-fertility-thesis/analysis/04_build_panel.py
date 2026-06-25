import pandas as pd

fert = pd.read_csv("data/processed/master_tfr.csv")
gdp = pd.read_csv("data/raw/raw_wb_gdp.csv")
urban = pd.read_csv("data/raw/wb_urban.csv")
flfp = pd.read_csv("data/raw/raw_wb_flfp.csv")
remit = pd.read_csv("data/raw/raw_wb_remittances.csv")
tert = pd.read_csv("data/raw/raw_wb_tertiary_edu.csv")
code_to_name = {
    "ARM":"Armenia","AZE":"Azerbaijan","BLR":"Belarus","EST":"Estonia",
    "GEO":"Georgia","KAZ":"Kazakhstan","KGZ":"Kyrgyzstan","LVA":"Latvia",
    "LTU":"Lithuania","MDA":"Moldova","RUS":"Russia","TJK":"Tajikistan",
    "UKR":"Ukraine","UZB":"Uzbekistan"
}
urban.columns = [c.lower() for c in urban.columns]
urban["country"] = urban["country"].replace(code_to_name)
panel = fert.copy()
for cov in [gdp, urban, flfp, remit, tert]:
    cov.columns = [c.lower() for c in cov.columns]
    panel = panel.merge(cov, on=["country", "year"], how="left")
    cov_cols = ["gdp_per_capita_usd", "urban_pop_pct", "flfp_pct",
            "remittances_gdp_pct", "female_tert_educ_pct"]

panel = panel.sort_values(["country", "year"])

for col in cov_cols:
    # flag column: True where the original value was missing
    panel[col + "_interp"] = panel[col].isna()
    # interpolate within each country, by year
    panel[col] = panel.groupby("country")[col].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    print("Rows:", len(panel), "(expect 336)")
print("Countries:", panel["country"].nunique())
print("\nInterpolated values per variable:")
for col in cov_cols:
    print(f"  {col}: {panel[col + '_interp'].sum()} filled")

panel.to_csv("data/processed/panel.csv", index=False)
print("\nSaved -> data/processed/panel.csv")