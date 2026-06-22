import os
import wbgapi as wb
import pandas as pd

INDICATOR = "SP.URB.TOTL.IN.ZS"
COUNTRIES = ['KAZ', 'KGZ', 'TJK', 'UZB', 'RUS', 'UKR', 'BLR', 'MDA',
             'EST', 'LVA', 'LTU', 'ARM', 'GEO', 'AZE']
YEARS = range(2000, 2024)

raw = wb.data.DataFrame(INDICATOR, economy=COUNTRIES, time=YEARS, labels=False)

df = raw.reset_index().melt(id_vars="economy", var_name="year", value_name="urban_pop_pct")
df["year"] = df["year"].str.replace("YR", "", regex=False).astype(int)
df = df.rename(columns={"economy": "Country"})
df = df[["Country", "year", "urban_pop_pct"]]
df = df.sort_values(["Country", "year"]).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/wb_urban.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/wb_urban.csv")
