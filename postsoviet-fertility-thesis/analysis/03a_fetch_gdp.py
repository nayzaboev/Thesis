import os
import wbgapi as wb
import pandas as pd

INDICATOR = "NY.GDP.PCAP.CD"
COUNTRIES = ['ARM', 'AZE', 'BLR', 'EST', 'GEO', 'KAZ', 'KGZ', 'LVA', 'LTU',
             'MDA', 'RUS', 'TJK', 'UKR', 'UZB']
YEARS = range(2000, 2024)

name_map = {'ARM': 'Armenia', 'AZE': 'Azerbaijan', 'BLR': 'Belarus', 'EST': 'Estonia',
            'GEO': 'Georgia', 'KAZ': 'Kazakhstan', 'KGZ': 'Kyrgyzstan', 'LVA': 'Latvia',
            'LTU': 'Lithuania', 'MDA': 'Moldova', 'RUS': 'Russia', 'TJK': 'Tajikistan',
            'UKR': 'Ukraine', 'UZB': 'Uzbekistan'}

raw = wb.data.DataFrame(INDICATOR, economy=COUNTRIES, time=YEARS, labels=False)

df = raw.reset_index().melt(id_vars="economy", var_name="year", value_name="gdp_per_capita_usd")
df["year"] = df["year"].str.replace("YR", "", regex=False).astype(int)
df["country"] = df["economy"].map(name_map)
df = df[["country", "year", "gdp_per_capita_usd"]]
df = df.sort_values(["country", "year"]).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/raw_wb_gdp.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/raw_wb_gdp.csv")
