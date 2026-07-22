"""
Fetch World Development Indicators (WDI) covariates for Part 2 regressions.

Salvaged from the old 03a_fetch_gdp.py wbgapi pattern (wb.data.DataFrame ->
melt -> rename -> save), generalised to loop over all four Part 2 indicators.

This script OVERWRITES the frozen data/raw/raw_wb_*.csv snapshots that the
committed analysis reproduces from. It is opt-in maintenance, not part of the
default pipeline (see run_all.py / README): run it only when you intend to
refresh the snapshots with live World Bank data, e.g. to extend coverage to a
new year. Pass --refresh to actually write; otherwise it is a no-op.

Run from repo root:  python analysis/10_fetch_wdi.py --refresh
"""

import os
import sys
import wbgapi as wb
import pandas as pd

if "--refresh" not in sys.argv:
    print("Frozen World Bank snapshots are in use (data/raw/raw_wb_*.csv).")
    print("The committed analysis reproduces from these frozen files, not live data.")
    print("Pass --refresh to overwrite them with a fresh download. Exiting without writing.")
    sys.exit(0)

INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp",
    "SP.URB.TOTL.IN.ZS": "urban_pop_pct",
    "BX.TRF.PWKR.DT.GD.ZS": "remittances_gdp_pct",
    "SH.DYN.MORT": "under5_mortality",
}

COUNTRIES = ['ARM', 'AZE', 'BLR', 'EST', 'GEO', 'KAZ', 'KGZ', 'LVA', 'LTU',
             'MDA', 'RUS', 'TJK', 'UKR', 'UZB']
YEARS = range(2000, 2024)

name_map = {'ARM': 'Armenia', 'AZE': 'Azerbaijan', 'BLR': 'Belarus', 'EST': 'Estonia',
            'GEO': 'Georgia', 'KAZ': 'Kazakhstan', 'KGZ': 'Kyrgyzstan', 'LVA': 'Latvia',
            'LTU': 'Lithuania', 'MDA': 'Moldova', 'RUS': 'Russia', 'TJK': 'Tajikistan',
            'UKR': 'Ukraine', 'UZB': 'Uzbekistan'}

os.makedirs("data/raw", exist_ok=True)

for indicator, value_name in INDICATORS.items():
    raw = wb.data.DataFrame(indicator, economy=COUNTRIES, time=YEARS, labels=False)

    df = raw.reset_index().melt(id_vars="economy", var_name="year", value_name=value_name)
    df["year"] = df["year"].str.replace("YR", "", regex=False).astype(int)
    df["country"] = df["economy"].map(name_map)
    df = df[["country", "year", value_name]]
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    out_path = f"data/raw/raw_wb_{value_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
