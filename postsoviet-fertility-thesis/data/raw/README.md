# Raw Data Directory

This directory is **git-ignored** (`data/raw/*`) because the files are either
large binaries or licensed microdata that cannot be redistributed. The exception
is this README, which is tracked.

To reproduce the analysis from scratch, download the files listed below into
`data/raw/` before running the scripts.

## Files auto-downloaded by the pipeline

These are fetched by `10_fetch_wdi.py` via the World Bank API (`wbgapi`). If
you have internet access, simply run the script; the CSVs appear automatically.

| File | WDI indicator | Variable |
|---|---|---|
| `raw_wb_gdp_per_capita_ppp.csv` | `NY.GDP.PCAP.PP.KD` | GDP per capita, PPP (constant 2021 int'l $) |
| `raw_wb_urban_pop_pct.csv` | `SP.URB.TOTL.IN.ZS` | Urban population (%) |
| `raw_wb_remittances_gdp_pct.csv` | `BX.TRF.PWKR.DT.GD.ZS` | Personal remittances received (% of GDP) |
| `raw_wb_under5_mortality.csv` | `SH.DYN.MORT` | Under-5 mortality rate (per 1,000) |

## Files requiring manual download

### UN World Population Prospects (TFR)

| File | Script | Source |
|---|---|---|
| `unpopulation_dataportal_20260616151558.csv` | `01_clean_data.py` | https://population.un.org/dataportal/ → Indicator: Total Fertility Rate → Locations: the 14 study countries → Variant: Median → Years: 2000–2023 → Export CSV. The filename includes a timestamp; update the path in `01_clean_data.py` if yours differs. |

### MICS6 women's microdata (for SMAM computation)

All four files are obtained from https://mics.unicef.org/surveys. Registration
is free; approval typically takes one working day. Download the Women's SPSS
dataset (`.sav`) from each survey page and rename as shown.

| File | Script | Survey |
|---|---|---|
| `wm.sav` | `compute_uzb_smam.py` | Uzbekistan MICS6, 2021–2022 |
| `blrwm.sav` | `compute_blr_smam.py` | Belarus MICS6, 2019 |
| `geo.wm.sav` | `compute_geo_smam.py` | Georgia MICS6, 2018 |
| `kyz.wm.sav` | `compute_kgz_smam.py` | Kyrgyzstan MICS6, 2018 |

### Russian Census 2021

| File | Script | Source |
|---|---|---|
| `rus_census2021_tab5.xlsx` | `compute_rus_smam.py` | Rosstat, Всероссийская перепись населения 2020 (conducted October 2021), Tom 2, Table 5: "Население по возрасту, полу и состоянию в браке." Download from https://rosstat.gov.ru/vpn/2020/ → Tom 2 → Table 5 (`Tоm2_tab5_VPN-2020.xlsx`). Rename to `rus_census2021_tab5.xlsx`. |

## Files used for the cultural cross-section (already in `data/manual/`)

These do not need to be re-downloaded to reproduce scripts 13–20. The final
values and full source/year metadata are committed in `data/manual/cultural_vars.csv`.
Sources are recorded here for provenance.

| Variable | Source | URL |
|---|---|---|
| Female SMAM (9 countries not covered by microdata) | UN DESA, World Marriage Data 2019 | https://www.un.org/development/desa/pd/data/world-marriage-data |
| Muslim population share (2020) | Pew Research Center, Religious Composition 2010–2020 | https://www.pewresearch.org/religion/ |
| Female mean years of schooling, age 25+, SSP2 (2020) | Wittgenstein Centre, Human Capital Data Explorer v3.0 | http://dataexplorer.wittgensteincentre.org/wcde-v3/ |

## What happens if raw files are missing

- Scripts `10`–`20` will still run once `10_fetch_wdi.py` has been executed
  (it auto-downloads the four WDI indicators).
- The five `compute_*_smam.py` scripts will fail with `FileNotFoundError`.
  They are **not required** for the main results: `data/manual/cultural_vars.csv`
  already contains the final computed SMAM values and is committed. Downstream
  scripts (`13_build_crosssection`, `18_crosssection`) read from that file and
  produce the correct cross-section outputs without re-running the SMAM scripts.
- `01_clean_data.py` requires the WPP CSV. If you re-download from the UN Data
  Portal, the filename will include a different timestamp; update line 9 of the
  script accordingly.