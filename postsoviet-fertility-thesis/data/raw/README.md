# Raw Data Directory

This directory is **git-ignored** because the underlying files are either large binaries or licensed microdata that cannot be redistributed. To reproduce the analysis from scratch, download the files listed below into `data/raw/` before running the scripts.

## Files required by the pipeline

| File | Used by script | Source | How to get it |
|---|---|---|---|
| `unpopulation_dataportal_YYYYMMDDHHMMSS.csv` | `01_clean_data.py` | UN Population Division Data Portal | https://population.un.org/dataportal/ → select "Total Fertility Rate," 14 countries, 2000–2023, "Median" variant → export CSV. Rename or update the path in `01_clean_data.py` to match. |
| `raw_wb_gdp_per_capita_ppp.csv` | `10_fetch_wdi.py` (fetches automatically) | World Bank WDI | Auto-downloaded via `wbgapi` when script runs. Indicator `NY.GDP.PCAP.PP.KD`. |
| `raw_wb_urban_pop_pct.csv` | `10_fetch_wdi.py` | World Bank WDI | Auto-downloaded. Indicator `SP.URB.TOTL.IN.ZS`. |
| `raw_wb_remittances_gdp_pct.csv` | `10_fetch_wdi.py` | World Bank WDI | Auto-downloaded. Indicator `BX.TRF.PWKR.DT.GD.ZS`. |
| `raw_wb_under5_mortality.csv` | `10_fetch_wdi.py` | World Bank WDI | Auto-downloaded. Indicator `SH.DYN.MORT`. |
| `wm.sav` | `compute_uzb_smam.py` | UNICEF MICS6 Uzbekistan 2021–2022, women's file | https://mics.unicef.org/surveys → Uzbekistan → MICS6 2021–2022 → request access (free) → download Women SPSS dataset. Save as `data/raw/wm.sav`. |
| `blrwm.sav` | `compute_blr_smam.py` | UNICEF MICS6 Belarus 2019, women's file | Same portal → Belarus MICS6 2019 → Women SPSS. Save as `data/raw/blrwm.sav`. |
| `geo.wm.sav` | `compute_geo_smam.py` | UNICEF MICS6 Georgia 2018, women's file | Same portal → Georgia MICS6 2018 → Women SPSS. Save as `data/raw/geo.wm.sav`. |
| `kyz.wm.sav` | `compute_kgz_smam.py` | UNICEF MICS6 Kyrgyzstan 2018, women's file | Same portal → Kyrgyzstan MICS6 2018 → Women SPSS. Save as `data/raw/kyz.wm.sav`. |
| `rus_census2021_tab5.xlsx` | `compute_rus_smam.py` | Rosstat, VPN-2020 (2021 Russian Census), Tom 2, Table 5 | https://rosstat.gov.ru/vpn/2020/ → Tom 2 (Возрастно-половой состав и состояние в браке) → download `Tоm2_tab5_VPN-2020.xlsx`. Save as `data/raw/rus_census2021_tab5.xlsx`. |

## Files used only for the cultural cross-section (already merged into `data/manual/cultural_vars.csv`)

These do not need to be re-downloaded to reproduce `18_crosssection.py`, but the sources are recorded here for provenance:

| Variable | Source | URL |
|---|---|---|
| Female SMAM (for the 9 countries not covered by microdata) | UN DESA World Marriage Data 2019 | https://www.un.org/development/desa/pd/data/world-marriage-data |
| Muslim population share (2020) | Pew Research Center, Religious Composition 2010–2020 | https://www.pewresearch.org/religion/2024/06/09/religious-composition-of-the-world-2010-2020/ |
| Female mean years of schooling, age 25+, SSP2 (2020) | Wittgenstein Centre Data Explorer v3.0 | http://dataexplorer.wittgensteincentre.org/wcde-v3/ |

## MICS access note

UNICEF MICS microdata is free but requires a registered account and a signed data-use agreement. Registration takes a few minutes; approval is typically within one working day. The women's SPSS file for each survey is named `wm.sav` in the raw archive — the country-specific filenames in this repo (`blrwm.sav`, `geo.wm.sav`, `kyz.wm.sav`) are renamings for local disambiguation.

## What to do if a raw file is missing

If you clone this repo without the raw files:
- Scripts `10`–`19` will still run **once `10_fetch_wdi.py` has been executed** (it downloads all four World Bank indicators automatically).
- The five SMAM computation scripts will fail with `FileNotFoundError`. They are not required for the main results; `data/manual/cultural_vars.csv` already contains the final computed values and is committed. Downstream scripts (`13`, `18`) read from that file and will produce the correct cross-section results without re-running the SMAM scripts.