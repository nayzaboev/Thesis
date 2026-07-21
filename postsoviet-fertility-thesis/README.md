# Post-Soviet Fertility Thesis

**Title:** Fertility Trends in the Post-Soviet Space: A Comparative Analysis of Central Asia and Other Former Soviet Countries (2000–2023)

**Degree:** Master's Thesis — University of Bonn

**Author:** Shokhrukh Nayzaboev

## Research Question

How have fertility trends evolved in Central Asia and other post-Soviet countries between 2000 and 2023, and what factors are associated with the persistence of comparatively higher fertility rates observed in Central Asia?

## Countries (n = 14)

The Central Asia group covers the **four Central Asian successor states with
sufficiently comparable data** (Kazakhstan, Kyrgyzstan, Tajikistan, Uzbekistan).
Turkmenistan, the fifth Central Asian successor state, is **excluded** — not for
convenience but because its official statistics are not comparable with the rest
of the sample:

| Issue | Evidence | Consequence for this study |
|---|---|---|
| Covariate coverage | World Bank WDI series used as controls (GDP-PPP, remittances % GDP, urbanisation, under-5 mortality) are missing or sparsely reported for Turkmenistan across 2000–2023 | Turkmenistan cannot enter the controlled panel (Part 2) on the same basis as the other 14 |
| Statistical transparency | Turkmenistan's demographic and economic reporting is widely regarded as opaque, with limited independent verification | Even the TFR series (Part 1) would not be comparably measured |
| Treatment in this thesis | Excluded from **all** analyses (Parts 1 and 2), not selectively | Scope is "four comparable Central Asian states", stated explicitly, rather than implying full coverage of the region |

Adding Turkmenistan merely to achieve nominal five-country coverage would import
non-comparable measurement into both the descriptive and the modelling parts;
the exclusion is therefore a deliberate scope decision, documented here.

- **Central Asia:** Kazakhstan, Kyrgyzstan, Tajikistan, Uzbekistan
- **Eastern European:** Belarus, Moldova, Russia, Ukraine
- **Baltic:** Estonia, Latvia, Lithuania
- **Caucasus:** Armenia, Azerbaijan, Georgia

## Headline Result

The Central Asia fertility premium (average of 2.89 vs 1.57 children per woman elsewhere) is a raw gap of **+1.31**. Under lagged economic controls (log GDP-PPP, urbanisation, remittances, under-5 mortality) with year fixed effects and country-clustered standard errors, the premium narrows to **+0.97** and remains highly significant. This attenuation is descriptive and is not interpreted as a causal decomposition. Muslim population share and female SMAM are strongly correlated bivariately with country-average fertility, but they cannot be empirically separated from broader Central Asian regional identity in the 14-country cross-section (r(CA, Muslim) ≈ 0.83). The cultural interpretation rests on the demographic literature, not on the cross-sectional regression.

## Project Structure

```
postsoviet-fertility-thesis/
├── analysis/                 numbered Python scripts, run in order
├── data/
│   ├── raw/                  raw downloads (git-ignored; see data/raw/README.md)
│   ├── manual/               hand-entered cultural variables (tracked)
│   └── processed/            all script outputs (tracked)
├── figures/                  generated plots (tracked)
└── requirements.txt          pinned Python dependencies
```

## Data Sources

| Variable | Source | Coverage |
|---|---|---|
| Total Fertility Rate | UN World Population Prospects 2024 (Median variant) | 2000–2023, all 14 |
| GDP per capita, PPP (const. 2021 int'l $) | World Bank WDI (`NY.GDP.PCAP.PP.KD`) | 2000–2023, all 14 |
| Urban population % | World Bank WDI (`SP.URB.TOTL.IN.ZS`) | 2000–2023, all 14 |
| Remittances % GDP | World Bank WDI (`BX.TRF.PWKR.DT.GD.ZS`) | 2000–2023, 8 missing cells |
| Under-5 mortality (per 1,000) | World Bank WDI (`SH.DYN.MORT`) | 2000–2023, all 14 |
| Female SMAM | UN World Marriage Data 2019 + MICS6 microdata + Rosstat 2021 Census | see below |
| Muslim population share | Pew Research Center, Religious Composition 2010–2020 | 2020, all 14 |
| Female mean years of schooling (25+) | Wittgenstein Centre Data Explorer v3.0, SSP2 | 2020, all 14 |

**SMAM sources per country:**
- UN World Marriage Data 2019 (latest available observation, years 2010–2018): Armenia, Azerbaijan, Estonia, Kazakhstan, Latvia, Lithuania, Moldova, Tajikistan, Ukraine.
- Author-computed from MICS6 women's microdata using the Hajnal (1953) formula: Belarus (2019), Georgia (2018), Kyrgyzstan (2018), Uzbekistan (2021–2022).
- Author-computed from the 2021 Russian Census (VPN-2020, Table 5): Russia.

## Method — Two-Layer Design

**Part 1 — Descriptive comparative** (`01`, `02`, `03`): TFR trend plots, bloc-mean comparisons, σ-convergence within groups.

**Part 2 — Explanatory analysis** in two layers:

- **Layer A — between-country gap** (`15`): pooled OLS with year fixed effects and country-clustered SEs. Model 1 estimates the raw Central Asia premium; Model 2 adds lagged economic controls and reports the surviving conditional premium; Model 3 tests interactions (CA × remittances, CA × urbanisation, CA × GDP).
- **Layer B — within-country robustness** (`16`): two-way fixed effects (country + year), first-difference model, average ADF t-bar unit-root check (descriptive form, not the formal IPS test), and a residual AR(1) serial-correlation check. This layer defends against the spurious-regression concern that motivated the analytical rebuild.
- **Diagnostics** (`17`): VIF, Mundlak test (FE vs RE — replaces the classical Hausman, which is undefined in this sample), and the few-cluster caveat.
- **Cross-section** (`18`): country-average TFR regressed on cultural variables (Muslim share, SMAM, female schooling) to examine whether selected cultural and nuptiality indicators align descriptively with the between-country pattern and to test whether they can be separated from Central Asian regional status (n = 14).

## Run Order

Scripts are numbered by execution order. Run all from the repo root.

```
# Part 1 — descriptive
python analysis/01_clean_data.py
python analysis/02_trend_plots.py
python analysis/03_convergence.py

# Part 2 — data pipeline
python analysis/10_fetch_wdi.py
python analysis/11_coverage_map.py
python analysis/12_build_panel.py
python analysis/13_build_crosssection.py
python analysis/14_descriptives.py

# Part 2 — models & diagnostics
python analysis/15_layerA_gap.py
python analysis/16_layerB_within.py
python analysis/17_diagnostics.py
python analysis/18_crosssection.py
python analysis/19_results_table.py
python analysis/20_robustness.py

# SMAM primary-data computations (order-independent; each updates cultural_vars.csv)
python analysis/compute_blr_smam.py
python analysis/compute_geo_smam.py
python analysis/compute_kgz_smam.py
python analysis/compute_uzb_smam.py
python analysis/compute_rus_smam.py
```

## Reproducibility Notes

- **Raw microdata files are not committed** (see `.gitignore` and `data/raw/README.md`). To reproduce the SMAM values you must download the MICS6 `.sav` files and the Russian census workbook yourself; instructions and URLs are in `data/raw/README.md`.
- **No interpolation is applied to any panel variable.** The eight missing cells (Uzbekistan remittances 2000–2004, Tajikistan remittances 2000–2001, Kyrgyzstan remittances 2023) are dropped from the estimation sample, not filled.
- **Standard errors are clustered by country (14 clusters)** in all panel regressions. This is below the conventional safe threshold (~30–50); the few-cluster fragility is flagged in `data/processed/diagnostics_results.txt`.

## Key Outputs

- `data/processed/panel.csv` — the working panel (336 rows, 14 countries × 24 years)
- `data/processed/crosssection.csv` — country averages + cultural variables (14 rows)
- `data/processed/results_main_table.md` — assembled M1–M4 table (paste-ready)
- `data/processed/{layerA,layerB,diagnostics,crosssection}_results.txt` — regression outputs
- `figures/` — trend plots, coverage map, cross-section scatter

## Install

```
pip install -r requirements.txt
```

Tested on Python 3.13. All dependencies pinned in `requirements.txt`.

## Limitations

- **n = 14 for the cross-section** limits statistical power; results are reported as descriptive associations, not causal estimates.
- **Muslim share and SMAM are collinear** in this sample; SMAM loses individual significance once Muslim share is included, and the two variables cannot be fully disentangled with 14 observations.
- **Cluster-robust SEs may be anti-conservative** given only 14 clusters. Wild-cluster bootstrap is recommended for future work.
- **SMAM temporal alignment is imperfect**: source-years range 2010–2022 across countries. This mixed-source, mixed-year approach follows the convention of the closest published literature (Dommaraju & Agadjanian 2008; Spoorenberg 2013, 2015; Kumo & Perugini 2024).