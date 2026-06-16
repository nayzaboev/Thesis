# Post-Soviet Fertility Thesis

**Title:** Fertility Trends in the Post-Soviet Space: A Comparative 
Analysis of Central Asia and Other Former Soviet Countries (2000–2023)

**Degree:** Master's Thesis — University of Bonn

## Project Structure
- `data/raw/` — raw downloads (WPP 2024, World Bank); git-ignored
- `data/processed/` — clean master table used in analysis
- `analysis/` — Stata or R or Python scripts (numbered in order of execution)
- `figures/` — generated plots and charts
- `drafts/` — chapter writing (one folder per chapter)
- `sources/` — literature PDFs; git-ignored

## Data Sources
- UN World Population Prospects 2024 (population.un.org/wpp)
- World Bank World Development Indicators (data.worldbank.org)
- Tajikistan DHS 2023 (dhsprogram.com)
- UNFPA Uzbekistan Fertility Survey 2023 (Kurylo et al.)

## Countries
**Central Asia:** Kazakhstan, Kyrgyzstan, Tajikistan, Uzbekistan

**Slavic:** Russia, Ukraine, Belarus, Moldova

**Baltic:** Estonia, Latvia, Lithuania

**Caucasus:** Armenia, Georgia, Azerbaijan

## How to Run
1. Place raw WPP CSV in `data/raw/`
2. Run `analysis/01_clean_data.py` → generates `data/processed/master_tfr.csv`
3. Run `analysis/02_trend_plots.py` → generates figures in `figures/`