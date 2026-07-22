# Provenance: `muslim_share` in `cultural_vars.csv`

## Source

- **Dataset title:** Religious Composition by Country, 2010-2020 (Pew Research
  Center, Pew-Templeton Global Religious Futures project), accompanying the
  report "How the Global Religious Landscape Changed From 2010 to 2020."
- **URL:** https://www.pewresearch.org/religion/feature/religious-composition-by-country-2010-2020/
- **File used:** `Religious Composition 2010-2020 (percentages).csv`, one of
  four CSVs bundled in the dataset ZIP downloaded from that page
  (`Religious-Composition-2010-2020-dataset.zip`).
- **Column used:** `Muslims`, filtered to `Year == 2020`.
- **Precision:** full unrounded percentages, as provided in the percentages
  file (e.g. `94.72725677` for Azerbaijan). Pew's bundled `README.txt` states
  the *unrounded* worksheet/figures "may be appropriate for regression
  analyses," while the rounded worksheet is for reporting; this project uses
  the unrounded percentages CSV throughout for statistical use, matching that
  guidance.
- **Download date:** 2026-07-22
- **SHA-256 checksum** of
  `data/raw/Religious Composition 2010-2020 (percentages).csv`:
  `0634d9ff61bcce57c4e9aecd5fdcdd84e01cd465089b39450fd85527c0351d2f`

## Rebuild command

```
python analysis/build_muslim_share.py
python analysis/validate_cultural_vars.py
```

The first command reads `data/raw/Religious Composition 2010-2020
(percentages).csv`, extracts the 2020 `Muslims` value for each of the 14
study countries (asserting exactly one Pew row per country), and overwrites
only `muslim_share`, `muslim_share_year`, and `muslim_share_source` in
`data/manual/cultural_vars.csv`. The second command fails loudly (exit 1) if
the resulting file has the wrong country set, missing values in the three
analytical columns, an out-of-range `muslim_share`, or a repeated
`muslim_share` value above 1.0 across countries (the class of bug this fixes:
Azerbaijan and Uzbekistan previously both carried 95.62 due to a copy-paste
error).

## Country name mapping

Pew uses the same country names as this project for all 14 study countries
(Armenia, Azerbaijan, Belarus, Estonia, Georgia, Kazakhstan, Kyrgyzstan,
Latvia, Lithuania, Moldova, Russia, Tajikistan, Ukraine, Uzbekistan) — no
renaming was necessary.
