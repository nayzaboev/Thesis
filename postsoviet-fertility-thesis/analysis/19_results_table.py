
# Assembled M1-M4 results table.
"""
19_results_table.py
-------------------
Assemble the main Part 2 results table: M1, M2, M4 (two-way FE), and the
first-difference within-country model, side by side.

Why re-fit here rather than parse 15/16 output:
  The .txt files in data/processed are human-readable reports, not machine data.
  Re-fitting in one place guarantees (a) all columns use the SAME sample so
  coefficients are directly comparable, and (b) the table can never silently
  drift out of sync with 15/16 if either gets re-run.

Sample discipline:
  All four columns are estimated on the controls-complete sample (drop any row
  with a missing lagged control). N is reported per column. This is the
  standard practice for a side-by-side specification table — it isolates
  "what changes when we add controls / change estimator" from "what changes
  when the sample changes".

Standard errors:
  Clustered by country throughout (14 clusters). The few-cluster fragility is
  documented in 17_diagnostics — flagged again in the table footer.

Outputs:
  data/processed/results_main_table.txt   (fixed-width, for inspection)
  data/processed/results_main_table.md    (markdown, paste-ready for the thesis)

Run from repo root:  python analysis/19_results_table.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, FirstDifferenceOLS

# --------------------------------------------------------------------------- #
# 1. Load panel and define the controls-complete estimation sample            #
# --------------------------------------------------------------------------- #
CONTROLS = [
    "log_gdp_ppp_lag1",
    "urban_pop_pct_lag1",
    "remittances_gdp_pct_lag1",
    "under5_mortality_lag1",
]

# Display labels for the table
LABELS = {
    "ca": "Central Asia dummy",
    "log_gdp_ppp_lag1": "log GDP per capita PPP (lag)",
    "urban_pop_pct_lag1": "Urban population %  (lag)",
    "remittances_gdp_pct_lag1": "Remittances % GDP  (lag)",
    "under5_mortality_lag1": "Under-5 mortality  (lag)",
}

p = pd.read_csv("data/processed/panel.csv")
# Same sample across all four columns:
sample = p.dropna(subset=["tfr"] + CONTROLS).copy()
print(f"Estimation sample: N = {len(sample)}, "
      f"countries = {sample['country'].nunique()}, "
      f"years = {sample['year'].min()}–{sample['year'].max()}\n")


# --------------------------------------------------------------------------- #
# 2. Helpers                                                                  #
# --------------------------------------------------------------------------- #
def stars(p):
    """Significance stars."""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def fmt(coef, se, p, decimals=3):
    """Coefficient with stars on first line, SE in parens on second."""
    return (f"{coef:+.{decimals}f}{stars(p)}",
            f"({se:.{decimals}f})")


# --------------------------------------------------------------------------- #
# 3. Fit the four models on the SAME sample                                   #
# --------------------------------------------------------------------------- #

# --- M1: TFR ~ CA + year effects (pooled OLS, cluster SE on country) ---
m1 = smf.ols("tfr ~ ca + C(year)", data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]}
)

# --- M2: M1 + lagged economic controls ---
f2 = "tfr ~ ca + " + " + ".join(CONTROLS) + " + C(year)"
m2 = smf.ols(f2, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]}
)

# --- M4: two-way FE (country + year) via linearmodels, cluster SE on entity ---
s_idx = sample.set_index(["country", "year"]).sort_index()
exog = s_idx[CONTROLS].assign(const=1.0)[["const"] + CONTROLS]
m4 = PanelOLS(s_idx["tfr"], exog,
              entity_effects=True, time_effects=True, drop_absorbed=True
              ).fit(cov_type="clustered", cluster_entity=True)

# --- M4-FD: first-difference (additional within-country robustness check) ---
fd = FirstDifferenceOLS(s_idx["tfr"], s_idx[CONTROLS]
                        ).fit(cov_type="clustered", cluster_entity=True)


# --------------------------------------------------------------------------- #
# 4. Pull coefficients / SEs / p-values into a uniform dict per model         #
# --------------------------------------------------------------------------- #
def pack_sm(res, names):
    """statsmodels result -> {name: (coef, se, p)}"""
    out = {}
    for n in names:
        if n in res.params.index:
            out[n] = (res.params[n], res.bse[n], res.pvalues[n])
    return out


def pack_lm(res, names):
    """linearmodels result -> {name: (coef, se, p)}"""
    out = {}
    for n in names:
        if n in res.params.index:
            out[n] = (float(res.params[n]),
                      float(res.std_errors[n]),
                      float(res.pvalues[n]))
    return out


rows = ["ca"] + CONTROLS  # row order in the table
M1 = pack_sm(m1, rows)
M2 = pack_sm(m2, rows)
M4 = pack_lm(m4, rows)      # 'ca' absorbed -> won't be in dict, handled below
MF = pack_lm(fd, rows)      # 'ca' differenced away -> same


# --------------------------------------------------------------------------- #
# 5. Build the table                                                          #
# --------------------------------------------------------------------------- #
COL_HEADERS = [
    "M1: raw gap",
    "M2: + controls",
    "M4: two-way FE",
    "FD: first-diff.",
]
COL_NOTES = [
    "Pooled OLS, year FE",
    "Pooled OLS, year FE",
    "Country + year FE",
    "Within (Δ)",
]

# --- (a) Plain-text fixed-width table ---
label_w = max(len(LABELS[r]) for r in rows) + 2
col_w = 21  # wide enough that "Pooled OLS, year FE" sub-headers don't touch
header = " " * label_w + "".join(h.center(col_w) for h in COL_HEADERS)
subhdr = " " * label_w + "".join(n.center(col_w) for n in COL_NOTES)
sep = "-" * len(header)
lines = [header, subhdr, sep]

for r in rows:
    label = LABELS[r].ljust(label_w)
    cells_top, cells_bot = [], []
    for M in (M1, M2, M4, MF):
        if r in M:
            c, s, pv = M[r]
            top, bot = fmt(c, s, pv)
            cells_top.append(top.center(col_w))
            cells_bot.append(bot.center(col_w))
        else:
            # Absorbed (FE) or differenced away (FD)
            cells_top.append("(absorbed)".center(col_w) if r == "ca" else "—".center(col_w))
            cells_bot.append("".center(col_w))
    lines.append(label + "".join(cells_top))
    lines.append(" " * label_w + "".join(cells_bot))

lines.append(sep)

# Footer rows: year FE, country FE, N, R²
def yes_no(b):
    return "Yes" if b else "No"


fe_year = [True, True, True, False]      # FD differences out year-invariant levels, no year dummies here
fe_ctry = [False, False, True, True]     # FD eliminates fixed country effects (level) by differencing
n_obs = [int(m1.nobs), int(m2.nobs), int(m4.nobs), int(fd.nobs)]
r2 = [m1.rsquared, m2.rsquared, float(m4.rsquared_within), float(fd.rsquared)]
r2_label = ["R² (overall)", "R² (overall)", "R² (within)", "R² (within)"]

for name, vals in [
    ("Year FE",        [yes_no(b) for b in fe_year]),
    ("Country FE",     [yes_no(b) for b in fe_ctry]),
    ("Observations",   [str(n) for n in n_obs]),
    ("R²",             [f"{v:.3f}" for v in r2]),
    ("R² type",        r2_label),
]:
    lines.append(name.ljust(label_w) + "".join(v.center(col_w) for v in vals))

lines.append(sep)
lines.append("Cluster-robust SE (country) in parentheses.  *** p<0.01, ** p<0.05, * p<0.10.")
lines.append("All columns estimated on the controls-complete sample (N=315).")
lines.append("M4: 'ca' is absorbed by country FE — that is by design; the between-country")
lines.append("question lives in Layer A (M1–M2), the within-country question in Layer B (M4, FD).")
lines.append("FD model: 'ca' is time-invariant and differenced away — same logic.")
lines.append("Cluster count = 14. Few-cluster fragility flagged in diagnostics (17_diagnostics).")
lines.append("Because the two-way FE residuals show strong serial correlation (AR(1)≈0.95), the")
lines.append("first-difference column is reported as an additional robustness check; it should be")
lines.append("interpreted cautiously as a short-run within-country association.")

text_table = "\n".join(lines)
print(text_table)

# --- (b) Markdown table (paste-ready) ---
md = []
md.append(f"| | {' | '.join(COL_HEADERS)} |")
md.append("|" + "---|" * (len(COL_HEADERS) + 1))
md.append(f"| | {' | '.join(COL_NOTES)} |")
for r in rows:
    label = LABELS[r]
    cells = []
    for M in (M1, M2, M4, MF):
        if r in M:
            c, s, pv = M[r]
            top, bot = fmt(c, s, pv)
            cells.append(f"{top}<br>{bot}")
        else:
            cells.append("*(absorbed)*" if r == "ca" else "—")
    md.append(f"| {label} | {' | '.join(cells)} |")
md.append(f"| Year FE | {' | '.join(yes_no(b) for b in fe_year)} |")
md.append(f"| Country FE | {' | '.join(yes_no(b) for b in fe_ctry)} |")
md.append(f"| Observations | {' | '.join(str(n) for n in n_obs)} |")
md.append(f"| R² | {' | '.join(f'{v:.3f}' for v in r2)} |")
md.append(f"| R² type | {' | '.join(r2_label)} |")

md_table = "\n".join(md)
md_footer = (
    "\n\n*Cluster-robust standard errors (country) in parentheses. "
    "\\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10.*  \n"
    "*All four columns estimated on the controls-complete sample (N=315). "
    "In M4 (two-way FE) the Central Asia dummy is absorbed by country effects; "
    "in the first-difference column it is differenced away — both by design. "
    "Cluster count is 14; few-cluster fragility is documented in the diagnostics "
    "appendix. Given AR(1) ≈ 0.95 in M4 residuals, the first-difference column "
    "is treated as an additional within-country robustness check; M4 is reported for "
    "transparency.*\n"
)

# --------------------------------------------------------------------------- #
# 6. Save                                                                     #
# --------------------------------------------------------------------------- #
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/results_main_table.txt", "w") as f:
    f.write(text_table)
with open("data/processed/results_main_table.md", "w") as f:
    f.write(md_table + md_footer)

print("\nSaved -> data/processed/results_main_table.txt")
print("Saved -> data/processed/results_main_table.md")