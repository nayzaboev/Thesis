
# Assembled M1-M4 results table.
"""
19_results_table.py
-------------------
Assemble the main Part 2 results table: M1, M2, M4 (two-way FE), FD without
year effects, and FD with year effects — five columns side by side.

Why five columns:
  M1/M2 answer the between-country question (Layer A).
  M4/FD/FD+yr answer the within-country question (Layer B).
  Showing FD both with and without year effects exposes the sensitivity of
  the under-5 mortality result (significant without year FE, disappears with
  year FE — flagged automatically in script 16).

Sample discipline:
  All five columns are estimated on the controls-complete sample (drop any
  row with a missing lagged control). N is reported per column.

Standard errors:
  Clustered by country throughout (14 clusters).

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

LABELS = {
    "ca": "Central Asia dummy",
    "log_gdp_ppp_lag1": "log GDP per capita PPP (lag)",
    "urban_pop_pct_lag1": "Urban population %  (lag)",
    "remittances_gdp_pct_lag1": "Remittances % GDP  (lag)",
    "under5_mortality_lag1": "Under-5 mortality  (lag)",
}

p = pd.read_csv("data/processed/panel.csv")
sample = p.dropna(subset=["tfr"] + CONTROLS).copy()
print(f"Estimation sample: N = {len(sample)}, "
      f"countries = {sample['country'].nunique()}, "
      f"years = {sample['year'].min()}–{sample['year'].max()}\n")

# --------------------------------------------------------------------------- #
# 2. Helpers                                                                  #
# --------------------------------------------------------------------------- #
def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def fmt(coef, se, p, decimals=3):
    return (f"{coef:+.{decimals}f}{stars(p)}", f"({se:.{decimals}f})")

# --------------------------------------------------------------------------- #
# 3. Fit five models on the SAME sample                                       #
# --------------------------------------------------------------------------- #

# --- M1: raw gap ---
m1 = smf.ols("tfr ~ ca + C(year)", data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

# --- M2: controlled gap ---
f2 = "tfr ~ ca + " + " + ".join(CONTROLS) + " + C(year)"
m2 = smf.ols(f2, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

# --- M4: two-way FE ---
s_idx = sample.set_index(["country", "year"]).sort_index()
exog = s_idx[CONTROLS].assign(const=1.0)[["const"] + CONTROLS]
m4 = PanelOLS(s_idx["tfr"], exog,
              entity_effects=True, time_effects=True, drop_absorbed=True
              ).fit(cov_type="clustered", cluster_entity=True)

# --- FD: first-difference WITHOUT year effects ---
fd = FirstDifferenceOLS(s_idx["tfr"], s_idx[CONTROLS]
                        ).fit(cov_type="clustered", cluster_entity=True)

# --- FD+yr: first-difference WITH year effects (manual differencing + OLS) ---
fd_df = sample.sort_values(["country", "year"]).copy()
for col in ["tfr"] + CONTROLS:
    fd_df[f"d_{col}"] = fd_df.groupby("country")[col].diff()
fd_df = fd_df.dropna(subset=[f"d_{c}" for c in ["tfr"] + CONTROLS])
d_controls = [f"d_{c}" for c in CONTROLS]
fd_yr = smf.ols(f"d_tfr ~ {' + '.join(d_controls)} + C(year)", data=fd_df).fit(
    cov_type="cluster", cov_kwds={"groups": fd_df["country"]})

# --------------------------------------------------------------------------- #
# 4. Pack coefficients                                                        #
# --------------------------------------------------------------------------- #
def pack_sm(res, names):
    out = {}
    for n in names:
        if n in res.params.index:
            out[n] = (res.params[n], res.bse[n], res.pvalues[n])
    return out

def pack_lm(res, names):
    out = {}
    for n in names:
        if n in res.params.index:
            out[n] = (float(res.params[n]), float(res.std_errors[n]), float(res.pvalues[n]))
    return out

rows = ["ca"] + CONTROLS
M1 = pack_sm(m1, rows)
M2 = pack_sm(m2, rows)
M4 = pack_lm(m4, rows)
MF = pack_lm(fd, rows)

# FD+yr uses d_ prefixed names — map back to original names for table alignment
MFY = {}
for v_raw, v_d in zip(CONTROLS, d_controls):
    if v_d in fd_yr.params.index:
        MFY[v_raw] = (fd_yr.params[v_d], fd_yr.bse[v_d], fd_yr.pvalues[v_d])

ALL_MODELS = [M1, M2, M4, MF, MFY]

# --------------------------------------------------------------------------- #
# 5. Build the table                                                          #
# --------------------------------------------------------------------------- #
COL_HEADERS = ["M1: raw gap", "M2: + controls", "M4: two-way FE",
               "FD", "FD + year FE"]
COL_NOTES = ["Pooled, year FE", "Pooled, year FE", "Country+year FE",
             "Within (Δ)", "Within (Δ)+yr"]

# --- (a) Plain-text ---
label_w = max(len(LABELS[r]) for r in rows) + 2
col_w = 18
header = " " * label_w + "".join(h.center(col_w) for h in COL_HEADERS)
subhdr = " " * label_w + "".join(n.center(col_w) for n in COL_NOTES)
sep = "-" * len(header)
lines = [header, subhdr, sep]

for r in rows:
    label = LABELS[r].ljust(label_w)
    cells_top, cells_bot = [], []
    for M in ALL_MODELS:
        if r in M:
            c, s, pv = M[r]
            top, bot = fmt(c, s, pv)
            cells_top.append(top.center(col_w))
            cells_bot.append(bot.center(col_w))
        else:
            cells_top.append("(absorbed)".center(col_w) if r == "ca" else "—".center(col_w))
            cells_bot.append("".center(col_w))
    lines.append(label + "".join(cells_top))
    lines.append(" " * label_w + "".join(cells_bot))

lines.append(sep)

fe_year = [True, True, True, False, True]
fe_ctry = [False, False, True, True, True]
n_obs = [int(m1.nobs), int(m2.nobs), int(m4.nobs), int(fd.nobs), int(fd_yr.nobs)]
r2 = [m1.rsquared, m2.rsquared, float(m4.rsquared_within), float(fd.rsquared), fd_yr.rsquared]
r2_label = ["R²(overall)", "R²(overall)", "R²(within)", "R²(within)", "R²(overall)"]

for name, vals in [
    ("Year FE", [("Yes" if b else "No") for b in fe_year]),
    ("Country FE", [("Yes" if b else "No") for b in fe_ctry]),
    ("Observations", [str(n) for n in n_obs]),
    ("R²", [f"{v:.3f}" for v in r2]),
    ("R² type", r2_label),
]:
    lines.append(name.ljust(label_w) + "".join(v.center(col_w) for v in vals))

lines.append(sep)
lines.append("Cluster-robust SE (country) in parentheses.  *** p<0.01, ** p<0.05, * p<0.10.")
lines.append(f"All columns estimated on the controls-complete sample (N={n_obs[0]}).")
lines.append("M4: 'ca' absorbed by country FE; FD columns: 'ca' differenced away — both by design.")
lines.append("Cluster count = 14. Few-cluster fragility flagged in diagnostics (17_diagnostics).")
lines.append("The FD column WITHOUT year effects shows under-5 mortality as significant (p=0.013);")
lines.append("with year effects (rightmost column) the result disappears (p~0.97), indicating it")
lines.append("was driven by common time shocks rather than within-country variation. Layer B as a")
lines.append("whole is best treated as a robustness exercise: within-country economic effects are")
lines.append("small and fragile, consistent with a structural between-country fertility gap.")

text_table = "\n".join(lines)
print(text_table)

# --- (b) Markdown table ---
md = []
md.append(f"| | {' | '.join(COL_HEADERS)} |")
md.append("|" + "---|" * (len(COL_HEADERS) + 1))
md.append(f"| | {' | '.join(COL_NOTES)} |")
for r in rows:
    label = LABELS[r]
    cells = []
    for M in ALL_MODELS:
        if r in M:
            c, s, pv = M[r]
            top, bot = fmt(c, s, pv)
            cells.append(f"{top}<br>{bot}")
        else:
            cells.append("*(absorbed)*" if r == "ca" else "—")
    md.append(f"| {label} | {' | '.join(cells)} |")
md.append(f"| Year FE | {' | '.join(('Yes' if b else 'No') for b in fe_year)} |")
md.append(f"| Country FE | {' | '.join(('Yes' if b else 'No') for b in fe_ctry)} |")
md.append(f"| Observations | {' | '.join(str(n) for n in n_obs)} |")
md.append(f"| R² | {' | '.join(f'{v:.3f}' for v in r2)} |")
md.append(f"| R² type | {' | '.join(r2_label)} |")
md_table = "\n".join(md)

md_footer = (
    "\n\n*Cluster-robust SE (country) in parentheses. "
    "\\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10.*  \n"
    f"*All columns estimated on controls-complete sample (N={n_obs[0]}). "
    "M4: CA dummy absorbed by country FE; FD columns: differenced away. "
    "Cluster count = 14; few-cluster fragility documented in diagnostics. "
    "Under-5 mortality is significant in FD without year FE but disappears "
    "with year FE, indicating sensitivity to common time shocks.*\n"
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