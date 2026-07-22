# Assembled main results table: M1, M2, M2h (Mundlak), M4 (two-way FE), FD, FD+yr.
"""
19_results_table.py
-------------------
Assemble the main Part 2 results table with SIX columns:
  M1    raw gap (pooled, year FE)
  M2    controlled gap (pooled, year FE)
  M2h   MUNDLAK HYBRID — between/within separation (the primary specification)
  M4    two-way fixed effects (country + year)
  FD    first-difference, no year effects
  FD+yr first-difference, with year effects

Why the Mundlak column is included:
  M2h is the design's primary between/within specification. Earlier versions of
  this table omitted it and showed the two first-difference columns instead,
  which gave less central models more prominence than the hybrid. For M2h the
  reported "CA" row is the between-country premium (the coefficient on the CA
  dummy); the individual control coefficients are the WITHIN coefficients (the
  between coefficients are collinear — see 17_diagnostics.py — and are not
  tabulated here to avoid over-interpretation).

Sample discipline:
  All columns are estimated on the controls-complete sample (drop any row with a
  missing lagged control). N is reported per column.

Standard errors:
  Clustered by country throughout (14 clusters). With only 14 clusters these are
  asymptotic and may be anti-conservative; the wild-cluster bootstrap in script
  20 (section G) is the few-cluster robustness check for the CA coefficients.

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

# Wild-cluster bootstrap p-value for M2h's "ca" coefficient (script 20, section G).
# The asymptotic clustered p-value overstates significance with only 14 clusters;
# the bootstrap p-value is used below for the M2h "ca" stars instead.
boot = pd.read_csv("data/processed/robustness_wild_bootstrap.csv").set_index("specification")
M2H_CA_BOOT_P = float(boot.loc["M2h Mundlak CA premium", "p_wild_bootstrap"])

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
# 3. Fit six models on the SAME sample                                        #
# --------------------------------------------------------------------------- #

# --- M1: raw gap ---
m1 = smf.ols("tfr ~ ca + C(year)", data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

# --- M2: controlled gap ---
f2 = "tfr ~ ca + " + " + ".join(CONTROLS) + " + C(year)"
m2 = smf.ols(f2, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

# --- M2h: Mundlak hybrid (between/within) ---
mh = sample.copy()
for c in CONTROLS:
    mh[f"{c}_mean"] = mh.groupby("country")[c].transform("mean")
    mh[f"{c}_dev"]  = mh[c] - mh[f"{c}_mean"]
between = [f"{c}_mean" for c in CONTROLS]
within  = [f"{c}_dev"  for c in CONTROLS]
f_mh = "tfr ~ ca + " + " + ".join(between + within) + " + C(year)"
m2h = smf.ols(f_mh, data=mh).fit(
    cov_type="cluster", cov_kwds={"groups": mh["country"]})

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

# For M2h: the CA row is the between-country premium; the control rows shown are
# the WITHIN coefficients (mapped back to the base control name for alignment).
M2H = {}
if "ca" in m2h.params.index:
    M2H["ca"] = (m2h.params["ca"], m2h.bse["ca"], m2h.pvalues["ca"])
for c in CONTROLS:
    wname = f"{c}_dev"
    if wname in m2h.params.index:
        M2H[c] = (m2h.params[wname], m2h.bse[wname], m2h.pvalues[wname])

M4 = pack_lm(m4, rows)
MF = pack_lm(fd, rows)

# FD+yr uses d_ prefixed names — map back to original names for table alignment
MFY = {}
for v_raw, v_d in zip(CONTROLS, d_controls):
    if v_d in fd_yr.params.index:
        MFY[v_raw] = (fd_yr.params[v_d], fd_yr.bse[v_d], fd_yr.pvalues[v_d])

ALL_MODELS = [M1, M2, M2H, M4, MF, MFY]

# --------------------------------------------------------------------------- #
# 5. Build the table                                                          #
# --------------------------------------------------------------------------- #
COL_HEADERS = ["M1: raw gap", "M2: + controls", "M2h: Mundlak",
               "M4: two-way FE", "FD", "FD + year FE"]
COL_NOTES = ["Pooled, year FE", "Pooled, year FE", "Between/within",
             "Country+year FE", "Within (Δ)", "Within (Δ)+yr"]

# --- (a) Plain-text ---
label_w = max(len(LABELS[r]) for r in rows) + 2
col_w = 17
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
            if M is M2H and r == "ca":
                pv = M2H_CA_BOOT_P
            top, bot = fmt(c, s, pv)
            cells_top.append(top.center(col_w))
            cells_bot.append(bot.center(col_w))
        else:
            cells_top.append("(absorbed)".center(col_w) if r == "ca" else "—".center(col_w))
            cells_bot.append("".center(col_w))
    lines.append(label + "".join(cells_top))
    lines.append(" " * label_w + "".join(cells_bot))

lines.append(sep)

# Column meta. NOTE on labels:
#  - "Year FE": M4/FD+yr include year effects; FD (no year FE) does not.
#  - "Country effects": M4 estimates country FE; M2h enters country means
#    (between block); FD/FD+yr REMOVE country effects by differencing — they do
#    not estimate country dummies. Labelled accordingly, not "Yes".
#  - R² type: FD/FD+yr R² is computed on DIFFERENCED observations, which is NOT
#    the fixed-effects within-R². Labelled "R²(differenced)".
year_fe   = ["Yes", "Yes", "Yes", "Yes", "No", "Yes"]
ctry_eff  = ["No", "No", "means (between)", "estimated (FE)",
             "differenced out", "differenced out"]
n_obs = [int(m1.nobs), int(m2.nobs), int(m2h.nobs), int(m4.nobs),
         int(fd.nobs), int(fd_yr.nobs)]
r2 = [m1.rsquared, m2.rsquared, m2h.rsquared, float(m4.rsquared_within),
      float(fd.rsquared), fd_yr.rsquared]
r2_label = ["R²(overall)", "R²(overall)", "R²(overall)", "R²(within)",
            "R²(differenced)", "R²(differenced)"]

for name, vals in [
    ("Year FE", year_fe),
    ("Country effects", ctry_eff),
    ("Observations", [str(n) for n in n_obs]),
    ("R²", [f"{v:.3f}" for v in r2]),
    ("R² type", r2_label),
]:
    lines.append(name.ljust(label_w) + "".join(v.center(col_w) for v in vals))

lines.append(sep)
lines.append("Cluster-robust SE (country) in parentheses.  *** p<0.01, ** p<0.05, * p<0.10.")
lines.append(f"All columns estimated on the controls-complete sample (N={n_obs[0]}).")
lines.append("M2h: 'CA' row is the between-country premium; control rows are the WITHIN")
lines.append("coefficients (deviations). The between-country control coefficients are")
lines.append("collinear (see 17_diagnostics) and are not tabulated individually.")
lines.append("M4: 'ca' absorbed by country FE; FD columns: 'ca' differenced away — by design.")
lines.append("FD / FD+yr R² is computed on first-differenced data, not the FE within-R².")
lines.append("Cluster count = 14; wild-cluster bootstrap (script 20, section G) was run for the")
lines.append("M1, M2, M2h, CA x remittances, and CA x urbanisation coefficients. The asymptotic")
lines.append("clustered p-value for M2h 'ca' is < 0.01, but the wild-cluster bootstrap p-value")
lines.append(f"is {M2H_CA_BOOT_P:.3f} (significant at 5%, not 1%); the stars for M2h 'ca' above")
lines.append("use the bootstrap p-value, not the asymptotic one.")
lines.append("Under-5 mortality is significant in the FD specification without year effects but")
lines.append("not after year effects are added; the estimate is therefore sensitive to the")
lines.append("inclusion of common year effects.")
lines.append("Layer B provides little stable evidence of within-country associations for the")
lines.append("selected macroeconomic indicators; it does not identify the mechanisms underlying")
lines.append("the persistent between-country regional difference.")

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
            if M is M2H and r == "ca":
                pv = M2H_CA_BOOT_P
            top, bot = fmt(c, s, pv)
            cells.append(f"{top}<br>{bot}")
        else:
            cells.append("*(absorbed)*" if r == "ca" else "—")
    md.append(f"| {label} | {' | '.join(cells)} |")
md.append(f"| Year FE | {' | '.join(year_fe)} |")
md.append(f"| Country effects | {' | '.join(ctry_eff)} |")
md.append(f"| Observations | {' | '.join(str(n) for n in n_obs)} |")
md.append(f"| R² | {' | '.join(f'{v:.3f}' for v in r2)} |")
md.append(f"| R² type | {' | '.join(r2_label)} |")
md_table = "\n".join(md)

md_footer = (
    "\n\n*Cluster-robust SE (country) in parentheses. "
    "\\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10.*  \n"
    f"*All columns estimated on controls-complete sample (N={n_obs[0]}). "
    "M2h: the CA row is the between-country premium; the control rows are the "
    "within coefficients (the collinear between-country control coefficients are "
    "not tabulated — see diagnostics). M4: CA dummy absorbed by country FE; "
    "FD columns: CA differenced away. FD / FD+yr R² is computed on first-"
    "differenced data, not the FE within-R². Cluster count = 14; wild-cluster "
    "bootstrap (script 20, section G) was run for M1, M2, M2h, CA x remittances, "
    "and CA x urbanisation. The asymptotic clustered p-value for M2h 'ca' is "
    f"< 0.01, but its wild-cluster bootstrap p-value is {M2H_CA_BOOT_P:.3f} "
    "(significant at 5%, not 1%); the stars for M2h 'ca' above use the bootstrap "
    "p-value, not the asymptotic one. Under-5 mortality is significant in FD "
    "without year FE but not once year effects are added, i.e. it is sensitive "
    "to the inclusion of common year effects.*\n"
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