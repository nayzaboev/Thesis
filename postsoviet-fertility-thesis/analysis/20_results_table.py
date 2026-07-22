# Assembled main results table: Panel A (M1, M2, M2h) + Panel B (M4, FD, FD+yr).
"""
20_results_table.py
-------------------
Assemble the main Part 2 results table as TWO PANELS, so estimands that are
not comparable are no longer placed in shared rows inviting cross-column
coefficient comparison:

  PANEL A — Regional fertility difference (Central Asia vs rest):
    M1    raw gap (pooled, year FE)
    M2    controlled gap (pooled, year FE)
    M2h   MUNDLAK HYBRID — between/within separation (the primary specification)

  PANEL B — Within-country associations (no between-country gap estimated):
    M4    two-way fixed effects (country + year)
    FD    first-difference, no year effects
    FD+yr first-difference, with year effects

Why the Mundlak column is in Panel A, not Panel B:
  M2h is the design's primary between/within specification, but the "CA" row it
  reports is a BETWEEN-country coefficient, so it belongs with M1/M2 (Panel A),
  not with the within-only M4/FD/FD+yr models (Panel B). M2h's own control
  coefficients are, however, WITHIN-country estimates — they are labelled
  explicitly as within-deviations in the table (marked "(w)") so they are not
  mistaken for the between-country regional-gap estimand that the rest of
  Panel A reports. The between-country control coefficients from M2h are
  collinear (see 17_diagnostics.py) and are not tabulated.

Sample discipline:
  All columns are estimated on the controls-complete sample (drop any row with a
  missing lagged control). N is reported per column.

Standard errors and inference:
  Clustered by country throughout (14 clusters). With only 14 clusters these are
  asymptotic and may be anti-conservative, so — rather than asymptotic
  significance stars — coefficient cells report the point estimate and its 95%
  cluster-robust confidence interval. For the Central Asia ("ca") coefficients
  in Panel A, a separate "Bootstrap p (CA)" row reports the wild-cluster
  bootstrap p-value from script 19 (section G) alongside the asymptotic one,
  since that bootstrap is the few-cluster robustness check for exactly these
  coefficients; Panel B has no such row because 'ca' is absorbed (M4) or
  differenced away (FD/FD+yr) in every within-country model.

Outputs:
  data/processed/results_main_table.txt   (fixed-width, for inspection)
  data/processed/results_main_table.md    (markdown, paste-ready for the thesis)

Run from repo root:  python analysis/20_results_table.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, FirstDifferenceOLS

from _assertions import assert_year_continuity

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
# Year-continuity guard on the FULL panel (before the complete-case drop
# below), so shift()/diff() downstream never silently treats a missing
# country-year row as a one-year change.
assert_year_continuity(p)
sample = p.dropna(subset=["tfr"] + CONTROLS).copy()
print(f"Estimation sample: N = {len(sample)}, "
      f"countries = {sample['country'].nunique()}, "
      f"years = {sample['year'].min()}–{sample['year'].max()}\n")

# Wild-cluster bootstrap p-values for the CA coefficients in Panel A (script 19,
# section G). The asymptotic clustered p-value overstates significance with
# only 14 clusters, so these are reported alongside the 95% CI in a separate
# "Bootstrap p (CA)" row rather than folded into significance stars.
boot = pd.read_csv("data/processed/robustness_wild_bootstrap.csv").set_index("specification")
M1_CA_BOOT_P  = float(boot.loc["M1 raw CA premium", "p_wild_bootstrap"])
M2_CA_BOOT_P  = float(boot.loc["M2 controlled CA premium", "p_wild_bootstrap"])
M2H_CA_BOOT_P = float(boot.loc["M2h Mundlak CA premium", "p_wild_bootstrap"])

# --------------------------------------------------------------------------- #
# 2. Helpers                                                                  #
# --------------------------------------------------------------------------- #
def fmt_ci(coef, lo, hi, decimals=3, marker=""):
    """Coefficient + 95% cluster-robust CI, no significance stars (see module
    docstring: stars are replaced by CIs; the CA coefficients additionally get
    a bootstrap p-value row)."""
    return (f"{coef:+.{decimals}f}{marker}", f"[{lo:+.{decimals}f}, {hi:+.{decimals}f}]")

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

# Staleness guard: the bootstrap p-value above (M2H_CA_BOOT_P) is only valid for
# THIS fitted CA coefficient. If robustness_wild_bootstrap.csv was produced by an
# older panel/model, the freshly estimated coefficient will diverge from the coef
# stored in the CSV (rounded to 4 decimals), and the p-value would be silently
# stale. Tolerance is 1e-3 to accommodate that rounding.
boot_m2h_coef = float(boot.loc["M2h Mundlak CA premium", "coef"])
assert abs(m2h.params["ca"] - boot_m2h_coef) < 1e-3, (
    f"M2h CA coefficient ({m2h.params['ca']:.4f}) does not match the coefficient "
    f"stored in robustness_wild_bootstrap.csv ({boot_m2h_coef:.4f}). The wild-"
    "cluster bootstrap p-value is stale — re-run 19_robustness.py to regenerate "
    "robustness_wild_bootstrap.csv before rebuilding this table."
)

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
# NOTE on year gaps: fd_df derives from `sample`, the controls-complete
# subset of the full panel `p` (asserted gap-free above), so a country can
# legitimately be missing a year here (e.g. a missing lagged control) and
# diff() will then span more than one year for that pair. This is expected,
# data-driven missingness, not a data error, so continuity is intentionally
# not asserted at this site.
for col in ["tfr"] + CONTROLS:
    fd_df[f"d_{col}"] = fd_df.groupby("country")[col].diff()
fd_df = fd_df.dropna(subset=[f"d_{c}" for c in ["tfr"] + CONTROLS])
d_controls = [f"d_{c}" for c in CONTROLS]
fd_yr = smf.ols(f"d_tfr ~ {' + '.join(d_controls)} + C(year)", data=fd_df).fit(
    cov_type="cluster", cov_kwds={"groups": fd_df["country"]})

# --------------------------------------------------------------------------- #
# 4. Pack coefficients (coef, SE, p, 95% CI lower, 95% CI upper)              #
# --------------------------------------------------------------------------- #
def pack_sm(res, names):
    out = {}
    ci = res.conf_int()
    for n in names:
        if n in res.params.index:
            lo, hi = ci.loc[n, 0], ci.loc[n, 1]
            out[n] = (res.params[n], res.bse[n], res.pvalues[n], lo, hi)
    return out

def pack_lm(res, names):
    out = {}
    ci = res.conf_int()
    for n in names:
        if n in res.params.index:
            lo, hi = float(ci.loc[n, "lower"]), float(ci.loc[n, "upper"])
            out[n] = (float(res.params[n]), float(res.std_errors[n]),
                      float(res.pvalues[n]), lo, hi)
    return out

rows = ["ca"] + CONTROLS
M1 = pack_sm(m1, rows)
M2 = pack_sm(m2, rows)

# For M2h: the CA row is the between-country premium; the control rows shown are
# the WITHIN coefficients (mapped back to the base control name for alignment,
# and marked "(w)" in the table so they are not read as the same between-
# country estimand as the CA row or as M1/M2's pooled control coefficients).
M2H = {}
ci_m2h = m2h.conf_int()
if "ca" in m2h.params.index:
    lo, hi = ci_m2h.loc["ca", 0], ci_m2h.loc["ca", 1]
    M2H["ca"] = (m2h.params["ca"], m2h.bse["ca"], m2h.pvalues["ca"], lo, hi)
for c in CONTROLS:
    wname = f"{c}_dev"
    if wname in m2h.params.index:
        lo, hi = ci_m2h.loc[wname, 0], ci_m2h.loc[wname, 1]
        M2H[c] = (m2h.params[wname], m2h.bse[wname], m2h.pvalues[wname], lo, hi)

M4 = pack_lm(m4, CONTROLS)
MF = pack_lm(fd, CONTROLS)

# FD+yr uses d_ prefixed names — map back to original names for table alignment
MFY = {}
ci_fdyr = fd_yr.conf_int()
for v_raw, v_d in zip(CONTROLS, d_controls):
    if v_d in fd_yr.params.index:
        lo, hi = ci_fdyr.loc[v_d, 0], ci_fdyr.loc[v_d, 1]
        MFY[v_raw] = (fd_yr.params[v_d], fd_yr.bse[v_d], fd_yr.pvalues[v_d], lo, hi)

# --------------------------------------------------------------------------- #
# 5. Build the table as TWO PANELS                                           #
#    Panel A: regional fertility difference (between-country gap): M1/M2/M2h #
#    Panel B: within-country associations (no gap estimand): M4/FD/FD+yr     #
# --------------------------------------------------------------------------- #
CA_BOOT_P = {"M1": M1_CA_BOOT_P, "M2": M2_CA_BOOT_P, "M2h": M2H_CA_BOOT_P}


def build_panel(panel_rows, models, col_headers, col_notes, n_obs, r2, r2_label,
                 year_fe, ctry_eff, ca_boot_row=False, within_markers=None):
    """Render one panel as (plain-text lines, markdown lines). `within_markers`
    maps column index -> set of row names whose coefficient should be flagged
    "(w)" as a within-deviation estimate (used for M2h's control rows)."""
    within_markers = within_markers or {}
    label_w = max(len(LABELS[r]) for r in panel_rows) + 2
    col_w = 19
    header = " " * label_w + "".join(h.center(col_w) for h in col_headers)
    subhdr = " " * label_w + "".join(n.center(col_w) for n in col_notes)
    sep = "-" * len(header)
    txt = [header, subhdr, sep]

    for r in panel_rows:
        label = LABELS[r].ljust(label_w)
        cells_top, cells_bot = [], []
        for i, M in enumerate(models):
            if r in M:
                c, s, pv, lo, hi = M[r]
                marker = "(w)" if r in within_markers.get(i, set()) else ""
                top, bot = fmt_ci(c, lo, hi, marker=marker)
                cells_top.append(top.center(col_w))
                cells_bot.append(bot.center(col_w))
            else:
                cells_top.append(("(absorbed)" if r == "ca" else "—").center(col_w))
                cells_bot.append("".center(col_w))
        txt.append(label + "".join(cells_top))
        txt.append(" " * label_w + "".join(cells_bot))
        if ca_boot_row and r == "ca":
            boot_label = "  Bootstrap p (CA, script 19-G)".ljust(label_w)
            boot_cells = [f"p = {CA_BOOT_P[h.split(':')[0]]:.3f}".center(col_w)
                          for h in col_headers]
            txt.append(boot_label + "".join(boot_cells))
    txt.append(sep)

    for name, vals in [
        ("Year FE", year_fe),
        ("Country effects", ctry_eff),
        ("Observations", [str(n) for n in n_obs]),
        ("R²", [f"{v:.3f}" for v in r2]),
        ("R² type", r2_label),
    ]:
        txt.append(name.ljust(label_w) + "".join(v.center(col_w) for v in vals))
    txt.append(sep)

    md = [f"| | {' | '.join(col_headers)} |",
          "|" + "---|" * (len(col_headers) + 1),
          f"| | {' | '.join(col_notes)} |"]
    for r in panel_rows:
        cells = []
        for i, M in enumerate(models):
            if r in M:
                c, s, pv, lo, hi = M[r]
                marker = "(w)" if r in within_markers.get(i, set()) else ""
                top, bot = fmt_ci(c, lo, hi, marker=marker)
                cells.append(f"{top}<br>{bot}")
            else:
                cells.append("*(absorbed)*" if r == "ca" else "—")
        md.append(f"| {LABELS[r]} | {' | '.join(cells)} |")
        if ca_boot_row and r == "ca":
            boot_cells = [f"p = {CA_BOOT_P[h.split(':')[0]]:.3f}" for h in col_headers]
            md.append(f"| *Bootstrap p (CA, script 19-G)* | {' | '.join(boot_cells)} |")
    md.append(f"| Year FE | {' | '.join(year_fe)} |")
    md.append(f"| Country effects | {' | '.join(ctry_eff)} |")
    md.append(f"| Observations | {' | '.join(str(n) for n in n_obs)} |")
    md.append(f"| R² | {' | '.join(f'{v:.3f}' for v in r2)} |")
    md.append(f"| R² type | {' | '.join(r2_label)} |")
    return txt, md


# --- Panel A: regional fertility difference ---
panelA_rows = ["ca"] + CONTROLS
panelA_models = [M1, M2, M2H]
panelA_headers = ["M1: raw gap", "M2: + controls", "M2h: Mundlak"]
panelA_notes = ["Pooled, year FE", "Pooled, year FE", "Between/within"]
panelA_n = [int(m1.nobs), int(m2.nobs), int(m2h.nobs)]
panelA_r2 = [m1.rsquared, m2.rsquared, m2h.rsquared]
panelA_r2lab = ["R²(overall)", "R²(overall)", "R²(overall)"]
panelA_yrfe = ["Yes", "Yes", "Yes"]
panelA_ctry = ["No", "No", "means (between)"]
panelA_within = {2: set(CONTROLS)}  # M2h (index 2): control rows are within-deviations

txtA, mdA = build_panel(panelA_rows, panelA_models, panelA_headers, panelA_notes,
                         panelA_n, panelA_r2, panelA_r2lab, panelA_yrfe, panelA_ctry,
                         ca_boot_row=True, within_markers=panelA_within)

# --- Panel B: within-country associations (no 'ca' row — absorbed/differenced) ---
panelB_rows = CONTROLS
panelB_models = [M4, MF, MFY]
panelB_headers = ["M4: two-way FE", "FD", "FD + year FE"]
panelB_notes = ["Country+year FE", "Within (Δ)", "Within (Δ)+yr"]
panelB_n = [int(m4.nobs), int(fd.nobs), int(fd_yr.nobs)]
panelB_r2 = [float(m4.rsquared_within), float(fd.rsquared), fd_yr.rsquared]
panelB_r2lab = ["R²(within)", "R²(differenced)", "R²(differenced)"]
panelB_yrfe = ["Yes", "No", "Yes"]
panelB_ctry = ["estimated (FE)", "differenced out", "differenced out"]

txtB, mdB = build_panel(panelB_rows, panelB_models, panelB_headers, panelB_notes,
                         panelB_n, panelB_r2, panelB_r2lab, panelB_yrfe, panelB_ctry,
                         ca_boot_row=False)

# --------------------------------------------------------------------------- #
# Footnotes (shared)                                                          #
# --------------------------------------------------------------------------- #
FOOTNOTES_TXT = [
    "Cluster-robust SE (country) in brackets are 95% confidence intervals, not raw SEs;",
    "significance stars are not used (see module docstring). Coefficient / [95% CI] per cell.",
    f"All panels estimated on the controls-complete sample (N={panelA_n[0]}).",
    "Panel A: 'ca' is the between-country Central Asia premium in M1/M2/M2h. M2h's control",
    "rows are marked (w) — they are WITHIN-country deviations, not the same between-country",
    "estimand as the 'ca' row or as M1/M2's pooled control coefficients; M2h's between-country",
    "control coefficients are collinear (see 17_diagnostics) and are not tabulated.",
    "Panel B: within-country associations only — no between-country gap is estimated here.",
    "M4: 'ca' absorbed by country FE; FD / FD+yr: 'ca' differenced away — by design, not shown.",
    "FD / FD+yr R² is computed on first-differenced data, not the FE within-R².",
    "Cluster count = 14; wild-cluster bootstrap (script 19, section G) was run for the CA",
    "coefficients in M1, M2, M2h (reported in the 'Bootstrap p (CA)' row of Panel A) and for",
    "CA x remittances / CA x urbanisation (not tabulated here). The asymptotic clustered",
    "p-value for M2h 'ca' is < 0.01, but its wild-cluster bootstrap p-value is "
    f"{M2H_CA_BOOT_P:.3f}",
    "(significant at 5%, not 1%) — the CI and bootstrap-p row for M2h above reflect that gap",
    "between asymptotic and few-cluster inference; the asymptotic p-value alone overstates it.",
    "Under-5 mortality is significant in the FD specification without year effects but not",
    "after year effects are added; the estimate is therefore sensitive to the inclusion of",
    "common year effects.",
    "Layer B (Panel B) provides little stable evidence of within-country associations for the",
    "selected macroeconomic indicators; it does not identify the mechanisms underlying the",
    "persistent between-country regional difference documented in Panel A.",
]

text_table = (
    "PANEL A — Regional fertility difference (Central Asia vs rest)\n" + "=" * 70 + "\n"
    + "\n".join(txtA)
    + "\n\nPANEL B — Within-country associations\n" + "=" * 70 + "\n"
    + "\n".join(txtB)
    + "\n" + "\n".join(FOOTNOTES_TXT)
)
print(text_table)

md_footer = (
    "\n\n*Brackets are 95% cluster-robust confidence intervals, not raw standard errors; "
    "significance stars are not used (see script docstring).*  \n"
    f"*All panels estimated on the controls-complete sample (N={panelA_n[0]}). "
    "Panel A: 'ca' is the between-country Central Asia premium in M1/M2/M2h; M2h's control "
    "rows are marked (w) as within-country deviations (not the same estimand as 'ca' or as "
    "M1/M2's pooled controls); M2h's collinear between-country control coefficients are not "
    "tabulated — see diagnostics. Panel B: within-country associations only, no between-"
    "country gap estimated. M4: 'ca' absorbed by country FE; FD / FD+yr: 'ca' differenced "
    "away. FD / FD+yr R² is computed on first-differenced data, not the FE within-R². "
    "Cluster count = 14; wild-cluster bootstrap (script 19, section G) was run for the CA "
    "coefficients in M1, M2, M2h (Panel A's 'Bootstrap p (CA)' row) and for CA x remittances "
    "/ CA x urbanisation. The asymptotic clustered p-value for M2h 'ca' is < 0.01, but its "
    f"wild-cluster bootstrap p-value is {M2H_CA_BOOT_P:.3f} (significant at 5%, not 1%) — "
    "the CI and bootstrap-p row above reflect that gap rather than the asymptotic p-value "
    "alone. Under-5 mortality is significant in FD without year FE but not once year effects "
    "are added, i.e. it is sensitive to the inclusion of common year effects.*\n"
)

md_table = (
    "## Panel A — Regional fertility difference (Central Asia vs rest)\n\n"
    + "\n".join(mdA)
    + "\n\n## Panel B — Within-country associations\n\n"
    + "\n".join(mdB)
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