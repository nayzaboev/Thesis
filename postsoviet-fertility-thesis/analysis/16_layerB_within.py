"""Part 2 scaffold."""
# Layer B (within): Model 4, two-way FE + first-difference + IPS unit root test.
"""
16_layerB_within.py
-------------------
LAYER B — the within-country question: over time, inside a given country, what
moves fertility? This is the robustness defence against the spurious-regression
critique (the old pooled model had Durbin-Watson = 0.13).

Three estimators / tests:

  (A) TWO-WAY FIXED EFFECTS (country + year), via linearmodels.PanelOLS,
      clustered SE by country.
      NOTE: the Central Asia dummy and any time-invariant variable are ABSORBED
      by country effects and cannot be estimated here — that is expected, not a
      bug. Layer A (script 15) is where the CA premium lives.

  (B) FIRST-DIFFERENCE model: regress delta-TFR on delta-covariates (lagged
      covariates differenced). A second within-estimator robust to different
      assumptions; if FE and FD broadly agree, the within story is credible.

  (C) DIAGNOSTICS for non-stationarity / serial correlation:
      - Im-Pesaran-Shin (IPS) panel unit-root test on TFR and each covariate
        (heterogeneous-panel test; handles the unbalanced structure reasonably).
      - Wooldridge test for AR(1) serial correlation in the FE residuals.

Controls (lagged 1 year): log_gdp_ppp_lag1, urban_pop_pct_lag1,
remittances_gdp_pct_lag1, under5_mortality_lag1

Output: data/processed/layerB_results.txt
Run from repo root:  python analysis/16_layerB_within.py
"""

import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, FirstDifferenceOLS

CONTROLS = ["log_gdp_ppp_lag1", "urban_pop_pct_lag1",
            "remittances_gdp_pct_lag1", "under5_mortality_lag1"]

p = pd.read_csv("data/processed/panel.csv")

# linearmodels needs a MultiIndex (entity, time)
p = p.sort_values(["country", "year"]).copy()
p_idx = p.set_index(["country", "year"])

lines = []
def out(s):
    print(s); lines.append(s)

# ----------------------------------------------------------------------------
# (A) Two-way fixed effects
# ----------------------------------------------------------------------------
d = p_idx.dropna(subset=["tfr"] + CONTROLS).copy()
exog = d[CONTROLS]
exog = exog.assign(const=1.0)[["const"] + CONTROLS]  # const first
fe = PanelOLS(d["tfr"], exog, entity_effects=True, time_effects=True, drop_absorbed=True)
fe_res = fe.fit(cov_type="clustered", cluster_entity=True)

out("="*70)
out("(A) TWO-WAY FIXED EFFECTS  (country + year, clustered SE by country)")
out("="*70)
out(f"N = {int(fe_res.nobs)}   entities = {d.index.get_level_values(0).nunique()}   "
    f"within R2 = {fe_res.rsquared_within:.3f}")
out("Note: CA dummy and time-invariant vars are absorbed by country effects (expected).\n")
for v in CONTROLS:
    if v in fe_res.params.index:
        out(f"  {v:28s}: {fe_res.params[v]:+.4f}  (SE {fe_res.std_errors[v]:.4f}, "
            f"p={fe_res.pvalues[v]:.3f})")

# ----------------------------------------------------------------------------
# (B) First-difference model
# ----------------------------------------------------------------------------
# FirstDifferenceOLS cannot include a constant (differenced away) and needs
# variation; use the same controls.
try:
    fd = FirstDifferenceOLS(d["tfr"], d[CONTROLS])
    fd_res = fd.fit(cov_type="clustered", cluster_entity=True)
    out("\n" + "="*70)
    out("(B) FIRST-DIFFERENCE model  (delta-TFR on delta-controls)")
    out("="*70)
    out(f"N = {int(fd_res.nobs)}   within-ish R2 = {fd_res.rsquared:.3f}\n")
    for v in CONTROLS:
        if v in fd_res.params.index:
            out(f"  {v:28s}: {fd_res.params[v]:+.4f}  (SE {fd_res.std_errors[v]:.4f}, "
                f"p={fd_res.pvalues[v]:.3f})")
    out("\n  Compare signs/magnitudes to (A). Broad agreement => within story is robust.")
except Exception as e:
    out(f"\n(B) First-difference model could not be estimated: {e}")

# ----------------------------------------------------------------------------
# (C1) Im-Pesaran-Shin panel unit-root test
#      Implemented manually: per-country ADF(1) t-stats, averaged (t-bar),
#      reported with the per-series mean ADF stat. (No external panel-root pkg.)
# ----------------------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller

def ips_tbar(panel_df, var):
    """Average of per-country ADF t-statistics (IPS t-bar, descriptive form)."""
    stats = []
    for c, g in panel_df.groupby(level=0):
        s = g[var].dropna().values
        if len(s) >= 8 and np.std(s) > 1e-8:
            try:
                stats.append(adfuller(s, maxlag=1, autolag=None, regression="c")[0])
            except Exception:
                pass
    return (np.mean(stats), len(stats)) if stats else (np.nan, 0)

out("\n" + "="*70)
out("(C1) Im-Pesaran-Shin panel unit-root (t-bar = mean of per-country ADF stats)")
out("="*70)
out("IMPLEMENTATION NOTE (report this honestly): this is the DESCRIPTIVE t-bar")
out("form of IPS — the simple average of per-country ADF(1) t-statistics. It is")
out("NOT the full standardized IPS W-statistic with exact p-values. It is")
out("directionally informative and adequate for a robustness section, but for")
out("formal IPS p-values use Stata (xtunitroot ips) or R (plm::purtest). Do not")
out("present the numbers below as exact IPS test p-values.")
out("More negative t-bar => stronger rejection of the unit-root null (=> stationary).")
out("Approx IPS 5% critical t-bar for these dims is around -1.7 to -2.3 (indicative only).\n")
for var in ["tfr"] + CONTROLS:
    tbar, k = ips_tbar(p_idx, var)
    flag = ""
    if not np.isnan(tbar):
        flag = "  <- likely stationary" if tbar < -2.0 else "  <- unit root not rejected"
    out(f"  {var:28s}: t-bar = {tbar:+.3f}  (from {k} countries){flag}")

# Also test FIRST DIFFERENCES of TFR (should be clearly stationary)
p_idx_d = p_idx.copy()
p_idx_d["d_tfr"] = p_idx.groupby(level=0)["tfr"].diff()
tbar_d, k_d = ips_tbar(p_idx_d, "d_tfr")
out(f"  {'d_tfr (first difference)':28s}: t-bar = {tbar_d:+.3f}  (from {k_d} countries)"
    f"{'  <- stationary' if tbar_d < -2.0 else ''}")

# ----------------------------------------------------------------------------
# (C2) Wooldridge AR(1) test for serial correlation in panel residuals
#      Regress FE residuals on their own lag within country; H0: no AR(1).
# ----------------------------------------------------------------------------
import statsmodels.formula.api as smf
res_df = d.copy()
res_df["resid"] = fe_res.resids
res_df = res_df.reset_index()
res_df = res_df.sort_values(["country", "year"])
res_df["resid_lag"] = res_df.groupby("country")["resid"].shift(1)
ww = res_df.dropna(subset=["resid", "resid_lag"])
ar1 = smf.ols("resid ~ resid_lag", data=ww).fit(
    cov_type="cluster", cov_kwds={"groups": ww["country"]})
out("\n" + "="*70)
out("(C2) Wooldridge-style AR(1) serial-correlation check on FE residuals")
out("="*70)
out(f"  AR(1) coefficient on lagged residual: {ar1.params['resid_lag']:+.3f} "
    f"(p={ar1.pvalues['resid_lag']:.3f})")
out("  Significant positive coefficient => residual serial correlation present;")
out("  this is why clustered SEs (which we use) are the appropriate correction.")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/layerB_results.txt", "w") as f:
    f.write("LAYER B RESULTS — within-country (two-way FE), first-difference, "
            "and stationarity/serial-correlation diagnostics.\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/layerB_results.txt")