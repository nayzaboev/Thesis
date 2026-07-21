"""
16_layerB_within.py
-------------------
LAYER B — the within-country question: over time, inside a given country, what
moves fertility? This is the robustness defence against the spurious-regression
critique (the old pooled model had Durbin-Watson = 0.13).

Four estimators / descriptive checks:

  (A) TWO-WAY FIXED EFFECTS (country + year), via linearmodels.PanelOLS,
      clustered SE by country.
      NOTE: the Central Asia dummy and any time-invariant variable are ABSORBED
      by country effects and cannot be estimated here — that is expected, not a
      bug. Layer A (script 15) is where the CA premium lives.

  (B1) FIRST-DIFFERENCE model WITHOUT year effects: regress delta-TFR on
       delta-covariates. Reported as a sensitivity check.

  (B2) FIRST-DIFFERENCE model WITH year effects: same as B1 but includes
       year dummies after differencing. This absorbs common time shocks. If a
       coefficient is significant in B1 but not in B2, the estimate is SENSITIVE
       to the inclusion of common time controls (year effects). This is a
       statement about specification sensitivity, not proof that the B1 result
       was "caused by" common shocks.

  (C) DESCRIPTIVE PERSISTENCE CHECKS (not formal tests):
      - Average per-country ADF t-statistic ("t-bar") on TFR and each covariate.
        This is a descriptive summary of persistence, NOT the formal
        Im-Pesaran-Shin (IPS) panel unit-root test and does NOT establish an
        order of integration.
      - Descriptive AR(1) coefficient on the FE residuals. This summarises
        residual persistence; it is NOT the Wooldridge-Drukker panel
        serial-correlation test and does not by itself imply a unit root or a
        spurious regression.

Controls (lagged 1 year): log_gdp_ppp_lag1, urban_pop_pct_lag1,
remittances_gdp_pct_lag1, under5_mortality_lag1

Output: data/processed/layerB_results.txt
Run from repo root:  python analysis/16_layerB_within.py
"""

import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, FirstDifferenceOLS
import statsmodels.formula.api as smf

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
# (B1) First-difference model WITHOUT year effects
# ----------------------------------------------------------------------------
try:
    fd = FirstDifferenceOLS(d["tfr"], d[CONTROLS])
    fd_res = fd.fit(cov_type="clustered", cluster_entity=True)
    out("\n" + "="*70)
    out("(B1) FIRST-DIFFERENCE model — NO year effects")
    out("="*70)
    out(f"N = {int(fd_res.nobs)}   R2 = {fd_res.rsquared:.3f}\n")
    for v in CONTROLS:
        if v in fd_res.params.index:
            out(f"  {v:28s}: {fd_res.params[v]:+.4f}  (SE {fd_res.std_errors[v]:.4f}, "
                f"p={fd_res.pvalues[v]:.3f})")
except Exception as e:
    out(f"\n(B1) First-difference model could not be estimated: {e}")

# ----------------------------------------------------------------------------
# (B2) First-difference model WITH year effects
#      linearmodels' FirstDifferenceOLS does not support time effects natively,
#      so we difference manually and run OLS with year dummies.
# ----------------------------------------------------------------------------
out("\n" + "="*70)
out("(B2) FIRST-DIFFERENCE model — WITH year effects")
out("="*70)

fd_df = d.reset_index().sort_values(["country", "year"]).copy()
# Difference TFR and controls within each country
for col in ["tfr"] + CONTROLS:
    fd_df[f"d_{col}"] = fd_df.groupby("country")[col].diff()
fd_df = fd_df.dropna(subset=[f"d_{c}" for c in ["tfr"] + CONTROLS])

d_controls = [f"d_{c}" for c in CONTROLS]
formula = f"d_tfr ~ {' + '.join(d_controls)} + C(year)"
fd_yr = smf.ols(formula, data=fd_df).fit(
    cov_type="cluster", cov_kwds={"groups": fd_df["country"]})
out(f"N = {int(fd_yr.nobs)}   R2 = {fd_yr.rsquared:.3f}\n")
for v in d_controls:
    out(f"  {v:28s}: {fd_yr.params[v]:+.4f}  (SE {fd_yr.bse[v]:.4f}, "
        f"p={fd_yr.pvalues[v]:.3f})")

# --- Compare B1 and B2 ---
out("\n  SENSITIVITY CHECK: compare B1 (no year FE) vs B2 (with year FE).")
out("  If a coefficient is significant in B1 but not in B2, the estimate is")
out("  SENSITIVE to the inclusion of common year effects. This is a statement")
out("  about specification sensitivity — it does not by itself prove the B1")
out("  result was caused by common time shocks (loss of within-country")
out("  variation, multicollinearity, measurement error and over-control are")
out("  alternative explanations). Coefficients stable across B1 and B2 are more")
out("  credible.")

# Flag any coefficient that flips significance
for v_raw, v_d in zip(CONTROLS, d_controls):
    try:
        p_b1 = fd_res.pvalues[v_raw]
        p_b2 = fd_yr.pvalues[v_d]
        if p_b1 < 0.05 and p_b2 >= 0.10:
            out(f"  ** {v_raw}: significant in B1 (p={p_b1:.3f}) but NOT in B2 "
                f"(p={p_b2:.3f}). This estimate is SENSITIVE to year effects.")
        elif p_b1 >= 0.10 and p_b2 < 0.05:
            out(f"  ** {v_d}: NOT significant in B1 but significant in B2 (p={p_b2:.3f}).")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# (C1) Descriptive persistence summary — average per-country ADF t-statistic
#      This is a DESCRIPTIVE summary of persistence. It is NOT a formal
#      Im-Pesaran-Shin (IPS) panel unit-root test, produces no valid p-values,
#      and is NOT used to assign an order of integration to any series.
# ----------------------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller

def adf_tbar(panel_df, var):
    """Average of per-country ADF t-statistics. Descriptive persistence summary
    only — NOT a formal IPS test and NOT an integration-order test."""
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
out("(C1) Descriptive persistence summary — average per-country ADF t-statistic")
out("="*70)
out("IMPORTANT: this is a DESCRIPTIVE persistence summary — the simple average of")
out("per-country ADF(1) t-statistics. It is NOT the formal Im-Pesaran-Shin (IPS)")
out("panel unit-root test, it produces no valid p-values, and it does NOT")
out("establish that any series is I(1) or I(0). For a formal panel unit-root")
out("test use Stata (xtunitroot ips / xtunitroot fisher) or R (plm::purtest).")
out("A more negative average t-statistic indicates weaker persistence (values")
out("further from a unit root); this is reported only as descriptive context.\n")
for var in ["tfr"] + CONTROLS:
    tbar, k = adf_tbar(p_idx, var)
    # Report the average ADF t-statistic without a threshold-based label. A
    # "high/low persistence" cutoff at t=-2 has no formal statistical basis here
    # (this is not a formal panel unit-root test); the raw number is left to
    # speak for itself and a more negative value indicates weaker persistence.
    out(f"  {var:28s}: avg ADF t = {tbar:+.3f}  (from {k} countries)")

# Also summarise FIRST DIFFERENCES of TFR
p_idx_d = p_idx.copy()
p_idx_d["d_tfr"] = p_idx.groupby(level=0)["tfr"].diff()
tbar_d, k_d = adf_tbar(p_idx_d, "d_tfr")
out(f"  {'d_tfr (first difference)':28s}: avg ADF t = {tbar_d:+.3f}  (from {k_d} countries)")

# ----------------------------------------------------------------------------
# (C2) Descriptive residual persistence check — AR(1) on FE residuals
#      Regress FE residuals on their own lag within country. This summarises
#      how persistent the residuals are. It is NOT the Wooldridge-Drukker panel
#      serial-correlation test, and a high AR(1) coefficient does NOT by itself
#      imply a unit root, an I(1) TFR series, or a spurious regression.
# ----------------------------------------------------------------------------
res_df = d.copy()
res_df["resid"] = fe_res.resids
res_df = res_df.reset_index()
res_df = res_df.sort_values(["country", "year"])
res_df["resid_lag"] = res_df.groupby("country")["resid"].shift(1)
ww = res_df.dropna(subset=["resid", "resid_lag"])
ar1 = smf.ols("resid ~ resid_lag", data=ww).fit(
    cov_type="cluster", cov_kwds={"groups": ww["country"]})
out("\n" + "="*70)
out("(C2) Descriptive residual persistence check — AR(1) on FE residuals")
out("="*70)
out("IMPORTANT: this is a DESCRIPTIVE AR(1) check on the fixed-effects residuals.")
out("It is NOT the Wooldridge-Drukker panel serial-correlation test. A high AR(1)")
out("coefficient indicates the FE-in-levels residuals are persistent, but it does")
out("NOT establish that the residuals contain a unit root, that TFR is I(1), or")
out("that the levels regression is spurious.\n")
out(f"  AR(1) coefficient on lagged residual: {ar1.params['resid_lag']:+.3f} "
    f"(p={ar1.pvalues['resid_lag']:.3f})")
out("  The FE-in-levels residuals are strongly persistent. Because of this")
out("  persistence, the first-difference models (B1/B2) are the more cautious")
out("  within-country specification and the levels FE model (A) should be read")
out("  as descriptive rather than as the primary within-country estimate.")
out("")
out("  INTERPRETATION OF LAYER B:")
out("  Within the available annual panel, the associations between TFR and the")
out("  four selected macroeconomic indicators are generally imprecise and")
out("  sensitive to the treatment of common year effects (see B1 vs B2). Layer B")
out("  is therefore best read as a robustness exercise showing that these")
out("  particular within-country coefficients are generally imprecise and")
out("  specification-sensitive in this sample (several point estimates are not")
out("  necessarily economically small). This does NOT establish that economic")
out("  factors in general have")
out("  small effects, and it does not rule out other economic, policy or")
out("  demographic mechanisms. Consistent with Layer A, most of the Central")
out("  Asia-rest fertility difference is between-country rather than year-to-year")
out("  within-country variation, but the present design cannot identify the")
out("  mechanisms behind that between-country difference.")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/layerB_results.txt", "w") as f:
    f.write("LAYER B RESULTS — within-country (two-way FE), first-difference "
            "(with and without year effects), and descriptive persistence "
            "checks (NOT formal panel unit-root or serial-correlation tests).\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/layerB_results.txt")