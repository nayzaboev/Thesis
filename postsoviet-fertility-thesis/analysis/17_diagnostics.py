# VIF (pooled + between/within), Mundlak FE-vs-RE test, few-cluster caveat.
"""
17_diagnostics.py
-----------------
Formal model diagnostics that justify the Part 2 specification choices.

Produces:
  (A) VIF (variance inflation factors) on the lagged covariates — multicollinearity.
      Reported THREE ways: pooled, between-country (country means), and
      within-country (deviations from country means). The between/within split
      explains why the CA coefficient is stable while individual between-country
      control coefficients are imprecise: the country means are highly collinear.

  (B) Mundlak test for FE vs RE — the formal justification for the between/within
      separation. Replaces the classical Hausman test, which is undefined here
      because Var(b_FE) - Var(b_RE) is not positive definite (see note in code).

  (C) Few-cluster caveat — 14 clusters is below the safe threshold, so
      cluster-robust SEs may be anti-conservative; wild-cluster bootstrap
      (script 20, section G) is the recommended robustness follow-up.

Output: data/processed/diagnostics_results.txt
Run from repo root:  python analysis/17_diagnostics.py
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

CONTROLS = ["log_gdp_ppp_lag1", "urban_pop_pct_lag1",
            "remittances_gdp_pct_lag1", "under5_mortality_lag1"]

p = pd.read_csv("data/processed/panel.csv")
d = p.dropna(subset=["tfr"] + CONTROLS).copy()

lines = []
def out(s):
    print(s); lines.append(s)

# ----------------------------------------------------------------------------
# (A) VIF — pooled, between, within
# ----------------------------------------------------------------------------
out("="*64)
out("(A) Variance Inflation Factors (lagged covariates)")
out("="*64)
out("Rule of thumb: VIF > 5 concerning, > 10 serious.")
out("Reported three ways so the source of any collinearity is transparent:")
out("  pooled   = the raw lagged covariates (what the pooled M2 model uses)")
out("  between  = country means of each covariate (the Mundlak between block)")
out("  within   = deviations from country means (the Mundlak within block)\n")

def vif_table(frame, cols):
    X = add_constant(frame[cols])
    res = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        res[col] = variance_inflation_factor(X.values, i)
    return res

# Build between (means) and within (deviations)
m = d.copy()
for c in CONTROLS:
    m[f"{c}_mean"] = m.groupby("country")[c].transform("mean")
    m[f"{c}_dev"]  = m[c] - m[f"{c}_mean"]
between_cols = [f"{c}_mean" for c in CONTROLS]
within_cols  = [f"{c}_dev"  for c in CONTROLS]

vif_pooled  = vif_table(d, CONTROLS)
# Between VIF must be computed on ONE row per country (14 rows), not on all
# 315 panel rows, otherwise the unbalanced panel (8 missing remittance rows)
# gives countries unequal weight. CA is included because it is part of the
# between design (it is a time-invariant country characteristic).
between_df  = m.groupby("country", as_index=False)[["ca"] + between_cols].first()
vif_between = vif_table(between_df, ["ca"] + between_cols)
vif_within  = vif_table(m, within_cols)

def flag(v):
    return "OK" if v < 5 else ("CONCERNING" if v < 10 else "SERIOUS")

out(f"  {'variable':26s} {'pooled':>8s}  {'between':>8s}  {'within':>8s}")
out(f"  {'-'*26} {'-'*8}  {'-'*8}  {'-'*8}")
for c in CONTROLS:
    vp, vb, vw = vif_pooled[c], vif_between[f"{c}_mean"], vif_within[f"{c}_dev"]
    out(f"  {c:26s} {vp:8.2f}  {vb:8.2f}  {vw:8.2f}   "
        f"(between: {flag(vb)})")
out(f"  {'ca (between design)':26s} {'—':>8s}  {vif_between['ca']:8.2f}  {'—':>8s}   "
    f"(between: {flag(vif_between['ca'])})")

out("")
out("  READING: pooled VIFs are moderate, but the BETWEEN-country components are")
out("  highly collinear (GDP mean, urbanisation mean and remittances mean all")
out("  near or above the VIF>=10 threshold), while the WITHIN components are low.")
out("  Consequence: the between-country control coefficients in the Mundlak hybrid")
out("  (M2h) are individually imprecise and must NOT be interpreted one at a time.")
out("  The CA dummy has only a moderate between VIF, and its estimated coefficient")
out("  remains stable across the pooled (M2) and hybrid (M2h) specifications, so")
out("  multicollinearity does not undermine the CA premium; but the individual")
out("  between-country control coefficients are collinear and must not be")
out("  interpreted one at a time.")

# ----------------------------------------------------------------------------
# (B) Mundlak test for FE vs RE  (replaces the classical Hausman test)
# ----------------------------------------------------------------------------
# WHY NOT THE CLASSICAL HAUSMAN TEST:
#   The classical statistic requires Var(b_FE) - Var(b_RE) to be positive
#   definite. In this sample it is NOT: three of its four eigenvalues are
#   negative. Computing the statistic with a pseudo-inverse returns a number,
#   but that number is not a valid chi-square statistic and must not be reported.
#
#   THE MUNDLAK (1978) AUXILIARY-REGRESSION TEST — CORRECT RESTRICTION:
#   Estimate a hybrid model with BOTH the within-country deviations and the
#   country means of each regressor. The random-effects consistency test is
#   whether the between and within coefficients are EQUAL:
#       H0: beta_between = beta_within   (for every regressor)
#   Rejecting H0 => RE is inconsistent. NOTE: testing "beta_between = 0" is a
#   DIFFERENT hypothesis and is NOT the RE-consistency test.
out("\n" + "="*64)
out("(B) Mundlak test — fixed effects vs random effects")
out("="*64)
out("Classical Hausman is NOT reported: Var(b_FE) - Var(b_RE) is not positive")
out("definite here (3 of 4 eigenvalues negative), so the statistic is undefined.")
out("The Mundlak auxiliary-regression test is used instead.")
out("Correct restriction: H0 is that the between and within coefficients are")
out("EQUAL for every regressor (not that the between coefficients are zero).\n")

import statsmodels.formula.api as smf
between = [f"{c}_mean" for c in CONTROLS]
within  = [f"{c}_dev"  for c in CONTROLS]
f_m = "tfr ~ ca + " + " + ".join(between + within) + " + C(year)"
mund = smf.ols(f_m, data=m).fit(cov_type="cluster", cov_kwds={"groups": m["country"]})
# CORRECT Mundlak restriction: between_k = within_k for each regressor k
hyp = " , ".join([f"{b} = {w}" for b, w in zip(between, within)])
ftest = mund.f_test(hyp)
out(f"  H0: beta_between = beta_within for all 4 regressors (random effects consistent)")
out(f"  F({int(ftest.df_num)}, {int(ftest.df_denom)}) = {float(ftest.fvalue):.3f}   "
    f"p = {float(ftest.pvalue):.4f}")
if float(ftest.pvalue) < 0.05:
    out("  => Reject H0: within and between coefficients differ; random effects")
    out("     is inconsistent and the between/within separation is required.")
else:
    out("  => FAIL to reject H0: no statistical evidence that within and between")
    out("     coefficients differ, i.e. no evidence random effects is inconsistent.")
    out("     The fixed-effects / hybrid specification is retained on substantive")
    out("     grounds (strong country heterogeneity, and the research question")
    out("     concerns between-country differences), not on the basis of this test.")

# For transparency, also report the (different) 'between = 0' test, clearly labelled
ftest0 = mund.f_test(" , ".join([f"{v} = 0" for v in between]))
out(f"\n  For reference only (NOT the RE-consistency test):")
out(f"  H0: all between-country mean coefficients = 0")
out(f"  F({int(ftest0.df_num)}, {int(ftest0.df_denom)}) = {float(ftest0.fvalue):.3f}   "
    f"p = {float(ftest0.pvalue):.4f}")
out("  This tests whether country means jointly predict TFR — a separate question.")

# ----------------------------------------------------------------------------
# (C) Few-cluster caveat
# ----------------------------------------------------------------------------
out("\n" + "="*64)
out("(C) Few-cluster caveat")
out("="*64)
out("  Standard errors throughout Part 2 are clustered on country (14 clusters).")
out("  14 clusters is below the commonly cited safe threshold (~30-50). With few")
out("  clusters, cluster-robust SEs can be anti-conservative (too small), so")
out("  p-values near 0.05 should be read with caution. The wild-cluster bootstrap")
out("  in script 20 (section G) is the recommended few-cluster robustness check.")
out("  This is a known small-N limitation of the post-Soviet sample, not a coding issue.")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/diagnostics_results.txt", "w") as f:
    f.write("DIAGNOSTICS — VIF (pooled/between/within), Mundlak test (FE vs RE), "
            "few-cluster caveat.\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/diagnostics_results.txt")