
# VIF, Mundlak FE-vs-RE test, few-cluster caveat.
"""
17_diagnostics.py
-----------------
Formal model diagnostics that justify the Part 2 specification choices.

Produces:
  (A) VIF (variance inflation factors) on the lagged covariates — multicollinearity.
      The OLD model had condition number ~1,310 and a urban/remittances VIF problem;
      this checks whether the new variable set (PPP-GDP, under-5 mortality, no FLFP,
      no tertiary enrolment) is cleaner.

  (B) Mundlak test for FE vs RE — the formal justification for the between/within
      separation. Replaces the classical Hausman test, which is undefined here
      because Var(b_FE) - Var(b_RE) is not positive definite (see note in code).

  (C) Few-cluster caveat — consolidated note; 14 clusters is below the safe
      threshold, so cluster-robust SEs may be anti-conservative; wild-cluster
      bootstrap (e.g. Stata boottest, R fwildclusterboot) is the recommended
      robustness follow-up.

Output: data/processed/diagnostics_results.txt
Run from repo root:  python analysis/17_diagnostics.py
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from linearmodels.panel import PanelOLS, RandomEffects

CONTROLS = ["log_gdp_ppp_lag1", "urban_pop_pct_lag1",
            "remittances_gdp_pct_lag1", "under5_mortality_lag1"]

p = pd.read_csv("data/processed/panel.csv")
d = p.dropna(subset=["tfr"] + CONTROLS).copy()

lines = []
def out(s):
    print(s); lines.append(s)

# ----------------------------------------------------------------------------
# (A) VIF
# ----------------------------------------------------------------------------
out("="*64)
out("(A) Variance Inflation Factors (lagged covariates)")
out("="*64)
out("Rule of thumb: VIF > 5 concerning, > 10 serious.\n")
X = add_constant(d[CONTROLS])
for i, col in enumerate(X.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X.values, i)
    flag = "  <- OK" if vif < 5 else ("  <- CONCERNING" if vif < 10 else "  <- SERIOUS")
    out(f"  {col:28s}: {vif:5.2f}{flag}")

# ----------------------------------------------------------------------------
# (B) Mundlak test for FE vs RE  (replaces the classical Hausman test)
# ----------------------------------------------------------------------------
# WHY NOT THE CLASSICAL HAUSMAN TEST:
#   The classical statistic requires Var(b_FE) - Var(b_RE) to be positive
#   definite. In this sample it is NOT: three of its four eigenvalues are
#   negative. Computing the statistic with a pseudo-inverse (as an earlier
#   version of this script did) returns a number, but that number is not a
#   valid chi-square statistic and must not be reported.
#
#   The Mundlak (1978) auxiliary-regression test is the robust alternative and
#   is valid with clustered standard errors. Add the country means of the
#   regressors to a pooled model; H0: all mean coefficients = 0 (RE consistent).
out("\n" + "="*64)
out("(B) Mundlak test — fixed effects vs random effects")
out("="*64)
out("Classical Hausman is NOT reported: Var(b_FE) - Var(b_RE) is not positive")
out("definite in this sample (3 of 4 eigenvalues negative), so the statistic")
out("is undefined. The Mundlak auxiliary-regression test is used instead.\n")

import statsmodels.formula.api as smf
m = d.copy()
for c in CONTROLS:
    m[f"{c}_mean"] = m.groupby("country")[c].transform("mean")
    m[f"{c}_dev"]  = m[c] - m[f"{c}_mean"]
between = [f"{c}_mean" for c in CONTROLS]
within  = [f"{c}_dev"  for c in CONTROLS]
f_m = "tfr ~ ca + " + " + ".join(between + within) + " + C(year)"
mund = smf.ols(f_m, data=m).fit(cov_type="cluster", cov_kwds={"groups": m["country"]})
ftest = mund.f_test(" , ".join([f"{v} = 0" for v in between]))
out(f"  H0: all between-country mean coefficients = 0 (random effects consistent)")
out(f"  F({int(ftest.df_num)}, {int(ftest.df_denom)}) = {float(ftest.fvalue):.3f}   "
    f"p = {float(ftest.pvalue):.4f}")
if float(ftest.pvalue) < 0.05:
    out("  => Reject H0: unobserved country effects are correlated with the")
    out("     regressors. Random effects is inconsistent; the between/within")
    out("     separation used in Layer A (M2h) and Layer B is required.")
else:
    out("  => Fail to reject H0: no evidence that random effects is inconsistent.")

# ----------------------------------------------------------------------------
# (C) Few-cluster caveat
# ----------------------------------------------------------------------------
out("\n" + "="*64)
out("(C) Few-cluster caveat")
out("="*64)
out("  Standard errors throughout Part 2 are clustered on country (14 clusters).")
out("  14 clusters is below the commonly cited safe threshold (~30-50). With few")
out("  clusters, cluster-robust SEs can be anti-conservative (too small), so")
out("  p-values near 0.05 should be read with caution. Recommended robustness")
out("  follow-up: wild-cluster bootstrap (Stata 'boottest', R 'fwildclusterboot').")
out("  This is a known small-N limitation of the post-Soviet sample, not a coding issue.")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/diagnostics_results.txt", "w") as f:
    f.write("DIAGNOSTICS — VIF, Mundlak test (FE vs RE), few-cluster caveat.\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/diagnostics_results.txt")