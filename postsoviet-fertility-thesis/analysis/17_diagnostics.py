"""Part 2 scaffold."""
# VIF, Hausman test, serial-correlation tests.
"""
17_diagnostics.py
-----------------
Formal model diagnostics that justify the Part 2 specification choices.

Produces:
  (A) VIF (variance inflation factors) on the lagged covariates — multicollinearity.
      The OLD model had condition number ~1,310 and a urban/remittances VIF problem;
      this checks whether the new variable set (PPP-GDP, under-5 mortality, no FLFP,
      no tertiary enrolment) is cleaner.

  (B) Hausman test, fixed effects vs random effects — the formal justification for
      using FE (Layer B). H0: RE is consistent/efficient. Rejection => prefer FE.

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
# (B) Hausman test (FE vs RE)
# ----------------------------------------------------------------------------
out("\n" + "="*64)
out("(B) Hausman test — fixed effects vs random effects")
out("="*64)
di = d.set_index(["country", "year"])
exog = add_constant(di[CONTROLS])

fe = PanelOLS(di["tfr"], exog, entity_effects=True).fit()
re = RandomEffects(di["tfr"], exog).fit()

# Hausman statistic: (b_fe - b_re)' [Var(b_fe) - Var(b_re)]^-1 (b_fe - b_re)
common = [c for c in CONTROLS if c in fe.params.index and c in re.params.index]
b_fe = fe.params[common].values
b_re = re.params[common].values
v_fe = fe.cov.loc[common, common].values
v_re = re.cov.loc[common, common].values
diff = b_fe - b_re
vdiff = v_fe - v_re
try:
    stat = float(diff.T @ np.linalg.pinv(vdiff) @ diff)
    from scipy import stats as ss
    dfree = len(common)
    pval = 1 - ss.chi2.cdf(stat, dfree)
    out(f"  Hausman chi2({dfree}) = {stat:.2f}   p = {pval:.4f}")
    if pval < 0.05:
        out("  => Reject H0: random effects is inconsistent. FIXED EFFECTS preferred.")
    else:
        out("  => Fail to reject H0: random effects not rejected (RE may be acceptable).")
    out("  (We use FE on substantive grounds regardless: strong country heterogeneity,")
    out("   and the research question concerns between-country differences handled in Layer A.)")
except Exception as e:
    out(f"  Hausman computation issue: {e}")

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
    f.write("DIAGNOSTICS — VIF, Hausman (FE vs RE), few-cluster caveat.\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/diagnostics_results.txt")