
# Layer A (gap): Models 1-3, pooled OLS + year FE, clustered standard errors.
"""
15_layerA_gap.py
----------------
LAYER A — the between-country question: how large is the Central Asia fertility
premium, and does it survive economic controls?

Estimator: pooled OLS with YEAR fixed effects (year dummies), standard errors
CLUSTERED by country. The Central Asia dummy (ca) identifies the cross-country
gap, so country fixed effects are deliberately NOT used here (they would absorb
the dummy — that is Layer B's job, script 16).

Models:
  M1  TFR ~ CA + year effects
        -> raw size of the premium.
  M2  TFR ~ CA + year effects + lagged economic controls
        -> does the premium survive? The surviving CA coefficient is the part of
           the gap NOT explained by economics — the opening for the cultural story.
  M3  TFR ~ CA + year effects + controls + (CA x control), one interaction at a time
        -> does the slope on each economic factor differ inside Central Asia?

Controls (all lagged 1 year, from 12_build_panel):
  log_gdp_ppp_lag1, urban_pop_pct_lag1, remittances_gdp_pct_lag1, under5_mortality_lag1

NOTE ON STANDARD ERRORS: clustering is on 14 countries. With so few clusters,
cluster-robust SEs are known to be anti-conservative (too small). Treat p-values
near 0.05 with caution; 17_diagnostics revisits this and a wild-cluster bootstrap
is the recommended robustness follow-up.

Output: data/processed/layerA_results.txt
Run from repo root:  python analysis/15_layerA_gap.py
"""

import pandas as pd
import statsmodels.formula.api as smf

p = pd.read_csv("data/processed/panel.csv")

CONTROLS = [
    "log_gdp_ppp_lag1",
    "urban_pop_pct_lag1",
    "remittances_gdp_pct_lag1",
    "under5_mortality_lag1",
]
CLUSTER = {"groups": None}  # filled per-fit below (cluster on the rows actually used)

def fit(formula, data):
    """OLS with year FE already in the formula; cluster-robust SE by country,
    aligned to the estimation sample (drops NaN rows first so clusters match)."""
    used_vars = [v.strip() for v in formula.replace("~", "+").split("+")]
    # keep only columns that are real variables present in data
    cols = ["tfr", "country", "year"] + [c for c in data.columns
            if c in used_vars or c in CONTROLS or c == "ca"]
    d = data.dropna(subset=[c for c in cols if c in data.columns and c not in ("country",)]).copy()
    model = smf.ols(formula, data=d)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["country"]})
    return res, int(res.nobs)

lines = []
def report(title, res, n):
    block = [f"\n===== {title}  (N={n}) =====",
             str(res.summary().tables[1]),
             f"R-squared: {res.rsquared:.3f}   Adj: {res.rsquared_adj:.3f}"]
    text = "\n".join(block)
    print(text)
    lines.append(text)

# ----- Model 1a: raw premium, ALL available TFR observations (year FE) -----
# This is the true unconditional Central Asia gap, using every country-year in the
# panel where TFR is observed — not restricted to the controls-complete sample.
m1a, n1a = fit("tfr ~ ca + C(year)", p)  # fit() already drops NaN in used vars only
# But fit() also drops on CONTROLS-list even if not in the formula, so recompute directly:
d1a = p.dropna(subset=["tfr", "ca", "year"]).copy()
import statsmodels.formula.api as _smf
m1a = _smf.ols("tfr ~ ca + C(year)", data=d1a).fit(
    cov_type="cluster", cov_kwds={"groups": d1a["country"]})
n1a = int(m1a.nobs)
ca_a = m1a.params["ca"]; se_a = m1a.bse["ca"]; p_a = m1a.pvalues["ca"]
msg_a = (f"\n===== M1a: raw Central Asia premium — FULL TFR sample (year FE) (N={n1a}) =====\n"
         f"  ca coefficient: {ca_a:+.3f}  (cluster SE {se_a:.3f}, p={p_a:.3f})\n"
         f"  R-squared: {m1a.rsquared:.3f}\n"
         f"  Interpretation: unconditional cross-country gap using every observed TFR row.")
print(msg_a); lines.append(msg_a)

# ----- Model 1b: raw premium, ESTIMATION sample (matches M2/M3) -----
# Restricted to rows where all Layer-A controls are present, so N is directly
# comparable to M2 and M3.
m1, n1 = fit("tfr ~ ca + C(year)", p)
# print only the CA row + a note (year dummies clutter the table)
ca_coef = m1.params["ca"]; ca_se = m1.bse["ca"]; ca_p = m1.pvalues["ca"]
msg = (f"\n===== M1b: raw Central Asia premium — estimation sample (year FE) (N={n1}) =====\n"
       f"  ca coefficient: {ca_coef:+.3f}  (cluster SE {ca_se:.3f}, p={ca_p:.3f})\n"
       f"  R-squared: {m1.rsquared:.3f}\n"
       f"  Interpretation: same specification as M1a, restricted to the rows M2/M3\n"
       f"  use (controls-complete sample); the small drop from N={n1a} to N={n1} reflects\n"
       f"  lagged-covariate coverage, not a change in the underlying gap.")
print(msg); lines.append(msg)

# ----- Model 2: controlled premium -----
f2 = "tfr ~ ca + " + " + ".join(CONTROLS) + " + C(year)"
m2, n2 = fit(f2, p)
ca2 = m2.params["ca"]; se2 = m2.bse["ca"]; p2 = m2.pvalues["ca"]
report("M2: controlled premium (CA + lagged economics + year FE)", m2, n2)
note = (f"  --> Surviving CA premium after economic controls: {ca2:+.3f} "
        f"(SE {se2:.3f}, p={p2:.3f})\n"
        f"  Interpretation: a large Central Asia premium remains after controlling for\n"
        f"  the selected macro-level economic and development indicators. This residual\n"
        f"  cross-country gap motivates the cross-section analysis of cultural factors\n"
        f"  (script 18); it should not, by itself, be labelled 'cultural'.")
print(note); lines.append(note)

# ----- Model 3: interactions, ONE AT A TIME -----
INTERACTIONS = {
    "urban_pop_pct_lag1": "urbanisation",
    "remittances_gdp_pct_lag1": "remittances",
    "log_gdp_ppp_lag1": "log GDP-PPP",
}
for var, label in INTERACTIONS.items():
    f3 = ("tfr ~ ca + " + " + ".join(CONTROLS) + f" + ca:{var}" + " + C(year)")
    m3, n3 = fit(f3, p)
    inter = f"ca:{var}"
    coef = m3.params.get(inter); se = m3.bse.get(inter); pv = m3.pvalues.get(inter)
    block = (f"\n===== M3: interaction CA x {label} (year FE) (N={n3}) =====\n"
             f"  {inter}: {coef:+.4f}  (SE {se:.4f}, p={pv:.3f})\n"
             f"  ca main: {m3.params['ca']:+.3f}   R-squared: {m3.rsquared:.3f}\n"
             f"  Reading: the effect of {label} on TFR differs in Central Asia "
             f"by {coef:+.4f} per unit relative to the rest.")
    print(block); lines.append(block)

# ----- Save -----
import os
os.makedirs("data/processed", exist_ok=True)
header = ("LAYER A RESULTS — Central Asia fertility premium\n"
          "Pooled OLS, year fixed effects, SE clustered by country (14 clusters).\n"
          "Controls lagged 1 year. Caution: few-cluster SEs may be anti-conservative.\n")
with open("data/processed/layerA_results.txt", "w") as f:
    f.write(header + "\n".join(lines))
print("\nSaved -> data/processed/layerA_results.txt")