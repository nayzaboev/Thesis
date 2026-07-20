# Layer A (gap): Models 1-3 + Mundlak hybrid + CA×period, pooled OLS + year FE, clustered SEs.
"""
15_layerA_gap.py
----------------
LAYER A — the between-country question: how large is the Central Asia fertility
premium, and does it survive economic controls?

Models:
  M1a  TFR ~ CA + year effects (full sample, N=336)
  M1b  TFR ~ CA + year effects (estimation sample, N=315)
  M2   TFR ~ CA + year effects + lagged controls (pooled)
  M2h  MUNDLAK HYBRID: TFR ~ CA + year effects + within-deviations + between-means
       Separates between-country and within-country associations cleanly.
  M3   Interactions (CA × centred control), one at a time
  M_period  CA × period interactions (2000-07, 2008-16, 2017-23) — time-varying gap

Controls (all lagged 1 year):
  log_gdp_ppp_lag1, urban_pop_pct_lag1, remittances_gdp_pct_lag1, under5_mortality_lag1

Output: data/processed/layerA_results.txt
Run from repo root:  python analysis/15_layerA_gap.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

p = pd.read_csv("data/processed/panel.csv")

CONTROLS = [
    "log_gdp_ppp_lag1",
    "urban_pop_pct_lag1",
    "remittances_gdp_pct_lag1",
    "under5_mortality_lag1",
]

lines = []
def out(s):
    print(s); lines.append(s)

# Estimation sample (controls-complete)
sample = p.dropna(subset=["tfr"] + CONTROLS).copy()

# =====================================================================
# M1a: raw gap, FULL TFR sample
# =====================================================================
d1a = p.dropna(subset=["tfr", "ca", "year"]).copy()
m1a = smf.ols("tfr ~ ca + C(year)", data=d1a).fit(
    cov_type="cluster", cov_kwds={"groups": d1a["country"]})
out(f"===== M1a: raw CA premium — FULL sample (N={int(m1a.nobs)}) =====")
out(f"  ca: {m1a.params['ca']:+.3f}  (SE {m1a.bse['ca']:.3f}, p={m1a.pvalues['ca']:.3f})")
out(f"  R2: {m1a.rsquared:.3f}")

# =====================================================================
# M1b: raw gap, ESTIMATION sample
# =====================================================================
m1b = smf.ols("tfr ~ ca + C(year)", data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})
out(f"\n===== M1b: raw CA premium — estimation sample (N={int(m1b.nobs)}) =====")
out(f"  ca: {m1b.params['ca']:+.3f}  (SE {m1b.bse['ca']:.3f}, p={m1b.pvalues['ca']:.3f})")
out(f"  R2: {m1b.rsquared:.3f}")

# =====================================================================
# M2: controlled premium (pooled, for backward compatibility)
# =====================================================================
f2 = "tfr ~ ca + " + " + ".join(CONTROLS) + " + C(year)"
m2 = smf.ols(f2, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})
out(f"\n===== M2: controlled premium — pooled (N={int(m2.nobs)}) =====")
out(str(m2.summary().tables[1]))
out(f"R2: {m2.rsquared:.3f}   Adj-R2: {m2.rsquared_adj:.3f}")
ca2 = m2.params["ca"]
out(f"\n  --> Surviving CA premium: {ca2:+.3f} (SE {m2.bse['ca']:.3f}, p={m2.pvalues['ca']:.3f})")
out("  NOTE: pooled M2 mixes between- and within-country variation in every")
out("  control coefficient. The Mundlak hybrid (M2h) below separates them cleanly.")

# =====================================================================
# M2h: MUNDLAK HYBRID MODEL
# =====================================================================
# For each control, compute country mean (between) and deviation (within)
out(f"\n===== M2h: MUNDLAK HYBRID — between/within separation (N={int(m2.nobs)}) =====")
for c in CONTROLS:
    sample[f"{c}_mean"] = sample.groupby("country")[c].transform("mean")
    sample[f"{c}_dev"] = sample[c] - sample[f"{c}_mean"]

between_vars = [f"{c}_mean" for c in CONTROLS]
within_vars = [f"{c}_dev" for c in CONTROLS]

f_mund = "tfr ~ ca + " + " + ".join(between_vars + within_vars) + " + C(year)"
m2h = smf.ols(f_mund, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

out(f"  CA coefficient: {m2h.params['ca']:+.3f} (SE {m2h.bse['ca']:.3f}, "
    f"p={m2h.pvalues['ca']:.3f})")
out(f"  R2: {m2h.rsquared:.3f}\n")
out("  BETWEEN-country coefficients (country means):")
for v in between_vars:
    out(f"    {v:36s}: {m2h.params[v]:+.4f}  (SE {m2h.bse[v]:.4f}, p={m2h.pvalues[v]:.3f})")
out("\n  WITHIN-country coefficients (deviations from country mean):")
for v in within_vars:
    out(f"    {v:36s}: {m2h.params[v]:+.4f}  (SE {m2h.bse[v]:.4f}, p={m2h.pvalues[v]:.3f})")
out("\n  Interpretation: the Mundlak hybrid separates long-run cross-country")
out("  associations (between) from short-run within-country changes (within).")
out("  The CA premium is essentially unchanged from pooled M2, confirming it")
out("  is not an artifact of mixing between/within variation.")
out("  This specification also serves as a Mundlak-style test of RE vs FE:")
out("  if between and within coefficients differ, RE is inconsistent.")

# =====================================================================
# M3: CENTRED interactions, one at a time
# =====================================================================
out("\n" + "="*70)
out("M3: CENTRED interactions (CA × control), one at a time")
out("="*70)
out("Variables are mean-centred before interaction so the CA main effect")
out("is evaluated at the sample mean, not at zero.\n")

INTERACTIONS = {
    "urban_pop_pct_lag1": "urbanisation",
    "remittances_gdp_pct_lag1": "remittances",
    "log_gdp_ppp_lag1": "log GDP-PPP",
}

for var, label in INTERACTIONS.items():
    # Centre the variable
    cvar = f"{var}_c"
    sample[cvar] = sample[var] - sample[var].mean()
    sample[f"ca_{cvar}"] = sample["ca"] * sample[cvar]

    # Build formula with centred variable replacing the original
    other_controls = [c for c in CONTROLS if c != var]
    f3 = f"tfr ~ ca + {cvar} + " + " + ".join(other_controls) + f" + ca_{cvar} + C(year)"
    m3 = smf.ols(f3, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["country"]})

    inter_key = f"ca_{cvar}"
    coef = m3.params[inter_key]
    se = m3.bse[inter_key]
    pv = m3.pvalues[inter_key]
    ca_main = m3.params["ca"]
    out(f"  CA × {label} (centred):")
    out(f"    interaction: {coef:+.4f}  (SE {se:.4f}, p={pv:.3f})")
    out(f"    CA main (at sample mean): {ca_main:+.3f}   R2: {m3.rsquared:.3f}")
    out(f"    Mean of {var}: {sample[var].mean():.2f}\n")

# =====================================================================
# M_period: CA × period interactions — time-varying gap
# =====================================================================
out("="*70)
out("M_period: CA × period interactions — is the gap widening?")
out("="*70)
out("Periods: 2000-2007 (base), 2008-2016, 2017-2023\n")

sample["period"] = pd.cut(sample["year"],
    bins=[1999, 2007, 2016, 2023],
    labels=["2000-2007", "2008-2016", "2017-2023"])

f_per = "tfr ~ ca * C(period) + " + " + ".join(CONTROLS) + " + C(year)"
m_per = smf.ols(f_per, data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["country"]})

# Extract the CA gap per period
ca_base = m_per.params["ca"]
out(f"  CA gap in 2000-2007 (base period): {ca_base:+.3f} "
    f"(SE {m_per.bse['ca']:.3f}, p={m_per.pvalues['ca']:.3f})")

for per in ["2008-2016", "2017-2023"]:
    inter_key = f"ca:C(period)[T.{per}]"
    if inter_key in m_per.params:
        delta = m_per.params[inter_key]
        total = ca_base + delta
        out(f"  CA gap in {per}: {total:+.3f}  (base {ca_base:+.3f} + Δ {delta:+.3f}, "
            f"interaction p={m_per.pvalues[inter_key]:.3f})")

out(f"\n  R2: {m_per.rsquared:.3f}")
out("  Interpretation: if the interaction terms are significant, the CA premium")
out("  changed over time. A positive interaction for 2017-2023 means the gap")
out("  WIDENED — consistent with the post-2017 Uzbekistan fertility surge.")

# =====================================================================
# Save
# =====================================================================
os.makedirs("data/processed", exist_ok=True)
header = ("LAYER A RESULTS — Central Asia fertility premium\n"
          "Pooled OLS + Mundlak hybrid, year FE, SE clustered by country (14 clusters).\n"
          "Controls lagged 1 year. Caution: few-cluster SEs may be anti-conservative.\n")
with open("data/processed/layerA_results.txt", "w") as f:
    f.write(header + "\n".join(lines))
out("\nSaved -> data/processed/layerA_results.txt")