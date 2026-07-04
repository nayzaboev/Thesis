"""
18_crosssection.py
------------------
Cross-country analysis (n=14): what explains the between-country fertility
premium that survives economic controls?

Two complementary approaches:

  (A) Direct: correlate mean TFR with cultural variables + stepwise OLS.
      Descriptive — shows which cultural factors track cross-country TFR.

  (B) Residuals-based (recommended): estimate TFR ~ lagged economic controls
      + year FE WITHOUT the CA dummy on the full panel, then average the
      country-level residuals. Correlate those residuals with the cultural
      variables. This isolates "what the economic model cannot explain" and
      asks whether cultural factors track THAT — which is more aligned with
      the thesis argument than regressing on mean TFR directly.

Both approaches converge in this sample, but (B) is the more defensible framing.

Caveats reported honestly:
  - n=14 severely limits statistical power; all findings are descriptive
    associations, NOT causal estimates.
  - Muslim share and the Central Asia dummy are highly correlated in this
    sample (r ~ 0.83), so cultural variables partly RE-DESCRIBE the CA/non-CA
    split rather than uniquely identifying religion's role.
  - Cross-sectional analysis cannot distinguish religion, ethnicity, family
    norms, rurality, or historical legacy — these bundle together in the data.

Outputs:
  data/processed/crosssection_results.txt
  figures/crosssection_scatter.png

Run from repo root:  python analysis/18_crosssection.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

cs = pd.read_csv("data/processed/crosssection.csv")
panel = pd.read_csv("data/processed/panel.csv")

CONTROLS = ["log_gdp_ppp_lag1", "urban_pop_pct_lag1",
            "remittances_gdp_pct_lag1", "under5_mortality_lag1"]

lines = []
def out(s):
    print(s); lines.append(s)

# =========================================================================
# Report the CA / Muslim-share correlation upfront — it disciplines everything
# =========================================================================
r_ca_muslim = cs["ca"].corr(cs["muslim_share"])
out("=" * 68)
out("Cross-section n=14 — cultural correlates of the fertility premium")
out("=" * 68)
out(f"\nUpfront caveat: corr(CA dummy, Muslim share) = {r_ca_muslim:+.3f}.")
out("Cultural variables largely re-describe the CA / non-CA split;")
out("they cannot uniquely identify religion vs ethnicity vs family norms.")

# =========================================================================
# (A) Direct correlations + stepwise OLS
# =========================================================================
out("\n" + "=" * 68)
out("(A) Direct: TFR ~ cultural variables")
out("=" * 68)

pairs = [
    ("muslim_share",          "Muslim share (%)"),
    ("smam_female",           "Female SMAM (years)"),
    ("female_mean_schooling", "Female mean schooling (years)"),
]
out("\nBivariate correlations:")
for var, label in pairs:
    r = cs["mean_tfr"].corr(cs[var])
    out(f"  TFR vs {label:36s}: r = {r:+.3f}")

out("\nStepwise OLS (mean_tfr as dependent variable):")
# NOTE: an "A4" specification adding female schooling was cut for parsimony —
# at n=14 with 3 predictors it has only 10 residual df, adds negligible R2 (0.007),
# and its only informative content (schooling p~0.58, wrong sign) is already
# conveyed by the bivariate correlation and by A3's demonstration of Muslim/SMAM
# collinearity. Schooling remains reported bivariately in (A) above.
models = {
    "A1: Muslim share only":
        "mean_tfr ~ muslim_share",
    "A2: SMAM only":
        "mean_tfr ~ smam_female",
    "A3: Muslim + SMAM":
        "mean_tfr ~ muslim_share + smam_female",
}
for name, formula in models.items():
    m = smf.ols(formula, data=cs).fit()
    out(f"\n  --- {name} ---")
    out(f"  R2 = {m.rsquared:.3f}   Adj-R2 = {m.rsquared_adj:.3f}")
    for v in m.params.index:
        if v == "Intercept":
            continue
        out(f"    {v:28s}: {m.params[v]:+.4f}  "
            f"(SE {m.bse[v]:.4f}, p={m.pvalues[v]:.3f})")

# =========================================================================
# (B) Residuals-based: strip economics + year effects, then correlate residuals
# =========================================================================
out("\n" + "=" * 68)
out("(B) Residuals-based: cultural correlates of the unexplained premium")
out("=" * 68)
out("Step 1: fit TFR ~ lagged economic controls + year FE (NO CA dummy).")
out("Step 2: country-average the residuals.")
out("Step 3: correlate country residuals with cultural variables.")
out("Interpretation: what the economic model cannot explain — does culture track it?\n")

# Step 1: pooled OLS with year FE, no CA dummy, on complete cases
est_df = panel.dropna(subset=["tfr"] + CONTROLS).copy()
formula = "tfr ~ " + " + ".join(CONTROLS) + " + C(year)"
econ_model = smf.ols(formula, data=est_df).fit()

# Step 2: average residuals per country
est_df["resid"] = econ_model.resid
country_resid = est_df.groupby("country")["resid"].mean()

# Step 3: attach to the cross-section frame
cs_r = cs.merge(country_resid.rename("mean_resid").reset_index(), on="country")

out("Country residuals (unexplained TFR after economics + year FE):")
for _, r in cs_r.sort_values("mean_resid", ascending=False).iterrows():
    out(f"  {r['country']:14s} ({r['bloc']:22s}): {r['mean_resid']:+.3f}")

out("\nCorrelations — country residuals vs cultural variables:")
for var, label in pairs:
    r_full = cs_r["mean_tfr"].corr(cs_r[var])
    r_resid = cs_r["mean_resid"].corr(cs_r[var])
    out(f"  {label:36s}: r(mean_TFR)={r_full:+.3f}  |  r(residual)={r_resid:+.3f}")

# =========================================================================
# Scatterplots — mean TFR vs the three cultural variables
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ca_mask = cs["ca"] == 1
for ax, (var, label) in zip(axes, pairs):
    ax.scatter(cs.loc[~ca_mask, var], cs.loc[~ca_mask, "mean_tfr"],
               c="steelblue", s=50, zorder=3, label="Other post-Soviet")
    ax.scatter(cs.loc[ca_mask, var], cs.loc[ca_mask, "mean_tfr"],
               c="tomato", s=70, marker="D", zorder=4, label="Central Asia")
    for _, row in cs.iterrows():
        offset = (3, 4) if row["ca"] else (3, -8)
        ax.annotate(row["country"][:3].upper(), (row[var], row["mean_tfr"]),
                    fontsize=7, textcoords="offset points", xytext=offset)
    z = np.polyfit(cs[var], cs["mean_tfr"], 1)
    xr = np.linspace(cs[var].min(), cs[var].max(), 50)
    ax.plot(xr, np.polyval(z, xr), "--", color="gray", linewidth=1, alpha=0.7)
    r = cs["mean_tfr"].corr(cs[var])
    ax.set_xlabel(label)
    ax.set_title(f"r = {r:+.2f}", fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Mean TFR (2000–2023)")
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Cross-country cultural correlates of fertility (n=14)", fontsize=12)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/crosssection_scatter.png", dpi=200)
plt.close()
out("\n  Saved -> figures/crosssection_scatter.png")

# =========================================================================
# (C) Cautious interpretation
# =========================================================================
out("\n" + "=" * 68)
out("(C) Interpretation")
out("=" * 68)

m_muslim = smf.ols("mean_tfr ~ muslim_share", data=cs).fit()
m_both   = smf.ols("mean_tfr ~ muslim_share + smam_female", data=cs).fit()

out(f"  Muslim share alone: R2 = {m_muslim.rsquared:.3f}")
out(f"  Muslim + SMAM together: R2 = {m_both.rsquared:.3f}")
out(f"  Bivariate TFR-schooling correlation: r = "
    f"{cs['mean_tfr'].corr(cs['female_mean_schooling']):+.3f} — weak, and consistent")
out("  with the Soviet-era compression of female education across the region.\n")

out("The cross-section is consistent with the interpretation that religious-cultural")
out("factors and nuptiality regimes are associated with the fertility gap that survives")
out("economic controls in Layer A. It cannot identify these factors causally: n=14 is")
out("too small, cultural variables are collinear with Central Asia status, and cross-")
out("sectional averages cannot distinguish religion from ethnicity, family norms, or")
out("historical legacy. Azerbaijan (95% Muslim, mean TFR 1.88) also shows that Muslim")
out("share alone does not determine fertility — the story involves religion INTERACTING")
out("with nuptiality patterns and demographic history, not religion by itself.")

# =========================================================================
# Save
# =========================================================================
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/crosssection_results.txt", "w") as f:
    f.write("CROSS-SECTION ANALYSIS — cultural correlates of the CA fertility "
            "premium (n=14)\n\n")
    f.write("\n".join(lines))
out("\nSaved -> data/processed/crosssection_results.txt")