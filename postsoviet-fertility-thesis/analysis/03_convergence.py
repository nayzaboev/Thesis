"""
03_convergence.py
-----------------
Convergence analysis for Part 1: did post-Soviet fertility levels become more
similar between 2000 and 2023, and did initially high-fertility countries fall
faster than initially low-fertility ones?

Two distinct concepts are tested, and they are NOT the same thing:

  SIGMA-CONVERGENCE (dispersion)
      Does cross-country dispersion in TFR shrink over time?
      Measured by the coefficient of variation (CV = sd / mean) computed
      across countries within each year, reported for all 14 countries and
      separately within each bloc and sub-group. A formal trend test regresses
      CV on a linear year term (Newey-West SEs, since CV series are serially
      correlated).

  BETA-CONVERGENCE (catch-up)
      Do countries that started higher experience larger subsequent declines?
      Estimated as a cross-section over the 14 countries:

          (1/T) * ln(TFR_2023 / TFR_2000)  =  alpha + beta * ln(TFR_2000) + e

      beta < 0 indicates catch-up. Reported with HC3 standard errors because
      n = 14. A conditional version adds the Central Asia dummy to test whether
      Central Asia followed a different trajectory given its starting level.

  IMPORTANT CAVEAT (Galton's fallacy / Quah 1993):
      Beta-convergence is necessary but NOT sufficient for sigma-convergence.
      A negative beta can arise purely from regression to the mean when the
      initial level is measured with error. Both are therefore reported, and
      neither is interpreted causally.

  All results are DESCRIPTIVE. With 14 countries these regressions summarise
  the pattern in the data; they do not identify a convergence mechanism.

Input:   data/processed/master_tfr.csv
Outputs: data/processed/cv_all.csv
         data/processed/cv_by_bloc.csv
         data/processed/cv_by_subgroup.csv
         data/processed/cv_trend_tests.csv
         data/processed/peaks.csv
         data/processed/beta_convergence.csv
         data/processed/convergence_results.txt
         figures/fig5_sigma_convergence.png

Run from repo root:  python analysis/03_convergence.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/master_tfr.csv")

lines = []
def out(s=""):
    print(s); lines.append(s)


def cv(s):
    """Coefficient of variation: sd / mean. Scale-free dispersion measure."""
    return s.std(ddof=1) / s.mean()


# =========================================================================
# (A) SIGMA-CONVERGENCE — dispersion across countries, by year
# =========================================================================
out("=" * 72)
out("(A) SIGMA-CONVERGENCE — cross-country dispersion in TFR")
out("=" * 72)
out("Coefficient of variation (sd/mean) computed across countries within")
out("each year. Falling CV = countries becoming more similar.")
out()

cv_all = df.groupby("year")["tfr"].apply(cv)
cv_by_bloc = df.groupby(["bloc", "year"])["tfr"].apply(cv).unstack(0)
cv_by_subgroup = df.groupby(["subgroup", "year"])["tfr"].apply(cv).unstack(0)

snapshot_years = [2000, 2005, 2010, 2015, 2020, 2023]

out("CV across all 14 countries:")
out(cv_all.round(3).loc[snapshot_years].to_string())
out()
out("CV within each bloc:")
out(cv_by_bloc.round(3).loc[snapshot_years].to_string())
out()
out("CV within each sub-group:")
out(cv_by_subgroup.round(3).loc[snapshot_years].to_string())

# --- Formal trend test on each CV series ---
out()
out("-" * 72)
out("Trend test: regress CV on a linear year term.")
out("Newey-West SEs (4 lags) because CV series are strongly serially correlated.")
out("A negative, significant slope indicates sigma-convergence.")
out()

trend_rows = []
series = {"All 14 countries": cv_all}
for c in cv_by_bloc.columns:
    series[f"Bloc: {c}"] = cv_by_bloc[c]
for c in cv_by_subgroup.columns:
    series[f"Sub-group: {c}"] = cv_by_subgroup[c]

for label, s in series.items():
    d = pd.DataFrame({"cv": s.values, "year": s.index.astype(int)}).dropna()
    d["t"] = d["year"] - d["year"].min()
    m = smf.ols("cv ~ t", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    slope, pval = m.params["t"], m.pvalues["t"]
    direction = ("convergence" if slope < 0 else "divergence")
    sig = "significant" if pval < 0.05 else "not significant"
    out(f"  {label:32s}: slope/yr = {slope:+.5f}  p = {pval:.3f}   "
        f"({direction}, {sig})")
    trend_rows.append({"series": label, "cv_slope_per_year": round(slope, 6),
                       "p_value": round(pval, 4),
                       "cv_start": round(s.iloc[0], 4),
                       "cv_end": round(s.iloc[-1], 4)})

out()
out("  Reading: a negative slope means the countries in that grouping grew more")
out("  alike; a positive slope means they diverged. Note that dispersion can fall")
out("  within a bloc while the gap BETWEEN blocs widens - these are separate facts.")

# =========================================================================
# (B) BETA-CONVERGENCE — do initially high-fertility countries fall faster?
# =========================================================================
out()
out("=" * 72)
out("(B) BETA-CONVERGENCE — catch-up across the 14 countries")
out("=" * 72)

first_year, last_year = int(df["year"].min()), int(df["year"].max())
T = last_year - first_year

wide = df.pivot(index="country", columns="year", values="tfr")
bc = pd.DataFrame({
    "tfr_start": wide[first_year],
    "tfr_end": wide[last_year],
}).reset_index()
bc = bc.merge(df.groupby("country")[["bloc", "subgroup"]].first().reset_index(),
              on="country")
bc["ca"] = (bc["bloc"] == "Central Asia").astype(int)
bc["ln_tfr_start"] = np.log(bc["tfr_start"])
bc["growth"] = (np.log(bc["tfr_end"]) - np.log(bc["tfr_start"])) / T
bc["abs_change"] = bc["tfr_end"] - bc["tfr_start"]

out(f"Specification: (1/{T}) * ln(TFR_{last_year} / TFR_{first_year}) "
    f"= alpha + beta * ln(TFR_{first_year})")
out(f"n = {len(bc)} countries. HC3 standard errors (small sample).")
out()

m_uncond = smf.ols("growth ~ ln_tfr_start", data=bc).fit(cov_type="HC3")
b = m_uncond.params["ln_tfr_start"]
se = m_uncond.bse["ln_tfr_start"]
pv = m_uncond.pvalues["ln_tfr_start"]
ci = m_uncond.conf_int().loc["ln_tfr_start"]

out("  UNCONDITIONAL beta-convergence:")
out(f"    beta = {b:+.4f}  (HC3 SE {se:.4f}, p = {pv:.3f})")
out(f"    95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]     R2 = {m_uncond.rsquared:.3f}")
if b < 0 and pv < 0.05:
    out("    => Negative and significant: initially higher-fertility countries")
    out("       declined faster over the period (catch-up pattern).")
elif b < 0:
    out("    => Negative but not statistically significant at the 5% level.")
else:
    out("    => Positive: no evidence of catch-up. Initially higher-fertility")
    out("       countries did NOT decline faster; if anything the opposite.")

# --- Conditional on Central Asia ---
m_cond = smf.ols("growth ~ ln_tfr_start + ca", data=bc).fit(cov_type="HC3")
out()
out("  CONDITIONAL beta-convergence (adds the Central Asia dummy):")
for v in ["ln_tfr_start", "ca"]:
    out(f"    {v:16s}: {m_cond.params[v]:+.4f}  "
        f"(HC3 SE {m_cond.bse[v]:.4f}, p = {m_cond.pvalues[v]:.3f})")
out(f"    R2 = {m_cond.rsquared:.3f}")
out("    Reading: the CA dummy asks whether Central Asian countries followed a")
out("    different trajectory once their 2000 starting level is accounted for.")
out("    A positive CA coefficient means Central Asia declined less (or rose")
out("    more) than its starting level alone would predict.")

out()
out("  CAVEAT (Galton's fallacy, Quah 1993): beta-convergence is necessary but")
out("  NOT sufficient for sigma-convergence, and a negative beta can arise from")
out("  regression to the mean if initial TFR is measured with error. Read")
out("  sections (A) and (B) together, and treat both as descriptive summaries")
out("  at n = 14.")

# =========================================================================
# (C) PEAK YEAR PER COUNTRY
# =========================================================================
out()
out("=" * 72)
out("(C) PEAK TFR YEAR PER COUNTRY")
out("=" * 72)
out("The year each country recorded its highest TFR in 2000-2023. Late peaks")
out("indicate recuperation or a genuine reversal rather than continued decline.")
out()

peaks = (df.loc[df.groupby("country")["tfr"].idxmax()]
           [["country", "subgroup", "year", "tfr"]]
           .sort_values(["subgroup", "country"])
           .rename(columns={"year": "peak_year", "tfr": "peak_tfr"}))
peaks["peak_tfr"] = peaks["peak_tfr"].round(2)
out(peaks.to_string(index=False))
out()
late = int((peaks["peak_year"] >= 2015).sum())
out(f"  Countries peaking in 2015 or later: {late} of {len(peaks)}")

# =========================================================================
# (D) FIGURE — sigma convergence
# =========================================================================
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].plot(cv_all.index, cv_all.values, color="#14213d", lw=2.2)
axes[0].set_title("Dispersion across all 14 countries", fontsize=11)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Coefficient of variation of TFR")
axes[0].grid(alpha=0.3)

colors = {"Central Asia": "#c1121f", "Rest of post-Soviet": "#14213d"}
for c in cv_by_bloc.columns:
    axes[1].plot(cv_by_bloc.index, cv_by_bloc[c].values,
                 lw=2.2, label=c, color=colors.get(c))
axes[1].set_title("Dispersion within each bloc", fontsize=11)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Coefficient of variation of TFR")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle("Sigma-convergence in post-Soviet fertility, 2000-2023",
             fontsize=12, weight="bold")
fig.text(0.5, -0.02,
         "Source: author's calculations using UN World Population Prospects 2024 "
         "(Median variant).",
         ha="center", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig("figures/fig5_sigma_convergence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
out()
out("Saved -> figures/fig5_sigma_convergence.png")

# =========================================================================
# Save
# =========================================================================
os.makedirs("data/processed", exist_ok=True)
cv_all.rename("cv").to_csv("data/processed/cv_all.csv")
cv_by_bloc.to_csv("data/processed/cv_by_bloc.csv")
cv_by_subgroup.to_csv("data/processed/cv_by_subgroup.csv")
peaks.to_csv("data/processed/peaks.csv", index=False)

beta_out = bc[["country", "bloc", "subgroup", "tfr_start", "tfr_end",
               "abs_change", "growth"]].copy()
beta_out[["tfr_start", "tfr_end", "abs_change"]] = \
    beta_out[["tfr_start", "tfr_end", "abs_change"]].round(3)
beta_out["growth"] = beta_out["growth"].round(5)
beta_out.to_csv("data/processed/beta_convergence.csv", index=False)
pd.DataFrame(trend_rows).to_csv("data/processed/cv_trend_tests.csv", index=False)

with open("data/processed/convergence_results.txt", "w") as f:
    f.write("CONVERGENCE ANALYSIS - sigma-convergence (dispersion) and "
            "beta-convergence (catch-up)\n"
            "All results are descriptive; n = 14 countries.\n\n")
    f.write("\n".join(lines))

out("Saved -> data/processed/cv_all.csv, cv_by_bloc.csv, cv_by_subgroup.csv,")
out("         cv_trend_tests.csv, peaks.csv, beta_convergence.csv,")
out("         convergence_results.txt")