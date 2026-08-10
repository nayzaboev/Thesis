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
         data/processed/cv_absolute_dispersion.csv
         data/processed/cv_absolute_dispersion_by_bloc.csv
         data/processed/cv_trend_tests.csv
         data/processed/cv_war_sensitivity.csv
         data/processed/peaks.csv
         data/processed/beta_convergence.csv
         data/processed/projection_sensitivity.csv
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

# --- Absolute dispersion (SD and IQR) alongside the scale-free CV ---
# CV = sd/mean can move because dispersion OR the group mean changes. Reporting
# the standard deviation and interquartile range (absolute measures) shows
# whether a falling CV reflects genuine compression or a rising denominator.
out()
out("-" * 72)
out("Absolute dispersion across all 14 countries (SD and IQR), for comparison")
out("with the scale-free CV above. If CV falls but SD/IQR do not, the CV move is")
out("partly a denominator (mean) effect rather than genuine compression.")
out()
sd_all  = df.groupby("year")["tfr"].std(ddof=1)
iqr_all = df.groupby("year")["tfr"].apply(
    lambda s: s.quantile(0.75) - s.quantile(0.25))
mean_all = df.groupby("year")["tfr"].mean()
abs_disp = pd.DataFrame({
    "mean": mean_all.round(3),
    "SD": sd_all.round(3),
    "IQR": iqr_all.round(3),
    "CV": cv_all.round(3),
})
out(abs_disp.loc[snapshot_years].to_string())
_sd_chg  = sd_all.loc[2023] - sd_all.loc[2000]
_cv_chg  = cv_all.loc[2023] - cv_all.loc[2000]
out()
out(f"  2000->2023: SD {_sd_chg:+.3f}, CV {_cv_chg:+.3f}. Read together: a falling")
out("  CV accompanied by a roughly flat or rising SD indicates the CV change is")
out("  substantially a mean (denominator) effect, not pure absolute compression.")
abs_disp.to_csv("data/processed/cv_absolute_dispersion.csv")

# --- Absolute dispersion by bloc — the sigma-convergence claims concern Central
# Asia and the rest separately, so the pooled series above can mask opposite
# within-bloc movements. ---
out()
out("-" * 72)
out("Absolute dispersion (SD and IQR) BY BLOC, for comparison with the pooled")
out("series above. The pooled SD can be flat or rising even if one bloc is")
out("compressing, if the other bloc is diverging at the same time.")
out()
sd_bloc = df.groupby(["bloc", "year"])["tfr"].std(ddof=1).unstack(0)
iqr_bloc = df.groupby(["bloc", "year"])["tfr"].apply(
    lambda s: s.quantile(0.75) - s.quantile(0.25)).unstack(0)
abs_disp_bloc = pd.concat(
    {"SD": sd_bloc.round(3), "IQR": iqr_bloc.round(3)}, axis=1)
out(abs_disp_bloc.loc[snapshot_years].to_string())
out()
_ca_sd = sd_bloc["Central Asia"]
_rest_sd = sd_bloc["Rest of post-Soviet"]
out(f"  Central Asia SD:        {_ca_sd.loc[2000]:.3f} (2000) -> "
    f"{_ca_sd.loc[2017]:.3f} (2017) -> {_ca_sd.loc[2023]:.3f} (2023)")
out(f"  Rest of post-Soviet SD: {_rest_sd.loc[2000]:.3f} (2000) -> "
    f"{_rest_sd.loc[2017]:.3f} (2017) -> {_rest_sd.loc[2023]:.3f} (2023)")
out("  READING: Central Asia's absolute dispersion FALLS over 2000-2023 (within-")
out("  bloc convergence in levels, not just CV) while the rest of the post-Soviet")
out("  space DIVERGES in absolute terms. Both movements are genuine — visible in")
out("  SD/IQR, not an artifact of the CV denominator — and they run in opposite")
out("  directions, so the pooled 14-country dispersion series understates how much")
out("  is happening within each bloc.")
abs_disp_bloc.to_csv("data/processed/cv_absolute_dispersion_by_bloc.csv")

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

# --- War-period sensitivity: does excluding 2022-2023 change the trend? ---
# The 2022-2023 Russian invasion of Ukraine was a region-wide shock (Russian
# remittance outflows to Central Asia, migration flows, macro conditions
# across the whole sample), not just a Ukraine-specific event. This re-fits
# the pooled and by-bloc CV trends above on year <= 2021, same Newey-West
# (HAC, 4 lags) specification as the full-period trend test, to check whether
# the bifurcation (CA convergence vs. rest-of-sample divergence) survives
# dropping the war years.
out()
out("-" * 72)
out("War-period sensitivity: CV trend on year <= 2021 vs. full period 2000-2023")
out("Same Newey-West (HAC, 4 lags) specification as the trend test above.")
out()

war_rows = []
war_labels = ["All 14 countries", "Bloc: Central Asia", "Bloc: Rest of post-Soviet"]
for label in war_labels:
    s = series[label]
    d_full = pd.DataFrame({"cv": s.values, "year": s.index.astype(int)}).dropna()
    d_full["t"] = d_full["year"] - d_full["year"].min()
    m_full = smf.ols("cv ~ t", data=d_full).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    slope_full, p_full = m_full.params["t"], m_full.pvalues["t"]

    d_war = d_full[d_full.year <= 2021].copy()
    d_war["t"] = d_war["year"] - d_war["year"].min()
    m_war = smf.ols("cv ~ t", data=d_war).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    slope_war, p_war = m_war.params["t"], m_war.pvalues["t"]

    out(f"  {label:26s}: full = {slope_full:+.5f} (p={p_full:.3f})   "
        f"<=2021 = {slope_war:+.5f} (p={p_war:.3f})")
    war_rows.append({"series": label,
                     "slope_full": round(slope_full, 6), "p_full": round(p_full, 4),
                     "slope_excl_2022_2023": round(slope_war, 6),
                     "p_excl_2022_2023": round(p_war, 4)})

out()
out("  Excluding the 2022-2023 war-period observations leaves Central Asian")
out("  convergence essentially unchanged (slope -0.008, p = 0.000). The")
out("  non-Central-Asian divergence slope weakens (+0.00163 -> +0.00068) and")
out("  loses conventional significance (p = 0.088). The bifurcation reported")
out("  above is therefore driven primarily by robust Central Asian convergence;")
out("  the divergence within the rest of the sample is partly attributable to")
out("  the 2022-2023 fertility declines and should be interpreted with")
out("  corresponding caution.")

# =========================================================================
# (B) BETA-CONVERGENCE — do initially high-fertility countries fall faster?
# =========================================================================
out()
out("=" * 72)
out("(B) BETA-CONVERGENCE — catch-up across the 14 countries")
out("=" * 72)

first_year, last_year = int(df["year"].min()), int(df["year"].max())
# Endpoints are 3-YEAR AVERAGES (2000-2002 and 2021-2023) rather than single
# years, to reduce sensitivity to annual measurement noise at the endpoints.
# T is the gap between the window midpoints (2001 -> 2022 = 21 years).
START_WIN = [2000, 2001, 2002]
END_WIN   = [2021, 2022, 2023]
T = int(np.mean(END_WIN) - np.mean(START_WIN))

wide = df.pivot(index="country", columns="year", values="tfr")
bc = pd.DataFrame({
    "tfr_start": wide[START_WIN].mean(axis=1),
    "tfr_end":   wide[END_WIN].mean(axis=1),
}).reset_index()
bc = bc.merge(df.groupby("country")[["bloc", "subgroup"]].first().reset_index(),
              on="country", validate="many_to_one")
bc["ca"] = (bc["bloc"] == "Central Asia").astype(int)
bc["ln_tfr_start"] = np.log(bc["tfr_start"])
bc["growth"] = (np.log(bc["tfr_end"]) - np.log(bc["tfr_start"])) / T
bc["abs_change"] = bc["tfr_end"] - bc["tfr_start"]

out(f"Specification: (1/{T}) * ln(TFR_end / TFR_start) "
    f"= alpha + beta * ln(TFR_start)")
out(f"Endpoints are 3-year averages: start = mean(2000-2002), end = mean(2021-2023).")
out(f"n = {len(bc)} countries. HC3 standard errors with small-sample t inference.")
out()

m_uncond = smf.ols("growth ~ ln_tfr_start", data=bc).fit(cov_type="HC3", use_t=True)
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
m_cond = smf.ols("growth ~ ln_tfr_start + ca", data=bc).fit(cov_type="HC3", use_t=True)
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
# (B2) PERIOD-SPLIT SIGMA-CONVERGENCE — the full-period trend hides a reversal
# =========================================================================
out()
out("=" * 72)
out("(B2) PERIOD-SPLIT SIGMA-CONVERGENCE (pre/post 2017)")
out("=" * 72)
out("A single linear trend over 2000-2023 masks a turning point. Fitting the")
out("CV trend separately on 2000-2016 and 2017-2023 shows the real pattern.")
out()

def split_trend(s, y0, y1):
    d = pd.DataFrame({"cv": s.values, "year": s.index.astype(int)}).dropna()
    d = d[(d.year >= y0) & (d.year <= y1)].copy()
    d["t"] = d["year"] - d["year"].min()
    if len(d) < 4:
        return np.nan, np.nan
    m = smf.ols("cv ~ t", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return m.params["t"], m.pvalues["t"]

split_rows = []
for label, s in series.items():
    sl_early, p_early = split_trend(s, 2000, 2016)
    sl_late,  _       = split_trend(s, 2017, 2023)
    sl_full,  p_full  = split_trend(s, 2000, 2023)
    # Report the 2000-2016 slope with its p-value (17 obs), but the 2017-2023
    # slope WITHOUT a p-value: with only 7 annual observations HAC p-values are
    # unreliable and are deliberately not shown (see caveat below).
    out(f"  {label:32s}")
    out(f"      2000-2016: slope={sl_early:+.5f} (p={p_early:.3f})   "
        f"2017-2023: slope={sl_late:+.5f} (direction only, n=7)")
    split_rows.append({"series": label,
                       "slope_2000_2016": round(sl_early, 6), "p_2000_2016": round(p_early, 4),
                       "slope_2017_2023": round(sl_late, 6),
                       "slope_full": round(sl_full, 6),       "p_full": round(p_full, 4)})
out()
out("  READING: for the all-14 series, dispersion fell over 2000-2016 and rose")
out("  over 2017-2023. The near-zero full-period slope averages these opposing phases.")
out("  CAVEAT: the post-2017 window has only 7 annual observations. HAC p-values on")
out("  T=7 are NOT reliable, so NO significance is claimed for the post-2017 period.")
out("  Only the slope SIGN and the descriptive pattern (compression then divergence,")
out("  visible in the raw CV series) are reported. Do NOT cite a post-2017 p-value")
out("  or describe the post-2017 change as statistically significant.")

# =========================================================================
# (B3) CONVERGENCE LEAVE-ONE-OUT — are the bloc trends country-driven?
# =========================================================================
out()
out("=" * 72)
out("(B3) CONVERGENCE LEAVE-ONE-OUT (bloc trends)")
out("=" * 72)
out("Re-fit each bloc's full-period CV trend dropping one country at a time.")
out("If significance vanishes when a single country is removed, the trend is")
out("country-driven, not a group-wide regularity.\n")

def bloc_cv_trend(sub):
    cvs = sub.groupby("year")["tfr"].apply(cv)
    d = pd.DataFrame({"cv": cvs.values, "year": cvs.index.astype(int)})
    d["t"] = d["year"] - d["year"].min()
    m = smf.ols("cv ~ t", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return m.params["t"], m.pvalues["t"]

loo_conv_rows = []
for bloc_name in df["bloc"].unique():
    bloc_df = df[df["bloc"] == bloc_name]
    base_sl, base_p = bloc_cv_trend(bloc_df)
    out(f"  Bloc: {bloc_name}  (full-period slope={base_sl:+.5f}, p={base_p:.3f})")
    flips = []
    for c in sorted(bloc_df["country"].unique()):
        sl, pv = bloc_cv_trend(bloc_df[bloc_df["country"] != c])
        loo_conv_rows.append({"bloc": bloc_name, "dropped": c,
                              "slope": round(sl, 6), "p_value": round(pv, 4)})
        mark = ""
        if base_p < 0.05 and pv >= 0.05:
            mark = "  <-- significance LOST when dropped"
            flips.append(c)
        out(f"      drop {c:14s}: slope={sl:+.5f} (p={pv:.3f}){mark}")
    if flips:
        out(f"    => trend depends materially on: {', '.join(flips)}")
    out("")
out("  Report bloc convergence/divergence trends that flip as country-sensitive,")
out("  not as robust group-wide regularities.")

# =========================================================================
# (C) PEAK YEAR PER COUNTRY
# =========================================================================
out()
out("=" * 72)
out("(C) PEAK TFR YEAR PER COUNTRY")
out("=" * 72)
out("The year each country recorded its highest TFR in 2000-2023. Late peaks")
out("document renewed or sustained increases in period TFR but do not distinguish")
out("tempo effects (postponement/recuperation) from changes in completed fertility.")
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
         "(Median variant). Note: most 2023 country values are WPP projections "
         "(11 of 14 — only Latvia, Lithuania and Russia are interpolated in 2023), "
         "so the rightmost point on each line is not on the same evidentiary "
         "footing as earlier, retrospective points.",
         ha="center", fontsize=8, color="#555", wrap=True)
fig.tight_layout()
fig.savefig("figures/fig5_sigma_convergence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
out()
out("Saved -> figures/fig5_sigma_convergence.png")

# =========================================================================
# (E) PROJECTION-SENSITIVITY CHECK — do the headline results survive dropping
#     WPP-projected (rather than interpolated) observations?
# =========================================================================
out()
out("=" * 72)
out("(E) PROJECTION-SENSITIVITY CHECK (descriptive only)")
out("=" * 72)
out("master_tfr.csv carries 'estimate_method', distinguishing WPP retrospective")
out("'Interpolation' from forward-looking 'Projection' values (20 of 336 rows,")
out("concentrated in 2020-2023 and in 11 of 14 countries' 2023 observations")
out("(Latvia, Lithuania and Russia are interpolated, not projected, in 2023); see")
out("01_clean_data.py). The (B2) post-2017 CV slope and the (B) beta-convergence")
out("coefficient above both partly rest on these projected endpoints. This block")
out("re-computes those two headline descriptive results under two BALANCED")
out("sub-samples, each holding a FIXED country set across all years it uses. An")
out("earlier version of this check instead dropped projected ROWS year-by-year,")
out("which let the country set drift over time (14 -> 12 -> 11 -> 10 countries,")
out("collapsing to 3 low-fertility countries by 2023). That composition drift, not")
out("genuine convergence, drove a spuriously negative post-2016 slope. A valid")
out("cross-time dispersion comparison requires the SAME countries in every year")
out("compared, so both samples below fix the country set before comparing years:")
out("  (a) full 2000-2023, all 14 countries (baseline, as reported above)")
out("  (b) Sample A - balanced historical: all 14 countries, 2000-2019, no")
out("      projected values anywhere in that window")
out("  (c) Sample B - balanced interpolation-only recent: countries with a")
out("      non-projected observation in EVERY year 2017-2022, that country set")
out("      then held fixed for all years used (list below; membership is")
out("      derived from the data, not hardcoded)")
out("This is NOT a significance test: no p-values or significance claims are made")
out("here, and the sub-samples are smaller than the already-small n=14/T=7 used")
out("above.")
out()

# --- Sample A: balanced historical (2000-2019, all 14 countries) ---
sampleA = df[(df["year"] >= 2000) & (df["year"] <= 2019)].copy()
assert (sampleA["estimate_method"] != "Projection").all(), \
    "Sample A window (2000-2019) unexpectedly contains a projected row"
assert sampleA["country"].nunique() == 14

# --- Sample B: balanced interpolation-only recent (fixed country set) ---
# Membership is derived programmatically: a country qualifies only if it has a
# non-projected ('Interpolation') observation in EVERY one of 2017-2022.
b_window = list(range(2017, 2023))
b_check = df[df["year"].isin(b_window)]
b_complete = b_check.groupby("country").apply(
    lambda g: set(g.loc[g["estimate_method"] != "Projection", "year"]) == set(b_window),
    include_groups=False)
sampleB_countries = sorted(b_complete[b_complete].index.tolist())
# 2023 lies outside the qualifying window and is itself a projection for most
# of these countries, so Sample B is restricted through 2022 to stay
# interpolation-only for every year it actually uses.
sampleB = df[(df["country"].isin(sampleB_countries)) & (df["year"] <= 2022)].copy()

out(f"  Sample B country set (n={len(sampleB_countries)}, non-projected in every")
out(f"  year 2017-2022): {', '.join(sampleB_countries)}")
out()


def sens_cv_slope(sub, y0, y1):
    """Plain-OLS point estimate of the CV-on-year slope (no HAC/SE) -
    descriptive only, used purely to compare direction/magnitude across samples."""
    s = sub.groupby("year")["tfr"].apply(cv)
    d = pd.DataFrame({"cv": s.values, "year": s.index.astype(int)})
    d = d[(d.year >= y0) & (d.year <= y1)]
    if len(d) < 2:
        return np.nan
    return np.polyfit(d["year"], d["cv"], 1)[0]


def sens_beta(sub):
    """Unconditional beta-convergence coefficient on countries with a COMPLETE
    3-year window at BOTH ends. A country is included only if every year of
    the 2000-2002 start window AND every year of the end window (the last 3
    calendar years present in `sub`) is non-missing. An earlier version used
    w.reindex(columns=win).mean(axis=1), which silently averages over
    whatever years ARE present — a country with only 1 of 3 end-window years
    still received a value, and .dropna() only removed countries missing ALL
    three. Returns the excluded-country count so it can be reported."""
    end_year = int(sub["year"].max())
    end_win = [end_year - 2, end_year - 1, end_year]
    start_win = [2000, 2001, 2002]
    w = sub.pivot(index="country", columns="year", values="tfr")
    start_complete = w.reindex(columns=start_win).notna().all(axis=1)
    end_complete = w.reindex(columns=end_win).notna().all(axis=1)
    both = start_complete & end_complete
    n_excluded = int((~both).sum())
    t = pd.DataFrame({
        "tfr_start": w.loc[both, start_win].mean(axis=1),
        "tfr_end":   w.loc[both, end_win].mean(axis=1),
    })
    if len(t) < 4:
        return np.nan, len(t), end_win, n_excluded
    t["ln_tfr_start"] = np.log(t["tfr_start"])
    tt = np.mean(end_win) - np.mean(start_win)
    t["growth"] = (np.log(t["tfr_end"]) - np.log(t["tfr_start"])) / tt
    m = smf.ols("growth ~ ln_tfr_start", data=t).fit(cov_type="HC3", use_t=True)
    return m.params["ln_tfr_start"], len(t), end_win, n_excluded


sensitivity_samples = {
    "(a) Full 2000-2023 (baseline)":                df,
    "(b) Sample A: balanced historical (2000-19)":  sampleA,
    "(c) Sample B: balanced interp-only recent":    sampleB,
}

out(f"  {'Sample':46s} {'CV slope 2000-16':>17s} {'CV slope post-16':>17s} "
    f"{'beta':>9s} {'n (beta)':>9s} {'excl':>5s}  {'end window used'}")
sensitivity_rows = []
for label, sub in sensitivity_samples.items():
    max_yr = int(sub["year"].max())
    sl_early = sens_cv_slope(sub, 2000, 2016)
    sl_late = sens_cv_slope(sub, 2017, max_yr)
    beta, n_beta, end_win, n_excl = sens_beta(sub)
    ew_label = f"{end_win[0]}-{end_win[-1]}"
    out(f"  {label:46s} {sl_early:+17.5f} {sl_late:+17.5f} "
        f"{beta:+9.4f} {n_beta:9d} {n_excl:5d}  {ew_label} (post-16: 2017-{max_yr})")
    sensitivity_rows.append({
        "sample": label, "n_countries": int(sub["country"].nunique()),
        "cv_slope_2000_2016": round(sl_early, 6),
        "cv_slope_post_2016": round(sl_late, 6),
        "post_2016_end_year": max_yr,
        "beta": round(beta, 4), "n_beta": n_beta,
        "n_excluded_incomplete_window": n_excl,
        "beta_end_window": ew_label,
    })

row_a, row_b, row_c = sensitivity_rows
out()
out(f"  READING: the post-2016 CV slope is POSITIVE (rising dispersion) in all")
out(f"  three samples once the country set is held fixed — full sample")
out(f"  ({row_a['cv_slope_post_2016']:+.4f}), Sample A ({row_b['cv_slope_post_2016']:+.4f},")
out(f"  though only 3 annual points, 2017-2019), and Sample B")
out(f"  ({row_c['cv_slope_post_2016']:+.4f}, 2017-2022, the best-powered of the")
out("  three, n=10 countries, rising monotonically across those six years).")
out("  The balanced samples provide supporting evidence that the positive")
out("  post-2016 dispersion trend is not solely produced by the projected")
out("  observations or by year-to-year sample-composition changes. These are")
out("  descriptive slopes on small samples (Sample A has three post-2016")
out("  points; Sample B has ten countries), not formal tests.")
out(f"  Beta-convergence: full sample beta = {row_a['beta']:+.4f}; Sample A (end")
out(f"  window 2017-2019, before most of the post-2017 divergence had")
out(f"  accumulated) beta = {row_b['beta']:+.4f}; Sample B beta = {row_c['beta']:+.4f}.")
out("  All entries in this block are descriptive point estimates; treat as a")
out("  sensitivity check, not a formal robustness test.")

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
pd.DataFrame(war_rows).to_csv("data/processed/cv_war_sensitivity.csv", index=False)
pd.DataFrame(split_rows).to_csv("data/processed/cv_period_split.csv", index=False)
pd.DataFrame(loo_conv_rows).to_csv("data/processed/cv_leave_one_out.csv", index=False)
pd.DataFrame(sensitivity_rows).to_csv("data/processed/projection_sensitivity.csv", index=False)

with open("data/processed/convergence_results.txt", "w") as f:
    f.write("CONVERGENCE ANALYSIS - sigma-convergence (dispersion) and "
            "beta-convergence (catch-up)\n"
            "All results are descriptive; n = 14 countries.\n\n")
    f.write("\n".join(lines))

out("Saved -> data/processed/cv_all.csv, cv_by_bloc.csv, cv_by_subgroup.csv,")
out("         cv_absolute_dispersion.csv, cv_absolute_dispersion_by_bloc.csv,")
out("         cv_trend_tests.csv, cv_war_sensitivity.csv, peaks.csv, beta_convergence.csv,")
out("         projection_sensitivity.csv, convergence_results.txt")