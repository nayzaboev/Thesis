"""
02_trend_plots.py
------------------
Generates Part I trend figures from data/processed/master_tfr.csv.
Run from the repo root:  python analysis/02_trend_plots.py
"""

import os
import pandas as pd 
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Setup ---
df = pd.read_csv("data/processed/master_tfr.csv")
os.makedirs("figures", exist_ok=True)
REPLACEMENT = 2.1  # replacement-level fertility, drawn as a reference line

SUB_COLORS = {
    "Central Asia": "#c1121f",   # red — the focus group
    "Eastern European":       "#14213d",   # navy
    "Baltic":       "#4a6fa5",   # blue
    "Caucasus":     "#2a9d8f",   # teal
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

def add_replacement_line(ax):
    ax.axhline(REPLACEMENT, ls="--", lw=1, color="#888")
    ax.text(2000.2, REPLACEMENT + 0.04, "Replacement (2.1)", fontsize=8, color="#666")

PROJECTION_NOTE = ("Note: dashed segments are WPP forward-looking projections rather "
                    "than retrospective estimates; most countries' 2023 values are "
                    "projections (11 of 14 — only Latvia, Lithuania and Russia are "
                    "interpolated in 2023).")

# --- Figure 1: all 14 countries, labels auto-spaced, Central Asia highlighted ---
# Projected years are plotted as a dashed continuation of each country's solid
# (interpolated) line, so 2023 country values are not shown identically to the
# retrospective estimates that precede them.
fig, ax = plt.subplots(figsize=(10, 6))
texts = []  # collect label objects so adjustText can space them out
for country, g in df.groupby("country"):
    g = g.sort_values("year")
    sub = g["subgroup"].iloc[0]
    is_ca = sub == "Central Asia"
    color = SUB_COLORS[sub]
    lw = 2.2 if is_ca else 1.0
    alpha = 1.0 if is_ca else 0.55
    zorder = 3 if is_ca else 1

    proj = g.loc[g["estimate_method"] == "Projection", "year"]
    if len(proj):
        first_proj_year = int(proj.min())
        solid = g[g["year"] <= first_proj_year]
        dashed = g[g["year"] >= first_proj_year]
        ax.plot(dashed["year"], dashed["tfr"], color=color, lw=lw, alpha=alpha,
                zorder=zorder, linestyle="--")
    else:
        solid = g
    ax.plot(solid["year"], solid["tfr"], color=color, lw=lw, alpha=alpha, zorder=zorder)

    last = g.iloc[-1]
    texts.append(ax.text(last["year"], last["tfr"], country,
                         fontsize=7.5, color=SUB_COLORS[sub],
                         weight="bold" if is_ca else "normal"))

# nudge labels apart so none overlap
adjust_text(texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5),
            expand_text=(1.2, 1.4))

add_replacement_line(ax)
ax.set_title("Total Fertility Rate, 14 post-Soviet states, 2000–2023\n"
             "(Central Asia highlighted in red)", fontsize=12, weight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Total Fertility Rate")
ax.set_xlim(2000, 2026)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
fig.text(0.5, -0.03, PROJECTION_NOTE, ha="center", fontsize=8, color="#555", wrap=True)
fig.tight_layout()
fig.savefig("figures/fig1_all_countries.png", bbox_inches="tight")
plt.close(fig)

# --- Figure 2: Central Asia vs Rest, mean + range band ---
# Dash the bloc-mean line for any year in which a MAJORITY of that bloc's
# countries contribute a projected (rather than interpolated) TFR value, so
# the mean line is not shown as equally reliable across its whole length.
fig, ax = plt.subplots(figsize=(9, 5.5))
for bloc, color in [("Central Asia", "#c1121f"), ("Rest of post-Soviet", "#14213d")]:
    bloc_df = df[df["bloc"] == bloc]
    g = bloc_df.groupby("year")["tfr"]
    mean, lo, hi = g.mean(), g.min(), g.max()
    proj_frac = bloc_df.groupby("year")["estimate_method"].apply(
        lambda s: (s == "Projection").mean())
    majority_proj_years = proj_frac[proj_frac > 0.5].index
    boundary = int(majority_proj_years.min()) if len(majority_proj_years) else None

    ax.plot(mean.index, mean.values, color=color, lw=2.5, label=f"{bloc} (mean)")
    if boundary is not None:
        tail = mean.loc[mean.index >= boundary - 1]
        ax.plot(tail.index, tail.values, color=color, lw=2.5, linestyle="--")
    ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.12)
add_replacement_line(ax)
ax.set_title("Central Asia vs. Rest of post-Soviet space\n"
             "Mean TFR (line) and country range (band), 2000–2023",
             fontsize=12, weight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Total Fertility Rate")
ax.set_xlim(2000, 2023)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(frameon=False, loc="upper left")
fig.text(0.5, -0.05,
         "Note: dashed segment = year in which a majority of the bloc's countries "
         "contribute a WPP-projected (not interpolated) value.",
         ha="center", fontsize=8, color="#555", wrap=True)
fig.tight_layout()
fig.savefig("figures/fig2_bloc_means.png", bbox_inches="tight")
plt.close(fig)

# --- Figure 3: four sub-group means ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for sub, color in SUB_COLORS.items():
    g = df[df["subgroup"] == sub].groupby("year")["tfr"].mean()
    ax.plot(g.index, g.values, color=color, lw=2.3, label=sub, marker="o", ms=3)
add_replacement_line(ax)
ax.set_title("Mean TFR by sub-group, 2000–2023", fontsize=12, weight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Total Fertility Rate")
ax.set_xlim(2000, 2023)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/fig3_subgroups.png", bbox_inches="tight")
plt.close(fig)

# --- Figure 4: the gap (Central Asia mean − Rest mean) ---
ca = df[df["bloc"] == "Central Asia"].groupby("year")["tfr"].mean()
rest = df[df["bloc"] == "Rest of post-Soviet"].groupby("year")["tfr"].mean()
gap = ca - rest
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(gap.index, gap.values, color="#c1121f", alpha=0.85)
ax.set_title("Fertility gap: Central Asia mean − Rest mean, 2000–2023",
             fontsize=12, weight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("TFR difference (children per woman)")
ax.set_xlim(1999.3, 2023.7)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.axhline(0, color="#333", lw=0.8)
fig.tight_layout()
fig.savefig("figures/fig4_gap.png", bbox_inches="tight")
plt.close(fig)

# --- Console summary ---
print("Saved 4 figures to figures/")
summary = pd.DataFrame({
    "Central Asia": ca.round(2), "Rest": rest.round(2), "Gap": gap.round(2),
}).loc[[2000, 2005, 2010, 2015, 2020, 2023]]
print(summary.to_string())