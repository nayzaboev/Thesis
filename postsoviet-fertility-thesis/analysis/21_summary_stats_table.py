"""
21_summary_stats_table.py
--------------------------
Writes the descriptive-statistics LaTeX table from
data/processed/summary_stats.csv, matching the formatting conventions of
the four existing thesis/tables/*.tex files (booktabs rules, \\small,
footnotesize tablenotes minipage, \\label convention).

GDP is reported here in natural-log terms (log GDP per capita, PPP) rather
than the raw levels stored in summary_stats.csv, because log GDP is the
transformation that actually enters the Layer A/B regressions — see
tab_layerA.tex / tab_layerB.tex, both of which label the control
"log GDP p.c.\\ PPP (lag)". The log-GDP row is therefore computed directly
from data/processed/panel.csv instead of taken from summary_stats.csv.

Output: thesis/tables/tab_summary_stats.tex
Run from repo root:  python analysis/21_summary_stats_table.py
"""

import os
import pandas as pd

stats = pd.read_csv("data/processed/summary_stats.csv").set_index("variable")
panel = pd.read_csv("data/processed/panel.csv")

log_gdp = panel["log_gdp_ppp"].dropna()
log_gdp_row = pd.Series({
    "N": len(log_gdp),
    "mean": log_gdp.mean(),
    "sd": log_gdp.std(ddof=1),
    "min": log_gdp.min(),
    "median": log_gdp.median(),
    "max": log_gdp.max(),
})

ROWS = [
    ("TFR",                       stats.loc["Total fertility rate"]),
    ("log GDP p.c.\\ PPP",        log_gdp_row),
    ("Urban population (\\%)",    stats.loc["Urban population (%)"]),
    ("Remittances (\\% GDP)",     stats.loc["Remittances (% of GDP)"]),
    ("Under-5 mortality",         stats.loc["Under-5 mortality (per 1,000)"]),
]


def fmt(v, n=2):
    return f"{v:.{n}f}"


row_lines = []
for label, r in ROWS:
    row_lines.append(
        f"{label:26s} & {int(r['N']):d} & {fmt(r['mean'])} & {fmt(r['sd'])} & "
        f"{fmt(r['min'])} & {fmt(r['median'])} & {fmt(r['max'])} \\\\"
    )

tex = r"""\begin{table}[htbp]
\centering
\caption{Descriptive statistics, panel variables}
\label{tab:summary_stats}
\small
\begin{tabular}{lrrrrrr}
\toprule
Variable & N & Mean & SD & Min & Median & Max \\
\midrule
""" + "\n".join(row_lines) + r"""
\bottomrule
\end{tabular}

\begin{minipage}{\textwidth}\footnotesize
\vspace{4pt}
\emph{Note:} Fourteen post-Soviet countries, 2000--2023 ($N \leq 336$ country-years).
GDP is reported here in natural-log terms (log GDP per capita, PPP, constant 2021
international \$), the transformation entering the Layer A/B regressions
(Tables~\ref{tab:layerA} and~\ref{tab:layerB}); computed from
\texttt{data/processed/panel.csv} rather than the raw-GDP figures in
\texttt{summary\_stats.csv}. Remittances has eight missing country-year
observations, all in Central Asia ($N=328$); all other variables are complete
($N=336$).
\end{minipage}
\end{table}
"""

os.makedirs("thesis/tables", exist_ok=True)
with open("thesis/tables/tab_summary_stats.tex", "w") as f:
    f.write(tex)
print(tex)
print("Saved -> thesis/tables/tab_summary_stats.tex")
