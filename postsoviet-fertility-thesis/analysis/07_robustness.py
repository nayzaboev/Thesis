import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

p = pd.read_csv("data/processed/panel.csv")
p["log_gdp"] = np.log(p["gdp_per_capita_usd"])
covs = ["urban_pop_pct", "female_tert_educ_pct", "log_gdp", "flfp_pct", "remittances_gdp_pct"]
X = add_constant(p[covs])
print("=== VIF (>5 concerning, >10 serious) ===")
for i, col in enumerate(X.columns):
    if col != "const":
        print(f"  {col}: {variance_inflation_factor(X.values, i):.2f}")
full = smf.ols("tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct + remittances_gdp_pct", data=p)
m_robust = full.fit(cov_type="cluster", cov_kwds={"groups": p["country"]})
print("\n=== Full model, robust SE clustered by country ===")
print(m_robust.summary().tables[1])
print(f"R-squared: {m_robust.rsquared:.3f}")
drop_remit = smf.ols("tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct", data=p)
m_drop = drop_remit.fit(cov_type="cluster", cov_kwds={"groups": p["country"]})
print("\n=== Model WITHOUT remittances (multicollinearity check) ===")
print(m_drop.summary().tables[1])
print(f"R-squared: {m_drop.rsquared:.3f}")
with open("data/processed/robustness_results.txt", "w") as f:
    f.write("FULL MODEL (robust SE):\n")
    f.write(m_robust.summary().as_text())
    f.write("\n\nMODEL WITHOUT REMITTANCES:\n")
    f.write(m_drop.summary().as_text())
print("\nSaved -> data/processed/robustness_results.txt")        