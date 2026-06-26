import pandas as pd
import statsmodels.formula.api as smf

p = pd.read_csv("data/processed/panel.csv")
import numpy as np
p["log_gdp"] = np.log(p["gdp_per_capita_usd"])
# Model 1: urbanisation alone (strongest correlate)
m1 = smf.ols("tfr ~ urban_pop_pct", data=p).fit()

# Model 2: add female tertiary education
m2 = smf.ols("tfr ~ urban_pop_pct + female_tert_educ_pct", data=p).fit()

# Model 3: add GDP (logged) and female labour
m3 = smf.ols("tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct", data=p).fit()

# Model 4: full model, add remittances
m4 = smf.ols("tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct + remittances_gdp_pct", data=p).fit()

for name, m in [("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)]:
    print(f"\n===== {name} =====")
    print(m.summary().tables[1])   # coefficients table
    print(f"R-squared: {m.rsquared:.3f}  | N: {int(m.nobs)}")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(p[["urban_pop_pct","female_tert_educ_pct","log_gdp","flfp_pct","remittances_gdp_pct"]])
print("\n=== VIF (>5 = concerning, >10 = serious) ===")
for i, col in enumerate(X.columns):
    if col != "const":
        print(f"  {col}: {variance_inflation_factor(X.values, i):.1f}")
        
with open("data/processed/regression_results.txt", "w") as f:
    f.write(m4.summary().as_text())
print("\nSaved -> data/processed/regression_results.txt")        