import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

p = pd.read_csv("data/processed/panel.csv")
p["log_gdp"] = np.log(p["gdp_per_capita_usd"])
p["ca"] = (p["bloc"] == "Central Asia").astype(int)   # 1 if Central Asia, else 0
m_urban = smf.ols(
    "tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct + remittances_gdp_pct"
    " + ca + urban_pop_pct:ca",
    data=p
).fit(cov_type="cluster", cov_kwds={"groups": p["country"]})

print("=== Interaction: urbanisation × Central Asia ===")
print(m_urban.summary().tables[1])
m_flfp = smf.ols(
    "tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct + remittances_gdp_pct"
    " + ca + flfp_pct:ca",
    data=p
).fit(cov_type="cluster", cov_kwds={"groups": p["country"]})

print("\n=== Interaction: FLFP × Central Asia ===")
print(m_flfp.summary().tables[1])
# Step 3b — Interaction 3: GDP × Central Asia (targets the positive-GDP puzzle)
m_gdp = smf.ols(
    "tfr ~ urban_pop_pct + female_tert_educ_pct + log_gdp + flfp_pct + remittances_gdp_pct"
    " + ca + log_gdp:ca",
    data=p
).fit(cov_type="cluster", cov_kwds={"groups": p["country"]})

print("\n=== Interaction: GDP × Central Asia ===")
print(m_gdp.summary().tables[1])
with open("data/processed/interaction_results.txt", "w") as f:
    f.write("URBANISATION × CENTRAL ASIA:\n")
    f.write(m_urban.summary().as_text())
    f.write("\n\nFLFP × CENTRAL ASIA:\n")
    f.write(m_flfp.summary().as_text())
    f.write("\n\nGDP × CENTRAL ASIA:\n")
    f.write(m_gdp.summary().as_text())
print("\nSaved -> data/processed/interaction_results.txt")