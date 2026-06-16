import pandas as pd
df = pd.read_csv("data/raw/unpopulation_dataportal_20260616151558.csv")
df = df[df["IndicatorName"] == "Total fertility rate"]
df = df[df["Variant"] == "Median"]
df = df[df["Sex"] == "Both sexes"]
df = df[["Location", "Time", "Value"]].copy()
df.columns = ["country", "year", "tfr"]
df["country"] = df["country"].replace({
    "Russian Federation": "Russia",
    "Republic of Moldova": "Moldova",
})
central_asia = ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan"]
slavic = ["Russia", "Ukraine", "Belarus", "Moldova"]
baltic = ["Estonia", "Latvia", "Lithuania"]
caucasus = ["Armenia", "Georgia", "Azerbaijan"]

def subgroup(c):
    if c in central_asia: return "Central Asia"
    if c in slavic: return "Slavic"
    if c in baltic: return "Baltic"
    if c in caucasus: return "Caucasus"
    return "Unassigned"

df["subgroup"] = df["country"].apply(subgroup)
df["bloc"] = df["subgroup"].apply(
    lambda s: "Central Asia" if s == "Central Asia" else "Rest of post-Soviet"
)
df = df.sort_values(["subgroup", "country", "year"])
df.to_csv("data/processed/master_tfr.csv", index=False)
print("Saved. Rows:", len(df), "Countries:", df["country"].nunique())