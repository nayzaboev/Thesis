import pandas as pd
df = pd.read_csv("data/processed/master_tfr.csv")
def cv(s):
    return s.std() / s.mean()

cv_all = df.groupby("year")["tfr"].apply(cv)
cv_by_bloc = df.groupby(["bloc", "year"])["tfr"].apply(cv).unstack(0)
peaks = df.loc[df.groupby("country")["tfr"].idxmax()][["country", "year", "tfr", "subgroup"]]
peaks = peaks.sort_values(["subgroup", "country"])
print("=== CV across all 14 countries ===")
print(cv_all.round(3).loc[[2000, 2005, 2010, 2015, 2020, 2023]])
print("\n=== CV within blocs ===")
print(cv_by_bloc.round(3).loc[[2000, 2010, 2020, 2023]])
print("\n=== Peak year per country ===")
print(peaks.to_string(index=False))
cv_all.to_csv("data/processed/cv_all.csv")
peaks.to_csv("data/processed/peaks.csv", index=False)