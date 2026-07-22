"""
build_muslim_share.py
---------------------
Rebuild the muslim_share column in data/manual/cultural_vars.csv from ONE
official Pew Research Center release, so the provenance is fully auditable and
reproducible (fixes Critical Issue 2, incl. the Azerbaijan=Uzbekistan
duplication).

Source:
  Pew Research Center, "Religious Composition by Country, 2010-2020"
  file: Religious Composition 2010-2020 (percentages).csv  (Muslims column, Year=2020)
  Report: "How the Global Religious Landscape Changed From 2010 to 2020"
  Per Pew's README, the unrounded percentages worksheet is the one appropriate
  for statistical/regression use, so full precision is retained here.

Usage:
  Place the Pew percentages CSV at the path in PEW_CSV below (or pass as argv[1]),
  then run from repo root:
      python analysis/build_muslim_share.py
  It rewrites data/manual/cultural_vars.csv in place (muslim_share, _year,
  _source) and prints an old-vs-new diff. Run validate_cultural_vars.py after.
"""
import sys
import pandas as pd

PEW_CSV = (sys.argv[1] if len(sys.argv) > 1 else
           "data/raw/Religious Composition 2010-2020 (percentages).csv")
PEW_YEAR = 2020
SOURCE_LABEL = "Pew Research Center, Religious Composition by Country 2010-2020 (2020, percentages)"

COUNTRIES = ["Armenia","Azerbaijan","Belarus","Estonia","Georgia","Kazakhstan",
             "Kyrgyzstan","Latvia","Lithuania","Moldova","Russia","Tajikistan",
             "Ukraine","Uzbekistan"]

# Pew uses these exact country names for the 14 study countries (verified).
# If Pew renames a country in a future release, add the mapping here.
PEW_NAME = {c: c for c in COUNTRIES}

def main():
    pew = pd.read_csv(PEW_CSV)
    pew.columns = [c.strip() for c in pew.columns]
    need = {"Country", "Year", "Muslims"}
    missing = need - set(pew.columns)
    if missing:
        sys.exit(f"Pew file missing expected columns: {missing}")

    sub = pew[pew["Year"] == PEW_YEAR]
    shares = {}
    for c in COUNTRIES:
        row = sub[sub["Country"] == PEW_NAME[c]]
        if len(row) != 1:
            sys.exit(f"Expected exactly 1 Pew row for {c} in {PEW_YEAR}, got {len(row)}")
        shares[c] = float(row["Muslims"].iloc[0])

    cv = pd.read_csv("data/manual/cultural_vars.csv")
    old = cv.set_index("country")["muslim_share"].to_dict()

    cv["muslim_share"] = cv["country"].map(shares)
    cv["muslim_share_year"] = PEW_YEAR
    cv["muslim_share_source"] = SOURCE_LABEL

    assert cv["muslim_share"].notna().all(), "Some countries did not receive a Pew value."
    cv.to_csv("data/manual/cultural_vars.csv", index=False)

    print("Rebuilt muslim_share from Pew 2020 percentages. Old -> New:\n")
    print(f"  {'country':12s} {'old':>8s} {'new':>12s}")
    for c in COUNTRIES:
        print(f"  {c:12s} {old[c]:8.2f} {shares[c]:12.6f}")
    print(f"\nWrote data/manual/cultural_vars.csv. "
          f"Source: {SOURCE_LABEL}")

if __name__ == "__main__":
    main()