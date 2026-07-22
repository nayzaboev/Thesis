"""
validate_cultural_vars.py
--------------------------
Sanity-check data/manual/cultural_vars.csv before it feeds the cross-section
build. Exits 1 (loudly) on any of:

  1. Country set is not exactly the 14 study countries.
  2. smam_female, muslim_share, or female_mean_schooling has a missing value.
  3. Two countries share an identical muslim_share above 1.0 (the
     Azerbaijan/Uzbekistan copy-paste bug this script guards against; values
     <=1.0 are exempt since near-zero shares can legitimately coincide).
  4. muslim_share falls outside [0, 100].

Usage:
  python analysis/validate_cultural_vars.py [path/to/cultural_vars.csv]
"""
import sys
import pandas as pd

CV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/manual/cultural_vars.csv"

EXPECTED_COUNTRIES = {
    "Armenia", "Azerbaijan", "Belarus", "Estonia", "Georgia", "Kazakhstan",
    "Kyrgyzstan", "Latvia", "Lithuania", "Moldova", "Russia", "Tajikistan",
    "Ukraine", "Uzbekistan",
}
ANALYTICAL_COLS = ["smam_female", "muslim_share", "female_mean_schooling"]


def main():
    cv = pd.read_csv(CV_PATH)
    errors = []

    countries = set(cv["country"])
    if countries != EXPECTED_COUNTRIES:
        errors.append(
            "Country set is not exactly the 14 study countries.\n"
            f"  Missing: {EXPECTED_COUNTRIES - countries}\n"
            f"  Unexpected: {countries - EXPECTED_COUNTRIES}"
        )

    missing = cv[ANALYTICAL_COLS].isna()
    if missing.any().any():
        counts = missing.sum()
        errors.append(
            "Missing values in analytical columns:\n"
            f"{counts[counts > 0].to_string()}"
        )

    dup_check = cv[cv["muslim_share"] > 1.0]
    dup_counts = dup_check["muslim_share"].value_counts()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        offending = cv[cv["muslim_share"].isin(dups.index)][["country", "muslim_share"]]
        errors.append(
            "Duplicate muslim_share values (>1.0) across countries:\n"
            f"{offending.to_string(index=False)}"
        )

    out_of_range = cv[(cv["muslim_share"] < 0) | (cv["muslim_share"] > 100)]
    if len(out_of_range) > 0:
        errors.append(
            "muslim_share outside [0, 100]:\n"
            f"{out_of_range[['country', 'muslim_share']].to_string(index=False)}"
        )

    if errors:
        print("VALIDATION FAILED:\n")
        for e in errors:
            print(e + "\n")
        sys.exit(1)

    print(f"OK: {CV_PATH} passed all checks ({len(cv)} countries).")


if __name__ == "__main__":
    main()
