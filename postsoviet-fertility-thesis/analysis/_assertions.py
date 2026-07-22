"""
_assertions.py
--------------
Shared data-integrity helper used across the analysis pipeline.

Not a pipeline stage itself (no SCRIPTS entry in run_all.py) — imported by
scripts that sort a panel by ["country", "year"] and then call shift()/diff()
within country, both of which silently treat a multi-year gap as a one-year
change.
"""

import pandas as pd


def assert_year_continuity(df, group_col="country", year_col="year"):
    """Assert that, within every group, consecutive years (after sorting) are
    exactly one year apart. Call this immediately before a shift()/diff() on
    a frame that is expected to be gap-free at the row level (e.g. the full
    rectangular panel) — NOT on a complete-case subset where dropped rows can
    legitimately create gaps; document those sites instead of asserting."""
    d = df.sort_values([group_col, year_col])
    gaps = d.groupby(group_col)[year_col].diff().dropna()
    bad = gaps[gaps != 1]
    assert bad.empty, (
        f"Year gap(s) detected within one or more '{group_col}' groups after "
        f"sorting by ['{group_col}', '{year_col}'] — shift()/diff() would "
        f"silently treat a multi-year gap as a one-year change. "
        f"Offending row indices -> year gap:\n{bad.to_string()}"
    )
