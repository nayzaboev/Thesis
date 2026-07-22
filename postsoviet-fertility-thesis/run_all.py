"""
run_all.py
----------
Run the full analysis pipeline in the definitive execution order, stopping at
the first script that fails.

Excludes:
  - analysis/10_fetch_wdi.py — opt-in maintenance script that refreshes the
    frozen World Bank snapshots; see README "Reproducibility Notes". Not part
    of the default run since the committed analysis uses the frozen files.
  - analysis/compute_*_smam.py — require licensed microdata not present in
    this repository, and mutate data/manual/cultural_vars.csv in place.

Run from repo root:  python run_all.py
"""

import subprocess
import sys

SCRIPTS = [
    "01_clean_data.py",
    "02_trend_plots.py",
    "03_convergence.py",
    "11_coverage_map.py",
    "12_build_panel.py",
    "13_build_crosssection.py",
    "14_descriptives.py",
    "15_layerA_gap.py",
    "16_layerB_within.py",
    "17_diagnostics.py",
    "18_crosssection.py",
    "19_robustness.py",
    "20_results_table.py",
]

for name in SCRIPTS:
    path = f"analysis/{name}"
    print(f"\n{'=' * 70}\nRunning {path}\n{'=' * 70}")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\nPIPELINE FAILED at {path} (exit code {result.returncode})")
        sys.exit(result.returncode)

print("\nAll scripts completed successfully.")
