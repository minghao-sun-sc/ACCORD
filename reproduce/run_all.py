"""Run every reproduction script and compare with the values reported in the paper.

Usage::

    python reproduce/run_all.py            # all analyses
    python reproduce/run_all.py crossed_benchmark objective_swap

Writes per-analysis tables to ``reproduce/outputs/`` and a combined comparison to
``reproduce/outputs/summary.csv``. Exits with status 1 if any reported value is
not reproduced.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import compare, save, verify_data  # noqa: E402

ANALYSES = [
    "single_oracle_failure",
    "objective_swap",
    "crossed_benchmark",
    "strict_sensitivity",
    "nupack_sensitivity",
    "panel_structure",
    "shape_calibration",
    "structure_3d",
]


def main(argv: list[str]) -> int:
    names = argv or ANALYSES
    unknown = [n for n in names if n not in ANALYSES]
    if unknown:
        print(f"unknown analyses {unknown}; choose from: {', '.join(ANALYSES)}")
        return 2
    changed = verify_data()
    if changed:
        print(f"warning: tables differ from data/MANIFEST.tsv: {', '.join(changed)}")
    else:
        print("data/: every table matches its checksum in data/MANIFEST.tsv\n")
    tables = []
    for name in names:
        t0 = time.time()
        results = importlib.import_module(name).run()
        table = compare(results)
        table.insert(0, "analysis", name)
        tables.append(table)
        n_ok = int(table["match"].sum())
        print(f"{name:<24} {n_ok:>4}/{len(table):<4} reproduced   ({time.time() - t0:.0f} s)")
        for r in table[~table["match"]].itertuples():
            print(f"    DIFF {r.result}: reported {r.reported}, reproduced {r.reproduced:.6g}")
    summary = pd.concat(tables, ignore_index=True)
    save(summary, "summary.csv")
    n_bad = int((~summary["match"]).sum())
    print(f"\n{len(summary) - n_bad}/{len(summary)} reported values reproduced; "
          f"details in reproduce/outputs/summary.csv")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
