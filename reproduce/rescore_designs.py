"""Refold released designs with the installed oracles and compare with the released MCC.

This closes the loop from sequence to score: every MCC in ``data/designs`` was
produced by the same oracle invocations that ``accord.score`` runs. The check needs
the oracle backends (see the README); oracles that are not installed are skipped.

Usage::

    python reproduce/rescore_designs.py --table crossed_57 --n 200 --jobs 8
    python reproduce/rescore_designs.py --table cohorts_2d --n 0     # every design

Prints, per oracle, the fraction of designs whose recomputed MCC equals the
released value (to 1e-9), and writes mismatches to
``reproduce/outputs/rescore_<table>_mismatches.csv``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import accord
from accord import oracles as O
from common import PANEL_4, load, save

TABLES = {"crossed_57": "designs/crossed_57.csv.gz",
          "strict_20": "designs/strict_20.csv.gz",
          "cohorts_2d": "designs/cohorts_2d.csv.gz",
          "natives_das": "designs/natives_das.csv.gz"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", choices=sorted(TABLES), default="crossed_57")
    ap.add_argument("--n", type=int, default=200, help="random sample size; 0 for all rows")
    ap.add_argument("--max-length", type=int, default=0, help="skip longer designs (0: no limit)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = load(TABLES[args.table]).dropna(subset=["sequence"])
    if args.max_length:
        df = df[df["sequence"].str.len() <= args.max_length]
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    status = O.availability()
    available = [o for o in PANEL_4 if status[o]["available"]]
    for o in sorted(set(PANEL_4) - set(available)):
        print(f"skipping {o}: {status[o]['detail']}")
    if not available:
        print("no oracle backend is available; nothing to check")
        return 2
    released = df[[f"mcc_{o}" for o in available]].copy()
    rescored = accord.score(df.drop(columns=[f"mcc_{o}" for o in PANEL_4 if f"mcc_{o}" in df]),
                            oracles=available, n_jobs=args.jobs)
    bad = []
    print(f"{args.table}: {len(df)} designs")
    for o in available:
        a = released[f"mcc_{o}"].to_numpy(float)
        b = rescored[f"mcc_{o}"].to_numpy(float)
        same = np.isclose(a, b, atol=1e-9, rtol=0) | (np.isnan(a) & np.isnan(b))
        print(f"  {o:<11} {same.mean():7.2%} identical   max |diff| {np.nanmax(np.abs(a - b)):.3g}")
        if (~same).any():
            m = rescored.loc[~same, ["target_id", "sequence", f"structure_{o}"]].copy()
            m["oracle"], m["released_mcc"], m["rescored_mcc"] = o, a[~same], b[~same]
            bad.append(m.rename(columns={f"structure_{o}": "rescored_structure"}))
    if bad:
        import pandas as pd

        path = save(pd.concat(bad, ignore_index=True), f"rescore_{args.table}_mismatches.csv")
        print(f"mismatches written to {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
