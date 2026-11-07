"""Command-line interface: ``accord oracles``, ``accord evaluate``, ``accord summarize``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Optional, Sequence

import pandas as pd

from . import __version__
from . import oracles as O
from .metrics import THRESHOLD
from .protocol import evaluate, report_from_scores


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--panel", nargs="+", default=list(O.PANEL_4), choices=O.SUPPORTED,
                   help="fixed common panel (default: %(default)s)")
    p.add_argument("--design-oracle", choices=O.SUPPORTED, default=None,
                   help="oracle used inside the design objective; adds the "
                        "leave-design-oracle-out view")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="pass threshold and gate (default: %(default)s)")
    p.add_argument("--weighting", choices=("target", "slot"), default="target",
                   help="average over targets with equal weight, or pool slots")
    p.add_argument("--out-dir", required=True, help="directory for per_design.csv and report.json")


def _print_summary(summary: dict) -> None:
    keys = ["n_slots", "n_targets", "C_complete", "C_any", "Y_all", "Q_mean", "Q_worst",
            "D_raw", "D_cond", "oracle_dependent_conditional", "oracle_dependent_raw"]
    for view, block in summary["views"].items():
        print(f"[{view}] panel = {', '.join(block['panel'])}")
        for k in keys:
            v = block.get(k)
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
    tiers = summary.get("tiers", {})
    if tiers.get("counts"):
        print("[tiers] " + ", ".join(f"{k}: {v}" for k, v in tiers["counts"].items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the ``accord`` command."""
    ap = argparse.ArgumentParser(prog="accord", description=__doc__)
    ap.add_argument("--version", action="version", version=f"accord {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("oracles", help="report which oracle backends are available")

    ev = sub.add_parser("evaluate", help="fold and score designs, then report")
    ev.add_argument("designs", help="CSV with target_id, sequence, target_structure "
                                    "(optional design_id); one row per requested slot")
    ev.add_argument("--reference", default=None,
                    help="optional CSV of reference sequences (same columns) for the "
                         "designed-to-reference disagreement ratio")
    ev.add_argument("--nupack-config", default="canonical", choices=sorted(O.nupack_configs()))
    ev.add_argument("--jobs", type=int, default=1, help="worker processes")
    _add_common(ev)

    sm = sub.add_parser("summarize", help="report from precomputed mcc_<oracle> columns")
    sm.add_argument("scores", help="CSV with target_id and mcc_<oracle> columns; "
                                   "NaN marks a failed slot")
    _add_common(sm)

    args = ap.parse_args(argv)
    if args.command == "oracles":
        print(json.dumps(O.availability(), indent=2))
        return 0
    read = dict(dtype={"target_id": str, "sequence": str, "target_structure": str},
                keep_default_na=False, na_values={"sequence": [""], "target_id": [""]})
    try:
        if args.command == "evaluate":
            designs = pd.read_csv(args.designs, **read)
            reference = pd.read_csv(args.reference, **read) if args.reference else None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="oracles not available")
                report = evaluate(designs, args.panel, args.design_oracle, args.threshold,
                                  args.nupack_config, args.weighting, reference, args.jobs)
            down = [o for o, s in O.availability().items()
                    if o in set(args.panel) | set(O.TIER_PANEL) and not s["available"]]
            if down:
                print(f"warning: unavailable oracles {down}; their slots are counted as "
                      "incomplete (run `accord oracles` for details)", file=sys.stderr)
        else:
            scores = pd.read_csv(args.scores, dtype={"target_id": str})
            report = report_from_scores(scores, args.panel, args.design_oracle, args.threshold,
                                        args.weighting)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report.save(args.out_dir)
    _print_summary(report.summary)
    print(f"wrote {args.out_dir}/per_design.csv and {args.out_dir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
