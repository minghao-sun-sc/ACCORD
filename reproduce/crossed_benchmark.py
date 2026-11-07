"""Candidate-matched comparison on 57 common Das targets.

Every method contributes K = 5 requested candidates per target, selected only by
its own design objective, and all candidates are scored by the same panel. Targets
are weighted equally and intervals resample targets (5,000 draws). The analysis is
repeated without NUPACK in the panel, because NUPACK is also one of the design
methods.

Outputs: ``outputs/crossed_57_<panel>.csv`` (per-method table) and
``outputs/crossed_57_<panel>_contrasts_Y_all.csv`` (paired contrasts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import accord
from accord import stats
from common import N_BOOT, PANEL_3, PANEL_4, SEED, load, report, save

PANELS = {"4oracle": PANEL_4, "3oracle_no_nupack": PANEL_3}


def method_table(pool: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    """Target-weighted end-to-end summary with bootstrap intervals per method."""
    rows = []
    for method, g in pool.groupby("method"):
        t = accord.metrics.design_table(g, panel)
        s = accord.summarize(g, panel)
        row = {"method": method, "paradigm": g["paradigm"].iloc[0],
               "leakage_status": g["leakage_status"].iloc[0],
               **{k: s[k] for k in ("n_targets", "n_slots", "C_complete", "C_any", "Y_all",
                                    "Q_mean", "Q_worst", "D_raw")}}
        comp = t[t["complete"]]
        for name, values, targets, inner in (
                ("C_any", t["any_pass"].astype(float), t["target_id"], np.mean),
                ("Y_all", t["all_pass"].astype(float), t["target_id"], np.mean),
                ("Q_mean", comp["q_mean"], comp["target_id"], np.mean),
                ("Q_worst", comp["q_worst"], comp["target_id"], np.mean),
                ("D_raw", comp["d"], comp["target_id"], np.median)):
            b = stats.target_weighted_bootstrap(values, targets, inner=inner,
                                                n_boot=N_BOOT, seed=SEED)
            row[f"{name}_lo"], row[f"{name}_hi"] = b["lo"], b["hi"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Y_all", ascending=False)


def per_target_yield(pool: pd.DataFrame, panel: list[str]) -> dict[str, dict]:
    """Per-target Y_all for every method."""
    out = {}
    for method, g in pool.groupby("method"):
        t = accord.metrics.design_table(g, panel)
        out[method] = t.groupby("target_id")["all_pass"].mean().to_dict()
    return out


def contrasts(pool: pd.DataFrame, panel: list[str], reference: str = "RNAinverse") -> pd.DataFrame:
    """Paired target-level bootstrap of Y_all: every method minus the reference,
    plus the two ensemble-physics samplers against each other."""
    per = per_target_yield(pool, panel)
    pairs = [(m, reference) for m in sorted(per) if m != reference]
    pairs.append(("SamplingDesign", "SAMFEO"))
    rows = []
    for a, b in pairs:
        r = stats.paired_target_bootstrap(per[a], per[b], n_boot=N_BOOT, seed=SEED)
        rows.append({"method_a": a, "method_b": b, "n_targets": r["n_targets"],
                     "mean_diff": r["mean_diff"], "lo": r["mean_ci"][0], "hi": r["mean_ci"][1],
                     "wins": r["wins"], "ties": r["ties"], "losses": r["losses"],
                     "wilcoxon_p": r["wilcoxon_p"]})
    return pd.DataFrame(rows)


def run() -> dict[str, float]:
    pool = load("designs/crossed_57.csv.gz")
    res: dict[str, float] = {}
    for label, panel in PANELS.items():
        table = method_table(pool, panel)
        save(table, f"crossed_57_{label}.csv")
        con = contrasts(pool, panel)
        save(con, f"crossed_57_{label}_contrasts_Y_all.csv")
        for r in table.itertuples():
            for k in ("C_any", "Y_all", "Q_mean", "Q_worst", "D_raw"):
                res[f"{label}.{r.method}.{k}"] = getattr(r, k)
        for r in con.itertuples():
            key = f"{label}.contrast_Y_all.{r.method_a}-{r.method_b}"
            res[f"{key}.diff"], res[f"{key}.lo"], res[f"{key}.hi"] = r.mean_diff, r.lo, r.hi
            res[f"{key}.p"] = r.wilcoxon_p
        res[f"{label}.Y_all_lo.NUPACK"] = table.set_index("method").loc["NUPACK", "Y_all_lo"]
        res[f"{label}.Y_all_hi.NUPACK"] = table.set_index("method").loc["NUPACK", "Y_all_hi"]
    return {f"crossed_benchmark.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("crossed_benchmark", run())
