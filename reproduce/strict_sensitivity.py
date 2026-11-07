"""Strict sensitivity: every submitted method on 20 shared targets.

The 57 common targets are intersected with RhoDesign's leakage-free held-out set,
which leaves 20 targets on which all nine submitted methods (with the clean gRNAde
retrain) plus RDesign can be compared with K = 5 slots each. Targets in this
subset are longer than the excluded ones, so it is a secondary analysis.

Outputs: ``outputs/strict_20_<panel>.csv`` and
``outputs/strict_20_<panel>_contrasts_Y_all.csv``.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as sps

from accord import stats
from common import N_BOOT, PANEL_3, PANEL_4, SEED, load, report, save
from crossed_benchmark import method_table, per_target_yield

PANELS = {"4oracle": PANEL_4, "3oracle_no_nupack": PANEL_3}


def run() -> dict[str, float]:
    pool = load("designs/strict_20.csv.gz")
    crossed = load("designs/crossed_57.csv.gz")
    res: dict[str, float] = {}
    for label, panel in PANELS.items():
        table = method_table(pool, panel)
        save(table, f"strict_20_{label}.csv")
        per = per_target_yield(pool, panel)
        rows = []
        for m in sorted(per):
            if m == "RNAinverse":
                continue
            r = stats.paired_target_bootstrap(per[m], per["RNAinverse"], n_boot=N_BOOT, seed=SEED)
            rows.append({"method": m, "minus": "RNAinverse", "diff": r["mean_diff"],
                         "lo": r["mean_ci"][0], "hi": r["mean_ci"][1], "wilcoxon_p": r["wilcoxon_p"]})
            key = f"{label}.contrast_Y_all.{m}-RNAinverse"
            res[f"{key}.diff"], res[f"{key}.lo"], res[f"{key}.hi"] = (
                r["mean_diff"], r["mean_ci"][0], r["mean_ci"][1])
        save(pd.DataFrame(rows), f"strict_20_{label}_contrasts_Y_all.csv")
        for r in table.itertuples():
            res[f"{label}.{r.method}.C_any"] = r.C_any
            res[f"{label}.{r.method}.Y_all"] = r.Y_all

    # Ordering concordance with the 57-target comparison on shared methods.
    a = method_table(crossed, PANEL_4).set_index("method")["Y_all"]
    b = method_table(pool, PANEL_4).set_index("method")["Y_all"]
    shared = sorted(set(a.index) & set(b.index))
    res["concordance_with_crossed_57.spearman_Y_all"] = sps.spearmanr(a[shared], b[shared])[0]
    res["concordance_with_crossed_57.n_methods"] = len(shared)
    return {f"strict_sensitivity.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("strict_sensitivity", run())
