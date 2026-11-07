"""Objective breadth versus search mechanism.

Part 1 (objective swap with the search fixed). RNAinverse's Hamming local search
is kept and its objective is changed from MFE matching to ensemble defect (ELS,
three replicates on 37 Eterna100 puzzles). Both arms use the same definition of
disagreement: the sample standard deviation (ddof=1) of target MCC across
ViennaRNA, EternaFold and CONTRAfold, on the same 37 puzzles. The paired
difference is RNAinverse minus ELS of the per-puzzle median, bootstrapped over
puzzles.

Part 2 (2 x 2 at matched wall-clock). Local search and PPO, each with the MFE and
the ensemble-defect objective, on the same targets and seeds. Convergence is
ViennaRNA ensemble defect < 0.05 at the 300 s checkpoint; the paired contrast is
the per-target mean ensemble defect under the MFE objective minus that under the
ensemble-defect objective.

Outputs: ``outputs/objective_swap_els_per_target.csv``,
``outputs/objective_swap_2x2.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from accord import stats
from accord.metrics import design_table
from common import N_BOOT, PANEL_3, SEED, load, report, save

ELS_SEED = 20260422
CHECKPOINT_S = 300.0
CONVERGED = 0.05


def els_part(cohorts: pd.DataFrame) -> dict[str, float]:
    els = load("designs/objective_swap_els.csv.gz")
    els["d"] = design_table(els, PANEL_3)["d"]
    targets = sorted(int(t) for t in els["target_id"].unique())
    rnainv = cohorts[cohorts["cohort"] == "rnainverse_eterna100"].copy()
    rnainv = rnainv[rnainv["target_id"].astype(int).isin(targets)]
    rnainv["d"] = design_table(rnainv, PANEL_3)["d"]
    per_els = els.groupby(els["target_id"].astype(int))["d"].median()
    per_rnainv = rnainv.groupby(rnainv["target_id"].astype(int))["d"].median()
    table = pd.DataFrame({"els_median_d": per_els, "rnainverse_median_d": per_rnainv,
                          "els_converged_fraction": els.groupby(els["target_id"].astype(int))[
                              "final_ensemble_defect"].apply(lambda s: (s < CONVERGED).mean())})
    save(table.reset_index(names="target_id"), "objective_swap_els_per_target.csv")
    r = stats.paired_target_bootstrap(per_rnainv.to_dict(), per_els.to_dict(),
                                      n_boot=N_BOOT, seed=ELS_SEED)
    return {"els.n_targets": r["n_targets"],
            "els.median_d_rnainverse": float(np.median(rnainv["d"])),
            "els.median_d_els": float(np.median(els["d"])),
            "els.paired_diff_mean": r["mean_diff"],
            "els.paired_diff_lo": r["mean_ci"][0],
            "els.paired_diff_hi": r["mean_ci"][1]}


def matched_part() -> dict[str, float]:
    m = load("designs/objective_swap_matched.csv.gz")
    m = m[(m["checkpoint_s"] == CHECKPOINT_S) & (m["status"] == "ok")].copy()
    m["target_id"] = m["target_id"].astype(int)     # Eterna100 puzzle numbers
    rows, res = [], {}
    for cell, g in m.groupby("cell"):
        conv = float((g["ensemble_defect"] < CONVERGED).mean())
        exact = float((g["mcc"] >= 1.0 - 1e-12).mean())
        rows.append({"cell": cell, "mechanism": g["mechanism"].iloc[0],
                     "objective": g["objective"].iloc[0], "n_runs": len(g),
                     "n_targets": g["target_id"].nunique(),
                     "median_objective_calls": float(g["n_objective_calls"].median()),
                     "converged": conv, "exact_mfe_match": exact,
                     "mean_ensemble_defect": float(g["ensemble_defect"].mean())})
        res[f"matched.{cell}.converged"] = conv
        res[f"matched.{cell}.exact_mfe_match"] = exact
    save(pd.DataFrame(rows), "objective_swap_2x2.csv")
    for mech, (mfe_cell, ed_cell) in {"LS": ("LS-MFE", "LS-ED"), "PPO": ("PPO-MFE", "PPO-ED")}.items():
        a = m[m["cell"] == mfe_cell].groupby("target_id")["ensemble_defect"].mean().to_dict()
        b = m[m["cell"] == ed_cell].groupby("target_id")["ensemble_defect"].mean().to_dict()
        r = stats.paired_target_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
        res[f"matched.{mech}.paired_ed_mfe_minus_ed.diff"] = r["mean_diff"]
        res[f"matched.{mech}.paired_ed_mfe_minus_ed.lo"] = r["mean_ci"][0]
        res[f"matched.{mech}.paired_ed_mfe_minus_ed.hi"] = r["mean_ci"][1]
    return res


def run() -> dict[str, float]:
    cohorts = load("designs/cohorts_2d.csv.gz")
    res = {**els_part(cohorts), **matched_part()}
    return {f"objective_swap.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("objective_swap", run())
