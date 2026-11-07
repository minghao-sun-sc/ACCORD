"""Sensitivity of the headline results to the NUPACK invocation.

NUPACK is both a design method and an oracle, so its column is recomputed under
five declared configurations (``accord/configs/nupack.json``) while the other
three oracles are unchanged. For each configuration:

* native-sequence NUPACK accuracy on the 95 Das natives;
* RNAinverse oracle dependence on Eterna100 (conditional on any pass);
* NUPACK's raw all-oracle yield on its full Eterna100 cohort;
* the NUPACK minus RNAinverse Y_all gap on the 57 common targets, with a paired
  target-level bootstrap interval;
* Kendall's tau between each configuration's full-cohort Y_all ordering of the
  nine method cohorts and the canonical ordering.

Output: ``outputs/nupack_sensitivity.csv``.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau

import accord
from accord import stats
from common import N_BOOT, PANEL_4, SEED, load, report, save

#: Method cohorts used for the ordering check (the leaky original gRNAde
#: release, the human Eterna solutions and the RhoDesign held-out split are
#: reported elsewhere).
NINE = ["rnainverse_eterna100", "nupack_eterna100", "samfeo_eterna100", "learna_eterna100",
        "metalearna_eterna100", "samplingdesign_eterna100", "grnade_retrain_das",
        "ribodiffusion_das", "rdesign_das"]
CONFIGS = ["canonical", "rna99", "wc_only", "celsius_25", "mg_0p01"]


def with_config(table: pd.DataFrame, sweep: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Replace the NUPACK column of ``table`` by the swept values."""
    out = table.drop(columns="mcc_nupack").merge(sweep[keys + ["mcc_nupack"]], on=keys,
                                                 how="left", validate="1:1")
    assert len(out) == len(table)
    return out


def run() -> dict[str, float]:
    cohorts = load("designs/cohorts_2d.csv.gz")
    crossed = load("designs/crossed_57.csv.gz")
    sweep = load("designs/nupack_sweep.csv.gz")
    rows, res, yields = [], {}, {}
    for cfg in CONFIGS:
        s = sweep[sweep["config"] == cfg]
        nat = s[s["pool"] == "natives_das"]["mcc_nupack"].mean()
        coh = with_config(cohorts, s[s["pool"] == "cohort"], ["cohort", "target_id", "slot"])
        dep = accord.summarize(coh[coh["cohort"] == "rnainverse_eterna100"], PANEL_4,
                               target_col=None)["oracle_dependent_conditional"]
        yields[cfg] = {c: accord.summarize(g, PANEL_4, target_col=None)["Y_all"]
                       for c, g in coh.groupby("cohort")}
        cr = with_config(crossed, s[s["pool"] == "crossed_57"], ["method", "target_id", "slot"])
        per = {m: accord.metrics.design_table(g, PANEL_4).groupby("target_id")["all_pass"].mean().to_dict()
               for m, g in cr.groupby("method")}
        gap = stats.paired_target_bootstrap(per["NUPACK"], per["RNAinverse"], n_boot=N_BOOT, seed=SEED)
        tau = kendalltau([yields["canonical"][c] for c in NINE], [yields[cfg][c] for c in NINE])[0]
        row = {"config": cfg, "native_mcc_nupack": nat, "rnainverse_dependent_conditional": dep,
               "nupack_cohort_Y_all": yields[cfg]["nupack_eterna100"],
               "crossed_gap_Y_all": gap["mean_diff"], "crossed_gap_lo": gap["mean_ci"][0],
               "crossed_gap_hi": gap["mean_ci"][1], "kendall_tau_Y_all_nine": tau}
        rows.append(row)
        res.update({f"{cfg}.{k}": v for k, v in row.items() if k != "config"})
    save(pd.DataFrame(rows), "nupack_sensitivity.csv")
    return {f"nupack_sensitivity.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("nupack_sensitivity", run())
