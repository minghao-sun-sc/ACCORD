"""Single-oracle evaluation failure and the end-to-end report for every cohort.

Reproduces
* the RNAinverse outcome counts on Eterna100 under the four-oracle panel and the
  oracle-dependent fraction under both denominators (conditional on passing at
  least one oracle, and on the raw pool);
* native-sequence accuracy of each oracle on the Das targets, the native
  disagreement baseline and the native oracle-dependent fraction;
* NUPACK design disagreement with and without its own design oracle;
* the descriptive paradigm table pooled over designs (any-pass count, any-pass
  rate, oracle dependence among any-pass designs and on the raw pool), written to
  ``outputs/paradigm_table_pooled.csv``;
* an end-to-end table (coverage, yield, quality, disagreement) for every cohort
  with equal target weight, written to ``outputs/end_to_end_cohorts.csv``.
The cohorts use different target sets, so both tables are descriptive; method
comparisons use ``crossed_benchmark.py``.
"""

from __future__ import annotations

import pandas as pd

import accord
from common import PANEL_3, PANEL_4, load, report, save

#: Design cohorts of the descriptive paradigm table, in table order.
PARADIGM_ROWS = ["nupack_eterna100", "samplingdesign_eterna100", "samfeo_eterna100",
                 "learna_eterna100", "metalearna_eterna100", "grnade_das",
                 "ribodiffusion_das", "rhodesign_heldout44", "rnainverse_eterna100"]


def run() -> dict[str, float]:
    cohorts = load("designs/cohorts_2d.csv.gz")
    natives = load("designs/natives_das.csv.gz")
    res: dict[str, float] = {}

    rnainv = cohorts[cohorts["cohort"] == "rnainverse_eterna100"]
    s = accord.summarize(rnainv, PANEL_4, target_col=None)
    res.update({
        "rnainverse_eterna100.n_slots": s["n_slots"],
        "rnainverse_eterna100.n_pass_all": s["n_pass_all"],
        "rnainverse_eterna100.n_pass_some_not_all": s["n_pass_some_not_all"],
        "rnainverse_eterna100.n_fail_all": s["n_fail_all"],
        "rnainverse_eterna100.oracle_dependent_conditional": s["oracle_dependent_conditional"],
        "rnainverse_eterna100.oracle_dependent_raw": s["oracle_dependent_raw"],
        "rnainverse_eterna100.pass_all_raw": s["n_pass_all"] / s["n_slots"],
    })

    for o in PANEL_4:
        res[f"natives.mean_mcc_{o}"] = natives[f"mcc_{o}"].mean()
    nat4 = accord.summarize(natives, PANEL_4, target_col=None)
    nat3 = accord.summarize(natives, PANEL_3, target_col=None)
    res["natives.median_disagreement_4oracle"] = nat4["D_raw"]
    res["natives.median_disagreement_3oracle"] = nat3["D_raw"]
    res["natives.oracle_dependent_conditional"] = nat4["oracle_dependent_conditional"]

    npk = cohorts[cohorts["cohort"] == "nupack_eterna100"]
    res["nupack_eterna100.median_disagreement_4oracle"] = accord.summarize(
        npk, PANEL_4, target_col=None)["D_raw"]
    res["nupack_eterna100.median_disagreement_without_nupack"] = accord.summarize(
        npk, PANEL_3, target_col=None)["D_raw"]
    res["nupack_eterna100.oracle_dependent_conditional"] = accord.summarize(
        npk, PANEL_4, target_col=None)["oracle_dependent_conditional"]

    pooled = []
    for name, g in [(c, cohorts[cohorts["cohort"] == c]) for c in PARADIGM_ROWS] + [("natives_das", natives)]:
        s = accord.summarize(g, PANEL_4, target_col=None)
        n_any = s["n_pass_all"] + s["n_pass_some_not_all"]
        row = {"cohort": name, "n_designs": s["n_slots"], "n_any_pass": n_any,
               "any_pass_rate": n_any / s["n_slots"],
               "oracle_dependent_conditional": s["oracle_dependent_conditional"],
               "pass_all_conditional": s["n_pass_all"] / n_any,
               "oracle_dependent_raw": s["oracle_dependent_raw"],
               "pass_all_raw": s["n_pass_all"] / s["n_slots"]}
        pooled.append(row)
        for k in ("n_any_pass", "any_pass_rate", "oracle_dependent_conditional",
                  "oracle_dependent_raw", "pass_all_raw"):
            res[f"paradigm_table.{name}.{k}"] = row[k]
    save(pd.DataFrame(pooled), "paradigm_table_pooled.csv")

    rows = []
    for (cohort, method, tset, leak), g in cohorts.groupby(
            ["cohort", "method", "target_set", "leakage_status"], sort=False):
        s4 = accord.summarize(g, PANEL_4, target_col="target_id")
        rows.append({"cohort": cohort, "method": method, "target_set": tset,
                     "leakage_status": leak, **{k: s4[k] for k in (
                         "n_targets", "n_slots", "C_complete", "C_any", "Y_all", "Q_mean",
                         "Q_worst", "D_raw", "D_cond", "oracle_dependent_conditional",
                         "oracle_dependent_raw")}})
    save(pd.DataFrame(rows), "end_to_end_cohorts.csv")
    return {f"single_oracle_failure.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("single_oracle_failure", run())
