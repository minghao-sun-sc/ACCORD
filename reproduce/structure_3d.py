"""Scope test: agreement between 2D oracles does not carry over to 3D structure.

TM-scores come from US-align (``-mol RNA``) normalised by the reference chain of each
Das/PDB target. All 3D predictors run without a multiple sequence alignment, except the
MSA-conditioned AlphaFold3 runs used only in part 3. Intervals resample targets.

1. Native panel: per-target SD (ddof = 1) of the native sequence's TM-score across
   RhoFold+, Boltz-2 and RoseTTAFold2NA, then with single-sequence AlphaFold3 added, on
   the 95 targets every member covers. DRfold2 is added as a fifth predictor on the 75
   targets it covers and compared with the four-predictor SD on those same targets.
   The three-predictor SD is also compared with the 2D disagreement (target-MCC SD)
   of the same natives.
2. Target-matched designs: RhoFold+ TM-score of up to five unique designs per method
   (first in generation order) from the original gRNAde release and RiboDiffusion on the
   36 targets where both have designs, averaged within target and paired across targets.
   The same contrast under single-sequence AlphaFold3 uses the first candidate per method
   and the leakage-clean gRNAde retrain.
3. AlphaFold3 MSA effect: paired TM difference (MSA-conditioned minus single-sequence)
   for natives whose searched alignment contains at least one homolog (depth > 1).
   Sequences with a depth-1 alignment receive the same input in both modes and serve as
   a noise floor.

Outputs: ``outputs/structure3d_panel.csv``, ``structure3d_vs_2d.csv``,
``structure3d_matched.csv``, ``structure3d_msa.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from accord import stats
from common import N_BOOT, SEED, load, report, save

THREE = ["rhofold", "boltz2", "rf2na"]
FOUR = THREE + ["af3"]
FIVE = FOUR + ["drfold2"]


def native_panel(res: dict) -> pd.DataFrame:
    """Cross-predictor TM-score SD on native sequences for each panel composition."""
    nat = load("structure3d/native_panel.csv.gz")
    wide = nat.pivot(index="target_id", columns="predictor", values="tm_score")
    rows = []
    for label, cols in (("three_predictor", THREE), ("four_predictor", FOUR),
                        ("five_predictor", FIVE)):
        sub = wide[cols].dropna()
        sd = sub.std(axis=1, ddof=1)
        b = stats.target_weighted_bootstrap(sd, sub.index, n_boot=N_BOOT, seed=SEED)
        rows.append({"panel": label, "predictors": "+".join(cols), "n_targets": len(sub),
                     "mean_sd": b["point"], "lo": b["lo"], "hi": b["hi"],
                     "median_sd": sd.median()})
        for k in ("n_targets", "mean_sd", "lo", "hi"):
            res[f"panel.{label}.{k}"] = rows[-1][k]
    same = wide[FIVE].dropna()
    four_same = same[FOUR].std(axis=1, ddof=1).mean()
    res["panel.five_predictor.four_predictor_same_targets"] = four_same
    res["panel.five_predictor.delta_same_targets"] = res["panel.five_predictor.mean_sd"] - four_same
    return pd.DataFrame(rows)


def ratio_3d_2d(res: dict) -> pd.DataFrame:
    """Ratio of 3D to 2D disagreement on the same native sequences.

    3D: TM-score SD across RhoFold+, Boltz-2 and RoseTTAFold2NA. 2D: target-MCC SD
    across the three tier-panel oracles, or across all four. Ratio of the two means,
    with a target-level bootstrap interval.
    """
    wide = load("structure3d/native_panel.csv.gz").pivot(index="target_id", columns="predictor",
                                                           values="tm_score")
    nat = load("designs/natives_das.csv.gz").set_index("target_id")
    sd3 = wide[THREE].dropna().std(axis=1, ddof=1)
    rows = []
    for label, oracles in (("vs_three_oracle_2d", ["vienna", "eternafold", "contrafold"]),
                           ("vs_four_oracle_2d", ["vienna", "nupack", "eternafold", "contrafold"])):
        idx = sd3.index.intersection(nat.index)
        a = sd3.loc[idx].to_numpy(float)
        b = nat.loc[idx, [f"mcc_{o}" for o in oracles]].std(axis=1, ddof=1).to_numpy(float)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(a), len(a))
            draws.append(a[i].mean() / b[i].mean())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        row = {"comparison": label, "n_targets": len(idx), "mean_sd_3d": a.mean(),
               "mean_sd_2d": b.mean(), "ratio": a.mean() / b.mean(), "lo": lo, "hi": hi}
        rows.append(row)
        for k in ("n_targets", "mean_sd_2d", "ratio", "lo", "hi"):
            res[f"ratio_3d_2d.{label}.{k}"] = row[k]
    return pd.DataFrame(rows)


def paired(per: pd.DataFrame, a: str, b: str) -> dict:
    return stats.paired_target_bootstrap(per[a].to_dict(), per[b].to_dict(),
                                         n_boot=N_BOOT, seed=SEED)


def matched(res: dict) -> pd.DataFrame:
    """RiboDiffusion minus gRNAde on shared targets, under RhoFold+ and AlphaFold3."""
    m = load("structure3d/matched_rhofold.csv.gz")
    per = m.groupby(["target_id", "method"])["tm_score"].mean().unstack()
    af = load("structure3d/alphafold3.csv.gz")
    ss = af[af["mode"] == "single_sequence"]
    af_per = ss.pivot(index="target_id", columns="method", values="tm_score")
    rows = []
    for label, table, ref in (("rhofold_36", per, "gRNAde"),
                              ("af3_36", af_per.reindex(per.index), "gRNAde-clean"),
                              ("af3_all", af_per, "gRNAde-clean")):
        table = table[["RiboDiffusion", ref]].dropna()
        r = paired(table, "RiboDiffusion", ref)
        rows.append({"comparison": label, "reference_method": ref, "n_targets": r["n_targets"],
                     "mean_tm_ribodiffusion": table["RiboDiffusion"].mean(),
                     "mean_tm_reference": table[ref].mean(), "diff": r["mean_diff"],
                     "lo": r["mean_ci"][0], "hi": r["mean_ci"][1], "wilcoxon_p": r["wilcoxon_p"]})
    res["matched.rhofold_36.n_designs"] = len(m)
    for r in rows[:2]:
        key = r["comparison"]
        res[f"matched.{key}.n_targets"] = r["n_targets"]
        res[f"matched.{key}.mean_tm.RiboDiffusion"] = r["mean_tm_ribodiffusion"]
        res[f"matched.{key}.mean_tm.{r['reference_method']}"] = r["mean_tm_reference"]
        for k in ("diff", "lo", "hi"):
            res[f"matched.{key}.{k}"] = r[k]
    arm = ss[ss["method"] != "Native"].groupby("method")["tm_score"].agg(["size", "mean"])
    for method in ("RiboDiffusion", "gRNAde-clean"):
        res[f"af3_single_sequence.arm_mean.{method}"] = arm.loc[method, "mean"]
        res[f"af3_single_sequence.arm_n.{method}"] = arm.loc[method, "size"]
    return pd.DataFrame(rows)


def msa_effect(res: dict) -> pd.DataFrame:
    """Paired AlphaFold3 TM difference, MSA-conditioned minus single-sequence."""
    af = load("structure3d/alphafold3.csv.gz")
    key = ["target_id", "method", "slot"]
    msa = af[af["mode"] == "msa"].set_index(key)
    single = af[af["mode"] == "single_sequence"].set_index(key)["tm_score"].rename("tm_single")
    pair = msa[["msa_depth", "tm_score"]].join(single, how="inner").reset_index()
    pair["delta"] = pair["tm_score"] - pair["tm_single"]
    subsets = {"all": pair,
               "depth_1_noise_floor": pair[pair["msa_depth"] <= 1],
               "with_homologs": pair[pair["msa_depth"] > 1],
               "natives_with_homologs": pair[(pair["method"] == "Native") & (pair["msa_depth"] > 1)]}
    rows = []
    for label, g in subsets.items():
        b = stats.cluster_bootstrap(g["delta"], g["target_id"], n_boot=N_BOOT, seed=SEED)
        rows.append({"subset": label, "n": b["n"], "n_targets": b["n_clusters"],
                     "mean_delta": b["point"], "lo": b["lo"], "hi": b["hi"],
                     "mean_abs_delta": g["delta"].abs().mean(),
                     "frac_exactly_zero": (g["delta"] == 0).mean()})
    nh = rows[-1]
    res["af3_msa.natives_with_homologs.n"] = nh["n"]
    for k in ("mean_delta", "lo", "hi"):
        res[f"af3_msa.natives_with_homologs.{k}"] = nh[k]
    return pd.DataFrame(rows)


def run() -> dict[str, float]:
    res: dict[str, float] = {}
    save(native_panel(res), "structure3d_panel.csv")
    save(ratio_3d_2d(res), "structure3d_vs_2d.csv")
    save(matched(res), "structure3d_matched.csv")
    save(msa_effect(res), "structure3d_msa.csv")
    return {f"structure_3d.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("structure_3d", run())
