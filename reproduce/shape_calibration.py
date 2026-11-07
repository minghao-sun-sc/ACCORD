"""Retrospective chemical-mapping calibration of cross-oracle disagreement.

Two cohorts are read from ``data/calibration/`` (derived statistics only):

* Ribonanza GPN15k, 2A3-MaP: 14,271 sequences in 99 clusters (parent construct
  or source genome). ``D_pair`` is the mean over positions of the population
  standard deviation of the oracles' paired/unpaired calls; the outcome is the
  per-sequence SHAPE AUC, averaged over the same oracles.
* OpenKnotAIDesignData, 2A3-MaP: 25,930 designs from 13 design-method entries.
  Designs have a target, so the target-MCC disagreement ``D_MCC`` (sample
  standard deviation) and the gate ``max_j MCC_j >= 0.5`` are also defined.

Reproduces
* Spearman associations with cluster-bootstrap intervals (seed 0; 2,000 draws
  over constructs for Ribonanza, 5,000 draws over design methods for OpenKnot);
* mean SHAPE AUC per uncertainty tier on Ribonanza (no target, so no gate);
* the partial association after rank-residualising thermodynamic covariates
  (sequence length is constant at 170 nt, so it cannot act as a control);
* the held-out-oracle control (disagreement of two oracles against the SHAPE AUC
  of the third) and the family-balanced statistic;
* mean SHAPE AUC per uncertainty tier on the gated OpenKnot pool;
* a top-1% selection check on the gated pool. Designs tied at the selection
  cut-off contribute their mean (the expectation under random tie-breaking);
  random selection is the expectation, i.e. the pool mean.

Outputs: ``outputs/shape_calibration_{associations,tiers,ribonanza_tiers,selection}.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import accord
from accord import stats
from common import PANEL_3, SEED, load, report, save

BOOT_RIBONANZA = 2000
BOOT_OPENKNOT = 5000
THERMO = ["gc", "mfe", "ens_gap", "ens_diversity", "pos_entropy"]
TIERS = (accord.metrics.TIER_HIGH, accord.metrics.TIER_MODERATE, accord.metrics.TIER_LOW)


def association(df: pd.DataFrame, x: str, y: str, cluster: str, n_boot: int,
                name: str, conditioning: str = "all") -> dict:
    """Spearman rho of ``x`` and ``y`` with a cluster-bootstrap interval.

    Disagreement is rounded to 12 decimals first: equal oracle scores can give a
    standard deviation of about 1e-17 instead of 0, which would break genuine
    ties in an arbitrary order.
    """
    r = stats.bootstrap_spearman(df[x].round(12), df[y], clusters=df[cluster],
                                 n_boot=n_boot, seed=SEED)
    return {"analysis": name, "x": x, "y": y, "conditioning": conditioning,
            "n_boot": n_boot, **r}


def top_mean(score: np.ndarray, outcome: np.ndarray, frac: float) -> tuple[float, int, int]:
    """Mean outcome of the ``frac`` lowest scores, averaging over ties at the cut.

    Returns the mean, the number selected and the number of items tied at the
    cut-off value.
    """
    k = int(round(frac * score.size))
    cut = np.sort(score)[k - 1]
    better = score < cut - 1e-12
    tied = np.abs(score - cut) <= 1e-12
    mean = (outcome[better].sum() + (k - better.sum()) * outcome[tied].mean()) / k
    return float(mean), k, int(tied.sum())


def ribonanza(rb: pd.DataFrame) -> tuple[list[dict], dict]:
    rows = [
        association(rb, "d_pair3", "mean_auc3", "cluster", BOOT_RIBONANZA, "d_pair3"),
        association(rb, "d_pair4", "mean_auc4", "cluster", BOOT_RIBONANZA, "d_pair4"),
        association(rb, "d_pair_no_vienna", "auc_vienna", "cluster", BOOT_RIBONANZA, "heldout_vienna"),
        association(rb, "d_pair_no_contrafold", "auc_contrafold", "cluster", BOOT_RIBONANZA,
                    "heldout_contrafold"),
        association(rb, "d_pair_no_eternafold", "auc_eternafold", "cluster", BOOT_RIBONANZA,
                    "heldout_eternafold"),
        association(rb, "d_family3", "mean_auc3", "cluster", BOOT_RIBONANZA, "family3"),
        association(rb, "d_family4", "mean_auc4", "cluster", BOOT_RIBONANZA, "family4"),
        association(rb, "d_pair3", "dms_auc_vienna", "cluster", BOOT_RIBONANZA, "dms_vienna"),
    ]
    for r in rows:
        r["cohort"] = "ribonanza"
    extra = {"length_values": rb["length"].nunique()}
    cuts = accord.metrics.tier_config()["cuts"]
    d = rb["d_pair3"].round(12)
    tier = np.where(d < cuts["high_below"], "High", np.where(d <= cuts["low_above"], "Moderate", "Low"))
    tier_rows = []
    for name in TIERS:
        sel = rb.loc[tier == name, "mean_auc3"]
        extra[f"tier.{name}.mean_auc"], extra[f"tier.{name}.n"] = sel.mean(), len(sel)
        tier_rows.append({"tier": name, "n": len(sel), "mean_auc": sel.mean()})
    extra["tier.high_minus_low"] = extra["tier.High.mean_auc"] - extra["tier.Low.mean_auc"]
    save(pd.DataFrame(tier_rows), "shape_calibration_ribonanza_tiers.csv")
    p = stats.partial_spearman(rb["d_pair3"], rb["mean_auc3"], [rb[c] for c in THERMO])
    extra["partial_thermo.rho"], extra["partial_thermo.n"] = p["rho_partial"], p["n"]
    extra["partial_thermo.retention"] = p["retention"]
    return rows, extra


def openknot(ok: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict]:
    t = accord.metrics.design_table(ok, PANEL_3)
    gated = t[t["any_pass"]]
    rows = [
        association(t, "d_pair", "mean_auc", "method", BOOT_OPENKNOT, "d_pair"),
        association(gated, "d_pair", "mean_auc", "method", BOOT_OPENKNOT, "d_pair", "gated"),
        association(t, "d_family", "mean_auc", "method", BOOT_OPENKNOT, "family3"),
        association(t, "d_mcc", "mean_auc", "method", BOOT_OPENKNOT, "d_mcc"),
        association(gated, "d_mcc", "mean_auc", "method", BOOT_OPENKNOT, "d_mcc", "gated"),
    ]
    for r in rows:
        r["cohort"] = "openknot"

    tier = pd.Series([accord.assign_tier(d, m) for d, m in zip(t["d_mcc"], t["q_best"])], index=t.index)
    tiers = (t.groupby(tier)["mean_auc"].agg(n="size", mean_auc="mean")
             .reindex(list(TIERS) + [accord.metrics.TIER_GATED]).reset_index(names="tier"))

    auc = gated["mean_auc"].to_numpy(float)
    crit = {"q_mean": -gated["q_mean"].to_numpy(float), "q_worst": -gated["q_worst"].to_numpy(float),
            "d_pair": gated["d_pair"].to_numpy(float), "d_mcc": gated["d_mcc"].to_numpy(float)}
    sel = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        for name, score in crit.items():
            m, k, n_tied = top_mean(score, auc, frac)
            sel.append({"fraction": frac, "criterion": name, "n_selected": k,
                        "n_tied_at_cutoff": n_tied, "mean_auc": m, "random": float(auc.mean())})
    extra = {"n": len(t), "n_gated": len(gated), "gated_share": len(gated) / len(t),
             "n_methods": t["method"].nunique()}
    return rows, tiers, pd.DataFrame(sel), extra


def run() -> dict[str, float]:
    rb = load("calibration/ribonanza_shape.csv.gz")
    ok = load("calibration/openknot_shape.csv.gz")
    res: dict[str, float] = {}

    rows_r, extra_r = ribonanza(rb)
    rows_o, tiers, sel, extra_o = openknot(ok)
    assoc = pd.DataFrame(rows_r + rows_o)
    save(assoc[["cohort", "analysis", "conditioning", "x", "y", "n", "n_clusters", "n_boot",
                "rho", "lo", "hi", "p"]], "shape_calibration_associations.csv")
    save(tiers, "shape_calibration_tiers.csv")
    save(sel, "shape_calibration_selection.csv")

    for r in assoc.itertuples():
        key = f"{r.cohort}.{r.analysis}" + ("_gated" if r.conditioning == "gated" else "")
        for k in ("rho", "lo", "hi", "n", "n_clusters"):
            res[f"{key}.{k}"] = getattr(r, k)
    res.update({f"ribonanza.{k}": v for k, v in extra_r.items()})
    res.update({f"openknot.{k}": v for k, v in extra_o.items()})
    for r in tiers.itertuples():
        res[f"openknot.tier_gated.{r.tier}.mean_auc"] = r.mean_auc
        res[f"openknot.tier_gated.{r.tier}.n"] = r.n
    top = sel[sel["fraction"] == 0.01]
    for r in top.itertuples():
        res[f"openknot.select_top1.{r.criterion}"] = r.mean_auc
    res["openknot.select_top1.random"] = float(top["random"].iloc[0])
    return {f"shape_calibration.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("shape_calibration", run())
