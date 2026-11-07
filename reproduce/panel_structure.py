"""How many independent views does the oracle panel provide?

Hill's N2 on the eigen-spectrum of the per-design MCC matrix, for several pooling
scopes, plus the principal-component spectrum used in the paper's PCA figure.
The four 2D oracles behave as roughly two effective families at every scope,
which is why ACCORD is described as a cross-parameter-family disagreement
detector rather than four independent votes.

Output: ``outputs/panel_structure.csv``.
"""

from __future__ import annotations

import pandas as pd

from accord import stats
from common import PANEL_4, load, report, save

PANEL_5 = PANEL_4 + ["ribonanzanet_ss"]
#: The five design paradigms of the original audit (one cohort each).
FIVE_PRIMARY = ["rnainverse_eterna100", "nupack_eterna100", "learna_eterna100",
                "grnade_das", "ribodiffusion_das"]
#: The four cohorts plotted in the PCA figure (natives are added to them).
PCA_COHORTS = ["rnainverse_eterna100", "nupack_eterna100", "grnade_das", "ribodiffusion_das"]


def n2(df: pd.DataFrame, oracles: list[str], normalise: str = "correlation") -> dict:
    return stats.hill_n2(df[[f"mcc_{o}" for o in oracles]].to_numpy(float), normalise)


def run() -> dict[str, float]:
    c = load("designs/cohorts_2d.csv.gz")
    natives = load("designs/natives_das.csv.gz")
    scopes = [
        ("all_cohorts", c, PANEL_4, "correlation"),
        ("five_primary", c[c["cohort"].isin(FIVE_PRIMARY)], PANEL_4, "correlation"),
        ("eterna100_solutions", c[c["cohort"] == "eterna100_solutions"], PANEL_4, "correlation"),
        ("rnainverse_eterna100", c[c["cohort"] == "rnainverse_eterna100"], PANEL_4, "correlation"),
        ("rnainverse_das_5oracle", c[c["cohort"] == "rnainverse_das"], PANEL_5, "correlation"),
        ("pca_designs_covariance", c[c["cohort"].isin(PCA_COHORTS)], PANEL_4, "covariance"),
    ]
    rows, res = [], {}
    for name, df, oracles, norm in scopes:
        r = n2(df, oracles, norm)
        rows.append({"scope": name, "panel": "+".join(oracles), "normalise": norm,
                     "n": r["n"], "N2": r["N2"], "explained_pc1": r["explained"][0],
                     "explained_pc2": r["explained"][1]})
        res[f"{name}.N2"] = r["N2"]
        res[f"{name}.n"] = r["n"]
    res["all_cohorts.explained_pc1"] = rows[0]["explained_pc1"]

    pca = pd.concat([c[c["cohort"].isin(PCA_COHORTS)], natives], ignore_index=True)
    r = n2(pca, PANEL_4, "covariance")
    rows.append({"scope": "pca_figure_with_natives", "panel": "+".join(PANEL_4),
                 "normalise": "covariance", "n": r["n"], "N2": r["N2"],
                 "explained_pc1": r["explained"][0], "explained_pc2": r["explained"][1]})
    res["pca_figure_with_natives.explained_pc1"] = r["explained"][0]
    res["pca_figure_with_natives.explained_pc2"] = r["explained"][1]
    save(pd.DataFrame(rows), "panel_structure.csv")
    return {f"panel_structure.{k}": v for k, v in res.items()}


if __name__ == "__main__":
    report("panel_structure", run())
