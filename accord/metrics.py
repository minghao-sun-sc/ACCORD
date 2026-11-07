"""Design-level and pool-level ACCORD quantities.

Definitions (``t`` is the pass threshold, 0.5 by default; ``MCC_j`` is the
base-pair MCC of oracle ``j`` against the design target):

* A requested output slot is *complete* when every oracle of the panel returned
  a finite MCC. Generation and scoring failures stay in the denominator.
* ``C_complete`` = fraction of slots that are complete.
* ``C_any`` = fraction of slots that are complete and have ``max_j MCC_j >= t``.
* ``Y_all`` = fraction of slots that are complete and have ``min_j MCC_j >= t``.
  This is the primary comparison scalar.
* ``Q_mean`` and ``Q_worst`` = mean over complete slots of ``mean_j MCC_j`` and
  ``min_j MCC_j``.
* ``D`` = sample standard deviation (``ddof=1``) of ``MCC_j`` across the panel.
  ``D_raw`` is summarised over all complete slots; ``D_cond`` only over slots
  that pass the gate ``max_j MCC_j >= t``. Only ``D_cond`` on the tier panel
  receives the SHAPE-calibrated tier interpretation.

With target weighting (the default for multi-target comparisons), every
quantity is first computed inside each target and then averaged over targets.
``D`` is summarised by its median inside a target. Outcome counts and the
oracle-dependent fractions are always pooled over slots.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .oracles import FAMILIES, TIER_PANEL

#: Default pass threshold and gate.
THRESHOLD = 0.5

TIER_HIGH = "High"
TIER_MODERATE = "Moderate"
TIER_LOW = "Low"
TIER_GATED = "Gated out"
TIER_INCOMPLETE = "Incomplete"
TIER_LABELS = (TIER_HIGH, TIER_MODERATE, TIER_LOW, TIER_GATED, TIER_INCOMPLETE)


@lru_cache(maxsize=None)
def tier_config() -> dict:
    """Load the versioned tier definition shipped in ``configs/tiers.json``."""
    return json.loads(resources.files("accord").joinpath("configs/tiers.json").read_text())


def mcc_columns(oracles: Sequence[str]) -> list[str]:
    """Column names that hold per-oracle MCC values."""
    return [f"mcc_{o}" for o in oracles]


# --------------------------------------------------------------------------
# Design level
# --------------------------------------------------------------------------

def disagreement(scores: Sequence[float]) -> float:
    """Cross-oracle disagreement of one design.

    Args:
        scores: Per-oracle MCC values for one design.

    Returns:
        Sample standard deviation (``ddof=1``); ``nan`` when fewer than two
        scores are given or any score is missing.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return float("nan")
    return float(_clean_sd(np.std(arr, ddof=1)))


def _clean_sd(sd):
    """Round a standard deviation to 12 decimals.

    Identical scores can give a floating-point residue of about 1e-17 instead of
    0, which breaks genuine ties in rank statistics and top-k selection. Real
    differences in MCC are many orders of magnitude larger.
    """
    return np.round(sd, 12)


def family_disagreement(scores: Mapping[str, float]) -> float:
    """Disagreement between the physics and learned family means.

    Defined as ``|q_physics - q_learned| / sqrt(2)``, the standard deviation of
    the two family means, so that a family with two implementations does not
    count twice. This is a sensitivity statistic: no tier cut-offs apply to it.

    Args:
        scores: Mapping oracle -> MCC.

    Returns:
        Family-balanced disagreement, or ``nan`` when a family is missing or a
        score is not finite.
    """
    fam: dict[str, list[float]] = {}
    for oracle, value in scores.items():
        if oracle not in FAMILIES:
            continue
        if value is None or not np.isfinite(value):
            return float("nan")
        fam.setdefault(FAMILIES[oracle], []).append(float(value))
    if "physics" not in fam or "learned" not in fam:
        return float("nan")
    return abs(np.mean(fam["physics"]) - np.mean(fam["learned"])) / np.sqrt(2.0)


def tier_gate() -> float:
    """Gate on the best tier-panel MCC that was fixed with the tier calibration."""
    return float(tier_config()["gate"]["threshold"])


def assign_tier(d: float, max_mcc: float, gate: Optional[float] = None) -> str:
    """Uncertainty tier of one design on the tier panel.

    The cut-offs come from ``configs/tiers.json``: ``High`` when ``d < 0.05``,
    ``Moderate`` when ``0.05 <= d <= 0.15``, ``Low`` when ``d > 0.15``. Designs
    whose best oracle fails the gate are shared failures and get no tier.

    Args:
        d: Disagreement on the tier panel (ViennaRNA, EternaFold, CONTRAfold).
        max_mcc: Best MCC on the tier panel.
        gate: Gate on ``max_mcc``; defaults to the calibrated gate (0.5). The
            tier cut-offs are only defined for that gate.

    Returns:
        One of :data:`TIER_LABELS`.
    """
    gate = tier_gate() if gate is None else gate
    if d is None or max_mcc is None or not np.isfinite(d) or not np.isfinite(max_mcc):
        return TIER_INCOMPLETE
    if d < 0:
        raise ValueError(f"disagreement must be non-negative, got {d}")
    if max_mcc < gate:
        return TIER_GATED
    cuts = tier_config()["cuts"]
    if d < cuts["high_below"]:
        return TIER_HIGH
    if d <= cuts["low_above"]:
        return TIER_MODERATE
    return TIER_LOW


def tier_from_scores(scores: Mapping[str, float], gate: Optional[float] = None) -> str:
    """Tier of one design from its per-oracle MCC values.

    Only the three tier-panel oracles are used, whatever else is in ``scores``.

    Args:
        scores: Mapping oracle -> MCC; must contain the tier-panel oracles.
        gate: Gate on the best tier-panel MCC; defaults to the calibrated gate.

    Returns:
        One of :data:`TIER_LABELS`.
    """
    vals = [scores.get(o, np.nan) for o in TIER_PANEL]
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return TIER_INCOMPLETE
    return assign_tier(disagreement(arr), float(arr.max()), gate)


def design_table(df: pd.DataFrame, panel: Sequence[str],
                 threshold: float = THRESHOLD) -> pd.DataFrame:
    """Attach design-level quantities for one panel.

    Args:
        df: One row per requested slot with ``mcc_<oracle>`` columns (``nan``
            for failures).
        panel: Oracles of the panel.
        threshold: Pass threshold.

    Returns:
        Copy of ``df`` with ``complete``, ``any_pass``, ``all_pass``, ``q_mean``,
        ``q_worst``, ``q_best`` and ``d`` columns. Continuous values are ``nan``
        for incomplete slots.
    """
    cols = mcc_columns(panel)
    if len(set(cols)) != len(cols):
        raise ValueError(f"panel lists an oracle more than once: {list(panel)}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing MCC columns: {missing}")
    duplicated = sorted({c for c in df.columns[df.columns.duplicated()] if c in cols})
    if duplicated:
        raise ValueError(f"duplicated MCC columns: {duplicated}")
    m = df[cols].to_numpy(dtype=float)
    complete = np.isfinite(m).all(axis=1)
    out = df.copy()
    out["complete"] = complete
    passes = np.nan_to_num(m, nan=-np.inf) >= threshold
    out["any_pass"] = complete & passes.any(axis=1)
    out["all_pass"] = complete & passes.all(axis=1)
    for name, fn in (("q_mean", np.mean), ("q_worst", np.min), ("q_best", np.max)):
        v = np.full(len(out), np.nan)
        if complete.any():
            v[complete] = fn(m[complete], axis=1)
        out[name] = v
    d = np.full(len(out), np.nan)
    if complete.any() and len(cols) >= 2:
        d[complete] = _clean_sd(m[complete].std(axis=1, ddof=1))
    out["d"] = d
    return out


# --------------------------------------------------------------------------
# Pool level
# --------------------------------------------------------------------------

def _rate_block(t: pd.DataFrame, oracles: Sequence[str]) -> dict:
    """Summaries of one group of slots (one target, or the pooled set)."""
    comp = t["complete"].to_numpy(bool)
    gated = t["any_pass"].to_numpy(bool)
    d = t["d"].to_numpy(float)
    d_comp, d_gated = d[comp & np.isfinite(d)], d[gated & np.isfinite(d)]
    out = {
        "C_complete": float(comp.mean()) if len(t) else np.nan,
        "C_any": float(t["any_pass"].mean()) if len(t) else np.nan,
        "Y_all": float(t["all_pass"].mean()) if len(t) else np.nan,
        "Q_mean": float(t.loc[comp, "q_mean"].mean()) if comp.any() else np.nan,
        "Q_worst": float(t.loc[comp, "q_worst"].mean()) if comp.any() else np.nan,
        "D_raw": float(np.median(d_comp)) if d_comp.size else np.nan,
        "D_cond": float(np.median(d_gated)) if d_gated.size else np.nan,
    }
    for o in oracles:
        vals = t.loc[t["complete"], f"mcc_{o}"].to_numpy(float)
        out[f"mean_mcc_{o}"] = float(vals.mean()) if vals.size else np.nan
    return out


def summarize(df: pd.DataFrame, panel: Sequence[str], threshold: float = THRESHOLD,
              target_col: Optional[str] = "target_id") -> dict:
    """End-to-end summary of a raw pool of requested output slots.

    Args:
        df: One row per requested slot with ``mcc_<oracle>`` columns. Rows for
            failed generations or failed oracle calls carry ``nan``.
        panel: Oracles of the panel.
        threshold: Pass threshold and gate.
        target_col: Column identifying the target. When given, every rate and
            mean is computed per target and then averaged with equal target
            weight; when ``None`` the slots are pooled.

    Returns:
        Dictionary with ``n_slots``, ``n_targets``, ``C_complete``, ``C_any``,
        ``Y_all``, ``Q_mean``, ``Q_worst``, ``D_raw``, ``D_cond``, per-oracle
        mean MCC, pooled outcome counts and oracle-dependent fractions.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if len(df) == 0:
        raise ValueError("no slots to summarise")
    t = design_table(df, panel, threshold)
    if target_col is not None:
        if target_col not in t.columns:
            raise KeyError(f"target weighting needs a {target_col!r} column; "
                           "pass target_col=None to pool slots")
        if t[target_col].isna().any():
            raise ValueError(f"{int(t[target_col].isna().sum())} slots have no {target_col}")
        per_target = pd.DataFrame([_rate_block(g, panel) for _, g in t.groupby(target_col, sort=False)])
        block = {k: float(per_target[k].mean(skipna=True)) for k in per_target.columns}
        weighting = "target"
    else:
        block = _rate_block(t, panel)
        weighting = "slot"
    id_col = target_col or ("target_id" if "target_id" in t.columns else None)
    n_targets = int(t[id_col].nunique()) if id_col is not None else None

    comp = t["complete"].to_numpy(bool)
    n_all = int(t["all_pass"].sum())
    n_any = int(t["any_pass"].sum())
    n_some = n_any - n_all
    counts = {
        "n_slots": int(len(t)),
        "n_incomplete": int((~comp).sum()),
        "n_pass_all": n_all,
        "n_pass_some_not_all": n_some,
        "n_fail_all": int(comp.sum()) - n_any,
    }
    dependent = {
        "oracle_dependent_conditional": n_some / n_any if n_any else float("nan"),
        "oracle_dependent_raw": n_some / len(t) if len(t) else float("nan"),
    }
    return {"panel": list(panel), "threshold": threshold, "weighting": weighting,
            "n_targets": n_targets, **counts, **block, **dependent}


def tier_counts(df: pd.DataFrame, gate: Optional[float] = None) -> dict[str, int]:
    """Count designs per uncertainty tier on the tier panel.

    Args:
        df: Slot table with ``mcc_vienna``, ``mcc_eternafold`` and
            ``mcc_contrafold`` columns.
        gate: Gate on the best tier-panel MCC; defaults to the calibrated gate.

    Returns:
        Mapping tier label -> count (every label present, possibly zero).
    """
    cols = mcc_columns(TIER_PANEL)
    tiers = [tier_from_scores(dict(zip(TIER_PANEL, row)), gate)
             for row in df[cols].to_numpy(dtype=float)]
    s = pd.Series(tiers, dtype=object).value_counts()
    return {label: int(s.get(label, 0)) for label in TIER_LABELS}


def reference_ratio(designed: pd.DataFrame, reference: pd.DataFrame,
                    panel: Sequence[str], target_col: str = "target_id",
                    threshold: float = THRESHOLD, flag_above: float = 2.0) -> dict:
    """Ratio of designed to reference disagreement on shared targets.

    Both pools must be scored with the same panel, oracle parameters and
    post-processing. The ratio uses the target-weighted median raw disagreement
    on the targets the two pools share. Without shared targets no ratio is
    formed. The ``flag_above`` value is a heuristic for attention, not a test.

    Args:
        designed: Slot table of designed sequences.
        reference: Slot table of reference sequences (for example natives).
        panel: Oracles of the panel.
        target_col: Target identifier column.
        threshold: Pass threshold used for completeness bookkeeping.
        flag_above: Ratio above which the result is flagged.

    Returns:
        Dictionary with the two disagreement summaries, the ratio, the number
        of shared targets and the flag.
    """
    designed = designed.assign(**{target_col: designed[target_col].astype(str)})
    reference = reference.assign(**{target_col: reference[target_col].astype(str)})
    shared = sorted(set(designed[target_col]) & set(reference[target_col]))
    if not shared:
        return {"n_shared_targets": 0, "ratio": None, "flag": None,
                "note": "no shared targets; report the two distributions separately"}
    a = summarize(designed[designed[target_col].isin(shared)], panel, threshold, target_col)
    b = summarize(reference[reference[target_col].isin(shared)], panel, threshold, target_col)
    ratio = a["D_raw"] / b["D_raw"] if b["D_raw"] and np.isfinite(b["D_raw"]) else float("nan")
    return {"n_shared_targets": len(shared), "D_raw_designed": a["D_raw"],
            "D_raw_reference": b["D_raw"], "ratio": ratio,
            "flag": bool(np.isfinite(ratio) and ratio > flag_above), "flag_above": flag_above}
