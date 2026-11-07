"""The ACCORD evaluation protocol: fold, score, summarise, report.

Reporting rules implemented here:

1. The requested output slot is the unit. Failed generations and failed oracle
   calls stay in the denominator.
2. ``Y_all`` is the primary scalar, always reported with ``C_complete``,
   ``C_any``, ``Q_mean``, ``Q_worst``, per-oracle scores and disagreement. No
   weighted composite is formed.
3. Disagreement is a robustness flag. Only the gated disagreement on the tier
   panel receives a tier.
4. The 0.5 gate is applied identically to every method and only screens out
   shared failures.
5. Tier cut-offs belong to the ViennaRNA + EternaFold + CONTRAfold panel.
6. Every report carries the fixed common panel. When a design oracle is
   declared, a leave-design-oracle-out view is added; self-evaluation and
   transfer claims must use it.
7. A designed-to-reference disagreement ratio is formed only on shared targets
   scored identically.
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from . import metrics as M
from . import oracles as O
from .structure import compare, is_dotbracket

REQUIRED_COLUMNS = ("target_id", "sequence", "target_structure")


def _score_one(args: tuple) -> dict:
    """Fold one sequence through the oracles and score it against its target."""
    seq, target, panel, nupack_config = args
    row: dict = {}
    seq = O.normalise_sequence(seq) if isinstance(seq, str) else ""
    if not seq:
        problem = "no sequence (failed generation)"
    elif not O.is_rna(seq):
        problem = "sequence contains letters other than A, C, G, U"
    elif not is_dotbracket(target):
        problem = "target structure is not a balanced dot-bracket string"
    elif len(target) != len(seq):
        problem = "sequence and target structure differ in length"
    else:
        problem = ""
    cfg = O.nupack_configs()[nupack_config]
    for oracle in panel:
        row[f"structure_{oracle}"] = None
        row[f"mcc_{oracle}"] = np.nan
        if problem:
            row[f"error_{oracle}"] = problem
            continue
        fr = O.fold(seq, oracle, nupack_config)
        if not fr.ok or fr.structure is None or len(fr.structure) != len(seq):
            row[f"error_{oracle}"] = fr.error or "no structure returned"
            continue
        ref = O.canonical_pairs_only(target, seq) if (oracle == "nupack" and cfg.wc_only_filter) else target
        row[f"structure_{oracle}"] = fr.structure
        row[f"mcc_{oracle}"] = compare(fr.structure, ref).mcc
        row[f"error_{oracle}"] = ""
    return row


_RESULT_PREFIXES = ("mcc_", "structure_", "error_")


def _validate(designs: pd.DataFrame) -> pd.DataFrame:
    """Check the input table, drop earlier results and add ``design_id`` if absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in designs.columns]
    if missing:
        raise KeyError(f"designs table is missing columns {missing}; "
                       f"required: {', '.join(REQUIRED_COLUMNS)}")
    if len(designs) == 0:
        raise ValueError("designs table has no rows")
    if designs["target_id"].isna().any():
        raise ValueError(f"{int(designs['target_id'].isna().sum())} rows have no target_id")
    stale = [c for c in designs.columns
             if c.startswith(_RESULT_PREFIXES) and c.split("_", 1)[1] in O.SUPPORTED]
    out = designs.drop(columns=stale).reset_index(drop=True)
    if stale:
        warnings.warn(f"ignoring existing result columns {stale}; every score is recomputed",
                      stacklevel=3)
    if "design_id" not in out.columns:
        out.insert(0, "design_id", [f"d{i}" for i in range(len(out))])
    out["target_id"] = out["target_id"].astype(str)
    return out


def _check_panel(panel: Sequence[str]) -> list[str]:
    """Return the panel as a list after checking names and duplicates."""
    panel = list(panel)
    if not panel:
        raise ValueError("the panel is empty")
    unknown = [o for o in panel if o not in O.SUPPORTED]
    if unknown:
        raise ValueError(f"unknown oracles {unknown}; supported: {', '.join(O.SUPPORTED)}")
    if len(set(panel)) != len(panel):
        raise ValueError(f"panel lists an oracle more than once: {panel}")
    return panel


def score(designs: pd.DataFrame, oracles: Sequence[str] = O.PANEL_4,
          nupack_config: str = "canonical", n_jobs: int = 1) -> pd.DataFrame:
    """Fold every design with every oracle and attach per-oracle MCC.

    Args:
        designs: One row per requested output slot with ``target_id``,
            ``sequence`` and ``target_structure``. Leave ``sequence`` empty for
            a slot whose generation failed; it is kept as a failure.
        oracles: Oracles to run.
        nupack_config: NUPACK configuration name.
        n_jobs: Worker processes.

    Returns:
        ``designs`` with ``structure_<oracle>``, ``mcc_<oracle>`` and
        ``error_<oracle>`` columns.
    """
    oracles = _check_panel(oracles)
    df = _validate(designs)
    if nupack_config not in O.nupack_configs():
        raise ValueError(f"unknown NUPACK configuration {nupack_config!r}")
    jobs = [(s, t, tuple(oracles), nupack_config)
            for s, t in zip(df["sequence"], df["target_structure"])]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_score_one, jobs, chunksize=16))
    else:
        rows = [_score_one(j) for j in jobs]
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


@dataclass
class Report:
    """Result of an ACCORD evaluation.

    Attributes:
        per_design: One row per slot with per-oracle MCC, design-level
            quantities on the common panel, the tier and status columns.
        summary: Machine-readable report (see :meth:`to_json`).
    """

    per_design: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def to_json(self, path: Optional[str | Path] = None, indent: int = 2) -> str:
        """Serialise the summary, optionally writing it to ``path``."""
        text = json.dumps(_jsonable(self.summary), indent=indent)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    def save(self, out_dir: str | Path) -> None:
        """Write ``per_design.csv`` and ``report.json`` to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.per_design.to_csv(out / "per_design.csv", index=False)
        self.to_json(out / "report.json")


def _jsonable(obj):
    """Convert numpy scalars and NaN for JSON output."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def report_from_scores(scored: pd.DataFrame, panel: Sequence[str] = O.PANEL_4,
                       design_oracle: Optional[str] = None,
                       threshold: float = M.THRESHOLD,
                       weighting: str = "target",
                       reference: Optional[pd.DataFrame] = None,
                       meta: Optional[dict] = None) -> Report:
    """Build the ACCORD report from per-oracle MCC columns.

    This is the entry point when MCC values were computed elsewhere; it runs no
    oracle.

    Args:
        scored: One row per slot with ``target_id`` and ``mcc_<oracle>``
            columns (``nan`` marks a failure).
        panel: Fixed common panel.
        design_oracle: Oracle used inside the design method's objective, if
            any. Adds the leave-design-oracle-out view.
        threshold: Pass threshold for C_any, Y_all and D_cond. Tiers always use
            the calibrated gate of ``configs/tiers.json`` (0.5).
        weighting: ``"target"`` (equal target weight) or ``"slot"``.
        reference: Optional slot table of reference sequences (for example
            natives) scored identically, for the reference ratio.
        meta: Extra provenance to store in the report.

    Returns:
        :class:`Report`.
    """
    panel = _check_panel(panel)
    if weighting not in ("target", "slot"):
        raise ValueError("weighting must be 'target' or 'slot'")
    target_col = "target_id" if weighting == "target" else None
    if target_col is not None and target_col not in scored.columns:
        raise KeyError("target weighting needs a target_id column; use weighting='slot'")
    views = {"common_panel": panel}
    if design_oracle is not None:
        if design_oracle not in O.SUPPORTED:
            raise ValueError(f"unknown design oracle {design_oracle!r}")
        lodo = [o for o in panel if o != design_oracle]
        views["leave_design_oracle_out"] = lodo
    per = M.design_table(scored, panel, threshold)
    tier_ready = all(c in scored.columns for c in M.mcc_columns(O.TIER_PANEL))
    if tier_ready:
        per["tier"] = [M.tier_from_scores(dict(zip(O.TIER_PANEL, r)))
                       for r in scored[M.mcc_columns(O.TIER_PANEL)].to_numpy(float)]
    if set(O.FAMILIES).issubset(panel):
        per["d_family"] = [M.family_disagreement(dict(zip(panel, r)))
                           for r in scored[M.mcc_columns(panel)].to_numpy(float)]

    summary: dict = {
        "accord_version": __version__,
        "threshold": threshold,
        "weighting": weighting,
        "design_oracle": design_oracle,
        "views": {name: M.summarize(scored, oracles, threshold, target_col)
                  for name, oracles in views.items()},
    }
    if design_oracle is not None and design_oracle not in panel:
        summary["views"]["leave_design_oracle_out"]["note"] = (
            "the design oracle is not in the common panel; this view equals the common panel")
    if tier_ready:
        summary["tiers"] = {"panel": list(O.TIER_PANEL), "gate": M.tier_gate(),
                            "cuts": M.tier_config()["cuts"]["rule"],
                            "counts": M.tier_counts(scored)}
    else:
        summary["tiers"] = {"panel": list(O.TIER_PANEL), "counts": None,
                            "note": "tier panel incomplete; no tiers assigned"}
    if reference is not None:
        summary["reference_ratio"] = M.reference_ratio(scored, reference, panel,
                                                       threshold=threshold)
    if meta:
        summary["meta"] = meta
    return Report(per_design=per, summary=summary)


def evaluate(designs: pd.DataFrame, panel: Sequence[str] = O.PANEL_4,
             design_oracle: Optional[str] = None, threshold: float = M.THRESHOLD,
             nupack_config: str = "canonical", weighting: str = "target",
             reference: Optional[pd.DataFrame] = None, n_jobs: int = 1) -> Report:
    """Run the full ACCORD evaluation on a pool of designs.

    The tier-panel oracles are always run, so that tiers are available even
    when the common panel differs.

    Args:
        designs: One row per requested output slot with ``target_id``,
            ``sequence`` and ``target_structure`` (and optionally
            ``design_id``).
        panel: Fixed common panel.
        design_oracle: Oracle used inside the design objective, if any.
        threshold: Pass threshold and gate.
        nupack_config: NUPACK configuration name (``canonical`` by default).
        weighting: ``"target"`` or ``"slot"``.
        reference: Optional reference table (same columns as ``designs``),
            scored with the same panel, for the reference ratio.
        n_jobs: Worker processes for folding.

    Returns:
        :class:`Report`.
    """
    panel = _check_panel(panel)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    run = list(dict.fromkeys(panel + list(O.TIER_PANEL)))
    status = O.availability()
    down = [o for o in run if not status[o]["available"]]
    if down:
        warnings.warn(f"oracles not available: {down}; their slots are reported as incomplete "
                      "(see accord.availability())", stacklevel=2)
    scored = score(designs, run, nupack_config, n_jobs)
    ref_scored = score(reference, run, nupack_config, n_jobs) if reference is not None else None
    meta = {"nupack_config": O.nupack_configs()[nupack_config].as_dict(),
            "environment": O.environment()}
    return report_from_scores(scored, panel, design_oracle, threshold, weighting,
                              ref_scored, meta)
