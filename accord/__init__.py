"""ACCORD: Across-oracle Calibration for COmputational RNA Design.

ACCORD evaluates designed RNA sequences against a fixed panel of secondary
structure predictors spanning two parameter families, reports failure-aware
pass rates and per-oracle quality, and flags cross-oracle disagreement with
SHAPE-calibrated uncertainty tiers.

Typical use::

    import pandas as pd
    import accord

    designs = pd.read_csv("designs.csv")   # target_id, sequence, target_structure
    report = accord.evaluate(designs, design_oracle="vienna")
    report.save("accord_out")
"""

__version__ = "2.0.0"

from .metrics import assign_tier, disagreement, family_disagreement, summarize  # noqa: E402
from .oracles import PANEL_4, TIER_PANEL, availability, fold  # noqa: E402
from .protocol import Report, evaluate, report_from_scores, score  # noqa: E402
from .structure import compare, mcc, pairing_disagreement  # noqa: E402

__all__ = [
    "PANEL_4", "TIER_PANEL", "Report", "assign_tier", "availability", "compare",
    "disagreement", "evaluate", "family_disagreement", "fold", "mcc",
    "pairing_disagreement", "report_from_scores", "score", "summarize",
]
