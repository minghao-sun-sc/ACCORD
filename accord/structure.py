"""Secondary-structure parsing and comparison.

All scores in ACCORD use the base-pair convention: the confusion matrix is built
over the ``n * (n - 1) / 2`` candidate base pairs of a length-``n`` sequence, and
a true positive is a pair present in both the prediction and the reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}
_ALLOWED = set(".") | set(_OPEN_TO_CLOSE) | set(_CLOSE_TO_OPEN)


def is_dotbracket(db: str) -> bool:
    """Return True if ``db`` is a balanced dot-bracket string.

    Args:
        db: Candidate structure string.

    Returns:
        True when every character is a dot or bracket and every bracket family
        is balanced.
    """
    if not isinstance(db, str) or not db or set(db) - _ALLOWED:
        return False
    depth = {k: 0 for k in _OPEN_TO_CLOSE}
    for ch in db:
        if ch in depth:
            depth[ch] += 1
        elif ch in _CLOSE_TO_OPEN:
            depth[_CLOSE_TO_OPEN[ch]] -= 1
            if depth[_CLOSE_TO_OPEN[ch]] < 0:
                return False
    return all(v == 0 for v in depth.values())


def dotbracket_to_pairs(db: str) -> set[tuple[int, int]]:
    """Convert a dot-bracket string into a set of zero-based ``(i, j)`` pairs.

    The four bracket families ``()``, ``[]``, ``{}`` and ``<>`` are parsed, so
    pseudoknotted annotations are kept rather than dropped.

    Args:
        db: Dot-bracket structure.

    Returns:
        Set of pairs with ``i < j``.
    """
    pairs: set[tuple[int, int]] = set()
    stacks: dict[str, list[int]] = {k: [] for k in _OPEN_TO_CLOSE}
    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in _CLOSE_TO_OPEN:
            stack = stacks[_CLOSE_TO_OPEN[ch]]
            if stack:
                j = stack.pop()
                pairs.add((min(i, j), max(i, j)))
    return pairs


def paired_mask(db: str) -> np.ndarray:
    """Boolean vector marking the positions that take part in a base pair.

    Args:
        db: Dot-bracket structure.

    Returns:
        Array of length ``len(db)``.
    """
    mask = np.zeros(len(db), dtype=bool)
    for i, j in dotbracket_to_pairs(db):
        mask[i] = True
        mask[j] = True
    return mask


@dataclass(frozen=True)
class Comparison:
    """Base-pair confusion matrix and derived scores for one structure pair."""

    mcc: float
    f1: float
    sensitivity: float
    ppv: float
    tp: int
    fp: int
    fn: int
    tn: int
    exact_match: bool

    def as_dict(self) -> dict:
        """Return the comparison as a plain dictionary."""
        return asdict(self)


def compare(pred: str, ref: str) -> Comparison:
    """Compare a predicted structure with a reference on the base-pair matrix.

    When both structures contain no pairs the MCC denominator is zero; the MCC is
    then defined as 1 (identical pair sets).

    Args:
        pred: Predicted dot-bracket structure.
        ref: Reference dot-bracket structure.

    Returns:
        :class:`Comparison`.

    Raises:
        ValueError: If the two structures differ in length.
    """
    if len(pred) != len(ref):
        raise ValueError(f"length mismatch: prediction {len(pred)} vs reference {len(ref)}")
    n = len(pred)
    p = dotbracket_to_pairs(pred)
    r = dotbracket_to_pairs(ref)
    tp = len(p & r)
    fp = len(p - r)
    fn = len(r - p)
    tn = n * (n - 1) // 2 - tp - fp - fn
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    if denom == 0.0:
        mcc = 1.0 if p == r else 0.0
    else:
        mcc = float(tp * tn - fp * fn) / denom
    sens = tp / (tp + fn) if (tp + fn) else 1.0
    ppv = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    return Comparison(mcc=float(mcc), f1=float(f1), sensitivity=float(sens),
                      ppv=float(ppv), tp=tp, fp=fp, fn=fn, tn=tn,
                      exact_match=pred == ref)


def mcc(pred: str, ref: str) -> float:
    """Base-pair Matthews correlation coefficient between two structures.

    Args:
        pred: Predicted dot-bracket structure.
        ref: Reference dot-bracket structure.

    Returns:
        MCC in ``[-1, 1]``.
    """
    return compare(pred, ref).mcc


def pairing_disagreement(structures: Sequence[str]) -> float:
    """Target-free disagreement among several predicted structures.

    Each structure is reduced to its paired/unpaired indicator vector; the
    population standard deviation (``ddof=0``) across oracles is taken at every
    position and averaged over positions. This is the statistic on which the
    uncertainty tiers were calibrated against SHAPE data, where no design target
    exists. For designs with a target, ACCORD reports the target-MCC disagreement
    of :func:`accord.metrics.disagreement` instead.

    Args:
        structures: Predicted dot-bracket structures of equal length.

    Returns:
        Mean per-position standard deviation in ``[0, 0.5]``.

    Raises:
        ValueError: If fewer than two structures are given or lengths differ.
    """
    if len(structures) < 2:
        raise ValueError("at least two structures are required")
    lengths = {len(s) for s in structures}
    if len(lengths) != 1:
        raise ValueError(f"structures differ in length: {sorted(lengths)}")
    stack = np.stack([paired_mask(s).astype(float) for s in structures])
    return float(stack.std(axis=0).mean())
