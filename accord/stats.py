"""Statistics used by ACCORD reports and the paper's analyses.

Designs are nested inside targets, and calibration sequences inside construct
families or design methods, so every interval here resamples whole clusters.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


def cluster_bootstrap(values: Sequence[float], clusters: Sequence,
                      statistic: Callable[[np.ndarray], float] = np.mean,
                      n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile interval of a statistic, resampling whole clusters.

    Args:
        values: Observation-level values.
        clusters: Cluster label per observation.
        statistic: Function applied to the pooled resampled values.
        n_boot: Number of bootstrap draws.
        seed: Random seed.
        alpha: Two-sided error rate.

    Returns:
        Dictionary with ``point``, ``lo``, ``hi``, ``n`` and ``n_clusters``.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    keep = np.isfinite(v)
    v, c = v[keep], c[keep]
    if v.size == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0}
    uniq = np.unique(c)
    index = {u: np.flatnonzero(c == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        draws[b] = statistic(v[np.concatenate([index[p] for p in picked])])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(statistic(v)), "lo": float(lo), "hi": float(hi),
            "n": int(v.size), "n_clusters": int(uniq.size)}


def target_weighted_bootstrap(values: Sequence[float], targets: Sequence,
                              inner: Callable[[np.ndarray], float] = np.mean,
                              n_boot: int = 5000, seed: int = 0,
                              alpha: float = 0.05) -> dict:
    """Equal-target-weight mean with a target-level bootstrap interval.

    Values are first aggregated inside each target with ``inner``; the
    per-target summaries are then averaged and resampled.

    Args:
        values: Slot-level values (non-finite values are ignored).
        targets: Target label per slot.
        inner: Within-target aggregator.
        n_boot: Number of bootstrap draws.
        seed: Random seed.
        alpha: Two-sided error rate.

    Returns:
        Dictionary with ``point``, ``lo``, ``hi``, ``n`` and ``n_clusters``.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(targets)
    keep = np.isfinite(v)
    v, c = v[keep], c[keep]
    uniq = np.unique(c)
    if uniq.size == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0}
    per_target = np.array([inner(v[c == u]) for u in uniq], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(per_target, size=per_target.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(per_target.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(v.size), "n_clusters": int(uniq.size)}


def paired_target_bootstrap(per_target_a: Mapping, per_target_b: Mapping,
                            n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired target-level bootstrap of the difference ``a - b``.

    Args:
        per_target_a: Mapping target -> summary for arm A.
        per_target_b: Mapping target -> summary for arm B.
        n_boot: Number of bootstrap draws.
        seed: Random seed.
        alpha: Two-sided error rate.

    Returns:
        Dictionary with ``n_targets``, ``mean_diff``, ``mean_ci``,
        ``median_diff``, ``median_ci``, win/tie/loss counts and the Wilcoxon
        signed-rank p-value.
    """
    shared = sorted(set(per_target_a) & set(per_target_b))
    diffs = np.array([per_target_a[t] - per_target_b[t] for t in shared], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"n_targets": 0, "mean_diff": np.nan, "mean_ci": (np.nan, np.nan),
                "median_diff": np.nan, "median_ci": (np.nan, np.nan), "wins": 0,
                "ties": 0, "losses": 0, "wilcoxon_p": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(diffs, size=diffs.size, replace=True)
        means[b] = s.mean()
        medians[b] = np.median(s)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    try:
        wp = float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else np.nan
    except ValueError:
        wp = np.nan
    return {"n_targets": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "mean_ci": tuple(float(x) for x in np.percentile(means, q)),
            "median_diff": float(np.median(diffs)),
            "median_ci": tuple(float(x) for x in np.percentile(medians, q)),
            "wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()),
            "losses": int((diffs < 0).sum()), "wilcoxon_p": wp}


def bootstrap_spearman(x: Sequence[float], y: Sequence[float],
                       clusters: Optional[Sequence] = None, n_boot: int = 5000,
                       seed: int = 0, alpha: float = 0.05) -> dict:
    """Spearman correlation with a percentile bootstrap interval.

    Args:
        x: First variable.
        y: Second variable.
        clusters: Cluster label per observation; observations are resampled
            individually when ``None``.
        n_boot: Number of bootstrap draws.
        seed: Random seed.
        alpha: Two-sided error rate.

    Returns:
        Dictionary with ``rho``, ``p``, ``lo``, ``hi``, ``n`` and ``n_clusters``.
        ``p`` is the ordinary Spearman p-value and ignores clustering; use the
        interval for inference.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    c = np.asarray(clusters)[keep] if clusters is not None else None
    if x.size < 3:
        return {"rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan,
                "n": int(x.size), "n_clusters": 0}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    if c is None:
        for b in range(n_boot):
            idx = rng.integers(0, x.size, x.size)
            draws[b] = stats.spearmanr(x[idx], y[idx])[0]
        n_clusters = 0
    else:
        uniq = np.unique(c)
        index = {u: np.flatnonzero(c == u) for u in uniq}
        for b in range(n_boot):
            idx = np.concatenate([index[u] for u in rng.choice(uniq, size=uniq.size, replace=True)])
            draws[b] = stats.spearmanr(x[idx], y[idx])[0]
        n_clusters = int(uniq.size)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rho": float(rho), "p": float(p), "lo": float(lo), "hi": float(hi),
            "n": int(x.size), "n_clusters": n_clusters}


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     controls: Sequence[Sequence[float]]) -> dict:
    """Spearman correlation after rank-residualising control variables.

    Args:
        x: First variable.
        y: Second variable.
        controls: Control variables, each the same length as ``x``.

    Returns:
        Dictionary with ``rho_raw``, ``rho_partial``, ``p_partial``, ``n`` and
        ``retention`` (partial over raw magnitude).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ctrl = [np.asarray(c, dtype=float) for c in controls]
    keep = np.isfinite(x) & np.isfinite(y)
    for c in ctrl:
        keep &= np.isfinite(c)
    x, y, ctrl = x[keep], y[keep], [c[keep] for c in ctrl]
    if x.size < 5:
        return {"rho_raw": np.nan, "rho_partial": np.nan, "p_partial": np.nan,
                "n": int(x.size), "retention": np.nan}
    design = np.column_stack([np.ones(x.size)] + [stats.rankdata(c) for c in ctrl])

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    r, p = stats.pearsonr(resid(stats.rankdata(x)), resid(stats.rankdata(y)))
    raw = stats.spearmanr(x, y)[0]
    return {"rho_raw": float(raw), "rho_partial": float(r), "p_partial": float(p),
            "n": int(x.size),
            "retention": float(abs(r) / abs(raw)) if abs(raw) > 1e-12 else np.nan}


def hill_n2(matrix: np.ndarray, normalise: str = "correlation") -> dict:
    """Effective number of oracles from the eigen-spectrum of a panel.

    Hill's ``N2 = (sum lambda)^2 / sum lambda^2`` over the eigenvalues of the
    oracle-by-oracle correlation (or covariance) matrix. ``N2`` equals the
    number of oracles when they are uncorrelated with equal variance and
    approaches 1 when they are redundant. The value depends on which designs
    are pooled, so the scope must always be stated with it.

    Args:
        matrix: ``(n_designs, n_oracles)`` score matrix; rows with a missing
            value are dropped.
        normalise: ``"correlation"`` or ``"covariance"``.

    Returns:
        Dictionary with ``n``, ``k``, ``N2``, ``eigenvalues`` and ``explained``
        (the explained-variance fractions, i.e. the PCA spectrum when
        ``normalise="covariance"``).
    """
    m = np.asarray(matrix, dtype=float)
    m = m[np.isfinite(m).all(axis=1)]
    if m.shape[0] < 5 or m.shape[1] < 2:
        return {"n": int(m.shape[0]), "k": int(m.shape[1]), "N2": np.nan,
                "eigenvalues": [], "explained": []}
    if normalise == "correlation":
        mat = np.nan_to_num(np.corrcoef(m, rowvar=False), nan=0.0)
    elif normalise == "covariance":
        mat = np.cov(m, rowvar=False)
    else:
        raise ValueError("normalise must be 'correlation' or 'covariance'")
    lam = np.clip(np.sort(np.linalg.eigvalsh(mat))[::-1], 0.0, None)
    return {"n": int(m.shape[0]), "k": int(m.shape[1]),
            "N2": float(lam.sum() ** 2 / np.sum(lam ** 2)),
            "eigenvalues": lam.tolist(), "explained": (lam / lam.sum()).tolist()}
