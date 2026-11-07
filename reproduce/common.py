"""Shared helpers for the reproduction scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "outputs"
EXPECTED = Path(__file__).resolve().parent / "expected.json"

PANEL_4 = ["vienna", "nupack", "eternafold", "contrafold"]
PANEL_3 = ["vienna", "eternafold", "contrafold"]
N_BOOT = 5000
SEED = 0


def verify_data() -> list[str]:
    """Check every table in ``data/`` against the checksums in ``data/MANIFEST.tsv``.

    Returns:
        Files whose content differs from the manifest (empty when all match).
    """
    import hashlib

    manifest = pd.read_csv(DATA / "MANIFEST.tsv", sep="\t")
    bad = []
    for r in manifest.itertuples():
        path = DATA / r.file
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != r.sha256:
            bad.append(r.file)
    return bad


def load(rel: str) -> pd.DataFrame:
    """Read a table from ``data/``; target identifiers are kept as strings."""
    df = pd.read_csv(DATA / rel, dtype={"target_id": str})
    return df


def save(df: pd.DataFrame, name: str) -> Path:
    """Write a result table to ``reproduce/outputs``."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def expected() -> dict:
    """Values reported in the paper, keyed by result name."""
    return json.loads(EXPECTED.read_text())["values"]


def compare(results: dict[str, float]) -> pd.DataFrame:
    """Compare computed values with the reported ones.

    A value matches when it agrees with the reported value to the precision at
    which that value was reported (``tol`` in ``expected.json``).

    Args:
        results: Mapping result name -> computed value.

    Returns:
        One row per reported value with the computed value and a pass flag.
    """
    rows = []
    for key, spec in expected().items():
        if key not in results:
            continue
        got = float(results[key])
        ok = bool(np.isfinite(got) and abs(got - spec["value"]) <= spec["tol"] + 1e-12)
        rows.append({"result": key, "reported": spec["value"], "reproduced": got,
                     "tol": spec["tol"], "match": ok, "where": spec.get("where", "")})
    return pd.DataFrame(rows, columns=["result", "reported", "reproduced", "tol", "match", "where"])


def report(name: str, results: dict[str, float]) -> pd.DataFrame:
    """Print and return the comparison for one script."""
    table = compare(results)
    if table.empty:
        print(f"[{name}] no reported values to compare")
        return table
    n_ok = int(table["match"].sum())
    print(f"[{name}] {n_ok}/{len(table)} reported values reproduced")
    for r in table.itertuples():
        mark = "ok " if r.match else "DIFF"
        print(f"  {mark} {r.result:<58} reported {r.reported:>9.4f}  reproduced {r.reproduced:>9.4f}")
    return table
