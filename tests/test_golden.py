"""Golden-output tests: real oracle backends must reproduce released scores exactly.

``golden/crossed_57_sample.csv`` holds 45 designs (5 per method) from the released
57-target comparison with their published per-oracle MCC. Each available backend
is run on them; oracles that are not installed are skipped, so these tests pass
trivially only when no backend is present.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import accord
from accord import oracles as O

GOLDEN = Path(__file__).parent / "golden" / "crossed_57_sample.csv"
STATUS = O.availability()


@pytest.fixture(scope="module")
def golden():
    return pd.read_csv(GOLDEN, dtype={"target_id": str})


@pytest.mark.oracles
@pytest.mark.parametrize("oracle", list(O.PANEL_4))
def test_backend_reproduces_published_mcc(golden, oracle):
    if not STATUS[oracle]["available"]:
        pytest.skip(f"{oracle} unavailable: {STATUS[oracle]['detail']}")
    scored = accord.score(golden.drop(columns=[c for c in golden if c.startswith("mcc_")]),
                          oracles=[oracle])
    got = scored[f"mcc_{oracle}"].to_numpy(float)
    want = golden[f"mcc_{oracle}"].to_numpy(float)
    assert np.isfinite(got).all(), scored[f"error_{oracle}"].unique()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
