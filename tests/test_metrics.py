import math

import numpy as np
import pandas as pd
import pytest

from accord import metrics as M


def slots(rows, target="t1"):
    """Build a slot table from rows of (vienna, nupack, eternafold, contrafold)."""
    df = pd.DataFrame(rows, columns=["mcc_vienna", "mcc_nupack", "mcc_eternafold", "mcc_contrafold"])
    df.insert(0, "target_id", target)
    return df


PANEL = ["vienna", "nupack", "eternafold", "contrafold"]


def test_disagreement_is_sample_sd():
    assert M.disagreement([0.9, 0.5, 0.7]) == pytest.approx(0.2)


def test_identical_scores_give_exactly_zero():
    # 0.1 + 0.2-style values can leave a ~1e-17 residue that breaks ties in ranks.
    scores = [0.8571428571428571, 0.8571428571428572, 0.8571428571428571]
    assert M.disagreement(scores) == 0.0
    df = slots([scores + [0.8571428571428571]])
    assert M.design_table(df, PANEL)["d"].iloc[0] == 0.0


def test_disagreement_requires_complete_scores():
    assert math.isnan(M.disagreement([0.9, np.nan, 0.7]))
    assert math.isnan(M.disagreement([0.9]))


def test_family_disagreement():
    s = {"vienna": 0.9, "nupack": 0.7, "eternafold": 0.5, "contrafold": 0.3}
    assert M.family_disagreement(s) == pytest.approx(0.4 / math.sqrt(2))
    assert math.isnan(M.family_disagreement({"vienna": 0.9, "nupack": 0.7}))


@pytest.mark.parametrize("d, tier", [(0.0, "High"), (0.0499, "High"), (0.05, "Moderate"),
                                     (0.15, "Moderate"), (0.1501, "Low"), (0.30, "Low"),
                                     (0.45, "Low")])
def test_tier_cutoffs(d, tier):
    # There is no extra "Unreliable" class above 0.25: large disagreement is Low.
    assert M.assign_tier(d, max_mcc=0.9) == tier


def test_gate_and_incomplete():
    assert M.assign_tier(0.01, max_mcc=0.49) == M.TIER_GATED
    assert M.assign_tier(float("nan"), max_mcc=0.9) == M.TIER_INCOMPLETE
    with pytest.raises(ValueError):
        M.assign_tier(-0.1, max_mcc=0.9)


def test_tiers_use_only_the_tier_panel():
    # NUPACK disagrees strongly, but tiers are defined on ViennaRNA/EternaFold/CONTRAfold.
    s = {"vienna": 0.90, "nupack": 0.10, "eternafold": 0.91, "contrafold": 0.92}
    assert M.tier_from_scores(s) == "High"
    s.pop("contrafold")
    assert M.tier_from_scores(s) == M.TIER_INCOMPLETE


def test_failures_stay_in_the_denominator():
    df = slots([[0.9, 0.9, 0.9, 0.9],          # pass all
                [0.9, 0.3, 0.9, 0.9],          # pass some
                [0.2, 0.2, 0.2, 0.2],          # fail all
                [0.9, np.nan, 0.9, 0.9]])      # incomplete (oracle failure)
    s = M.summarize(df, PANEL, target_col=None)
    assert s["n_slots"] == 4 and s["n_incomplete"] == 1
    assert s["C_complete"] == pytest.approx(0.75)
    assert s["C_any"] == pytest.approx(0.50)
    assert s["Y_all"] == pytest.approx(0.25)
    assert (s["n_pass_all"], s["n_pass_some_not_all"], s["n_fail_all"]) == (1, 1, 1)
    assert s["oracle_dependent_conditional"] == pytest.approx(0.5)
    assert s["oracle_dependent_raw"] == pytest.approx(0.25)
    # continuous quantities use complete slots only
    assert s["Q_mean"] == pytest.approx(np.mean([0.9, 0.75, 0.2]))
    assert s["Q_worst"] == pytest.approx(np.mean([0.9, 0.3, 0.2]))


def test_gated_disagreement_excludes_shared_failures():
    df = slots([[0.9, 0.9, 0.9, 0.9], [0.1, 0.4, 0.1, 0.4]])
    s = M.summarize(df, PANEL, target_col=None)
    assert s["D_cond"] == pytest.approx(0.0)
    assert s["D_raw"] > s["D_cond"]


def test_target_weighting_gives_equal_weight():
    a = slots([[0.9] * 4] * 9, target="a")         # 9 passing slots
    b = slots([[0.1] * 4], target="b")             # 1 failing slot
    df = pd.concat([a, b], ignore_index=True)
    assert M.summarize(df, PANEL)["Y_all"] == pytest.approx(0.5)
    assert M.summarize(df, PANEL, target_col=None)["Y_all"] == pytest.approx(0.9)


def test_tier_counts_cover_every_label():
    df = slots([[0.9, 0.9, 0.9, 0.9], [0.2, 0.2, 0.2, 0.2], [0.9, 0.9, np.nan, 0.9]])
    counts = M.tier_counts(df)
    assert set(counts) == set(M.TIER_LABELS)
    assert counts["High"] == 1 and counts[M.TIER_GATED] == 1 and counts[M.TIER_INCOMPLETE] == 1


def test_reference_ratio_needs_shared_targets():
    designed = slots([[0.9, 0.5, 0.9, 0.5]], target="a")
    reference = slots([[0.9, 0.8, 0.9, 0.8]], target="b")
    assert M.reference_ratio(designed, reference, PANEL)["ratio"] is None
    reference["target_id"] = "a"
    r = M.reference_ratio(designed, reference, PANEL)
    assert r["n_shared_targets"] == 1 and r["ratio"] > 2 and r["flag"]
