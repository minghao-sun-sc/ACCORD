import json

import numpy as np
import pandas as pd
import pytest

import accord
from accord import oracles as O
from accord import protocol as P

DESIGNS = pd.DataFrame({
    "design_id": ["a1", "a2", "b1", "b2"],
    "target_id": ["A", "A", "B", "B"],
    "sequence": ["GGGAAACCC", "GGGAAACCC", "GGGGAAACCCC", ""],   # b2: failed generation
    "target_structure": ["(((...)))", "(((...)))", "((((...))))", "((((...))))"],
})


def fake_fold(seq, oracle, nupack_config="canonical"):
    """Deterministic stand-in for the real oracles; NUPACK is 'not installed'."""
    seq = O.normalise_sequence(seq)
    if oracle == "nupack":
        return O.FoldResult("nupack", seq, None, ok=False, error="nupack not importable")
    n = len(seq)
    k = n // 3
    hairpin = "(" * k + "." * (n - 2 * k) + ")" * k
    return O.FoldResult(oracle, seq, hairpin if oracle != "contrafold" else "." * n)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(P.O, "fold", fake_fold)


def test_missing_backend_marks_slots_incomplete(patched):
    report = accord.evaluate(DESIGNS)
    common = report.summary["views"]["common_panel"]
    # The panel is never silently reduced to the oracles that happen to be installed.
    assert common["panel"] == list(O.PANEL_4)
    assert common["C_complete"] == 0.0 and common["Y_all"] == 0.0
    assert report.per_design["mcc_nupack"].isna().all()


def test_failed_generation_is_kept(patched):
    report = accord.evaluate(DESIGNS, panel=["vienna", "eternafold"])
    s = report.summary["views"]["common_panel"]
    assert s["n_slots"] == 4 and s["n_incomplete"] == 1
    assert s["C_complete"] == pytest.approx((1.0 + 0.5) / 2)    # target-weighted


def test_leave_design_oracle_out_view(patched):
    report = accord.evaluate(DESIGNS, panel=["vienna", "eternafold", "contrafold"],
                             design_oracle="vienna")
    views = report.summary["views"]
    assert views["leave_design_oracle_out"]["panel"] == ["eternafold", "contrafold"]
    assert views["common_panel"]["panel"] == ["vienna", "eternafold", "contrafold"]


def test_tiers_are_reported_on_the_tier_panel(patched):
    report = accord.evaluate(DESIGNS, panel=["vienna", "eternafold"])
    assert report.summary["tiers"]["panel"] == list(O.TIER_PANEL)
    assert sum(report.summary["tiers"]["counts"].values()) == len(DESIGNS)


def test_report_is_valid_json_without_nan(patched, tmp_path):
    report = accord.evaluate(DESIGNS)
    report.save(tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert "NaN" not in (tmp_path / "report.json").read_text()
    assert data["accord_version"] == accord.__version__
    assert (tmp_path / "per_design.csv").exists()


def test_report_from_scores_without_oracles():
    scores = pd.DataFrame({"target_id": ["A", "A", "B"],
                           "mcc_vienna": [0.9, 0.8, np.nan],
                           "mcc_nupack": [0.9, 0.4, 0.7],
                           "mcc_eternafold": [0.9, 0.9, 0.7],
                           "mcc_contrafold": [0.8, 0.9, 0.6]})
    report = accord.report_from_scores(scores, design_oracle="nupack")
    lodo = report.summary["views"]["leave_design_oracle_out"]
    common = report.summary["views"]["common_panel"]
    assert lodo["panel"] == ["vienna", "eternafold", "contrafold"]
    assert lodo["Y_all"] >= common["Y_all"]
    assert "d_family" in report.per_design.columns


def test_unknown_oracle_is_rejected():
    with pytest.raises(ValueError):
        accord.score(DESIGNS, oracles=["vienna", "rnastructure"])


def test_missing_columns_are_rejected():
    with pytest.raises(KeyError):
        accord.score(DESIGNS.drop(columns=["target_structure"]))


def test_availability_reports_every_oracle():
    status = accord.availability()
    assert set(status) == set(O.SUPPORTED)
    assert all(isinstance(s["available"], bool) for s in status.values())


def test_existing_scores_are_replaced_not_duplicated(patched):
    stale = DESIGNS.assign(mcc_vienna=0.1, mcc_eternafold=0.1, mcc_contrafold=0.1,
                           structure_vienna="x", error_vienna="old")
    with pytest.warns(UserWarning, match="ignoring existing result columns"):
        report = accord.evaluate(stale, panel=["vienna", "eternafold", "contrafold"])
    per = report.per_design
    assert not per.columns.duplicated().any()
    fresh = accord.evaluate(DESIGNS, panel=["vienna", "eternafold", "contrafold"]).per_design
    pd.testing.assert_series_equal(per["d"], fresh["d"])
    assert (per["tier"] == fresh["tier"]).all()


def test_tiers_use_the_calibrated_gate_whatever_the_threshold():
    scores = pd.DataFrame({"target_id": ["A"], "mcc_vienna": [0.6], "mcc_eternafold": [0.6],
                           "mcc_contrafold": [0.6]})
    panel = ["vienna", "eternafold", "contrafold"]
    strict = accord.report_from_scores(scores, panel, threshold=0.7)
    assert strict.per_design["tier"].iloc[0] == "High"
    assert strict.summary["views"]["common_panel"]["Y_all"] == 0.0


def test_malformed_target_structure_is_a_failed_slot(patched):
    bad = DESIGNS.copy()
    bad.loc[0, "target_structure"] = "(((xxx)))"
    bad.loc[1, "target_structure"] = "(((...))."
    per = accord.score(bad, oracles=["vienna"])
    assert per["mcc_vienna"].iloc[:2].isna().all()
    assert "dot-bracket" in per["error_vienna"].iloc[0]


def test_missing_target_id_is_rejected():
    with pytest.raises(ValueError, match="target_id"):
        accord.score(DESIGNS.assign(target_id=[None, "A", "B", "B"]), oracles=["vienna"])
    scores = pd.DataFrame({"target_id": ["A", None], "mcc_vienna": [0.9, 0.1],
                           "mcc_eternafold": [0.9, 0.1], "mcc_contrafold": [0.9, 0.1]})
    with pytest.raises(ValueError, match="target_id"):
        accord.summarize(scores, ["vienna", "eternafold", "contrafold"])


def test_invalid_panels_and_thresholds_are_rejected():
    scores = pd.DataFrame({"target_id": ["A"], "mcc_vienna": [0.9], "mcc_eternafold": [0.9],
                           "mcc_contrafold": [0.9]})
    with pytest.raises(ValueError, match="more than once"):
        accord.report_from_scores(scores, ["vienna", "vienna", "eternafold"])
    with pytest.raises(ValueError, match="threshold"):
        accord.summarize(scores, ["vienna", "eternafold"], threshold=5)
    with pytest.raises(KeyError, match="target_id"):
        accord.report_from_scores(scores.drop(columns="target_id"), ["vienna", "eternafold"])


def test_misconfigured_backend_path_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("ACCORD_CONTRAFOLD", str(tmp_path / "missing" / "contrafold"))
    status = accord.availability()["contrafold"]
    assert not status["available"] and "does not exist" in status["detail"]


def test_unknown_nupack_configuration_raises():
    with pytest.raises(ValueError, match="unknown NUPACK configuration"):
        O.fold_nupack("GGGAAACCC", "rna2000")


def test_nupack_configs_have_one_primary():
    cfgs = O.nupack_configs()
    assert [n for n, c in cfgs.items() if c.primary] == ["canonical"]
    assert {"rna99", "wc_only", "celsius_25", "mg_0p01"} <= set(cfgs)
