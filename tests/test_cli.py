import json

import pandas as pd

from accord.cli import main


def test_summarize_command(tmp_path):
    scores = pd.DataFrame({"target_id": ["A", "A", "B"],
                           "mcc_vienna": [0.9, 0.8, 0.2],
                           "mcc_nupack": [0.9, 0.4, 0.3],
                           "mcc_eternafold": [0.9, 0.9, 0.2],
                           "mcc_contrafold": [0.8, 0.9, 0.1]})
    src = tmp_path / "scores.csv"
    scores.to_csv(src, index=False)
    out = tmp_path / "out"
    assert main(["summarize", str(src), "--design-oracle", "nupack", "--out-dir", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert set(report["views"]) == {"common_panel", "leave_design_oracle_out"}
    assert report["views"]["common_panel"]["n_slots"] == 3


def test_input_errors_are_reported_without_traceback(tmp_path, capsys):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"sequence": ["GGGAAACCC"]}).to_csv(src, index=False)
    assert main(["evaluate", str(src), "--out-dir", str(tmp_path / "out")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_target_ids_are_read_as_text(tmp_path):
    scores = pd.DataFrame({"target_id": ["007", "007"], "mcc_vienna": [0.9, 0.8],
                           "mcc_eternafold": [0.9, 0.9], "mcc_contrafold": [0.9, 0.7]})
    src = tmp_path / "scores.csv"
    scores.to_csv(src, index=False)
    out = tmp_path / "out"
    assert main(["summarize", str(src), "--panel", "vienna", "eternafold", "contrafold",
                 "--out-dir", str(out)]) == 0
    per = pd.read_csv(out / "per_design.csv", dtype={"target_id": str})
    assert set(per["target_id"]) == {"007"}


def test_oracles_command(capsys):
    assert main(["oracles"]) == 0
    assert "vienna" in capsys.readouterr().out
