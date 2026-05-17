import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from plum_ml1m.eval.report import EvaluationReportError, build_evaluation_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tiny_eval_report"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def test_eval_report_recomputes_metrics_from_predictions(tmp_path):
    output = tmp_path / "report.json"
    report = build_evaluation_report(
        config_path=FIXTURE / "config.yaml",
        predictions_path=FIXTURE / "predictions.csv",
        targets_path=FIXTURE / "targets.csv",
        seen_history_path=FIXTURE / "seen_history.csv",
        output_path=output,
        run_id="tiny_eval_report",
    )
    expected = json.loads((FIXTURE / "expected_report.json").read_text(encoding="utf-8"))

    assert output.exists()
    assert report["split"] == expected["split"]
    assert report["dataset"] == expected["dataset"]
    assert report["metrics"] == expected["metrics"]

    for key, value in expected["diagnostics"].items():
        assert report["diagnostics"][key] == pytest.approx(value)

    assert report["artifact_policy"]["predictions_committed"] is False
    assert report["artifact_policy"]["checkpoints_committed"] is False


def test_eval_report_cli_runs_on_fixture(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plum_ml1m.cli",
            "eval-report",
            "--config",
            str(FIXTURE / "config.yaml"),
            "--predictions",
            str(FIXTURE / "predictions.csv"),
            "--targets",
            str(FIXTURE / "targets.csv"),
            "--seen-history",
            str(FIXTURE / "seen_history.csv"),
            "--output",
            str(output),
            "--run-id",
            "tiny_eval_report",
        ],
        cwd=ROOT,
        env=_cli_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert output.exists()


def test_eval_report_rejects_missing_prediction_columns(tmp_path):
    bad_predictions = tmp_path / "bad_predictions.csv"
    bad_predictions.write_text("user_idx,rank\n1,1\n", encoding="utf-8")

    with pytest.raises(EvaluationReportError, match="predicted_item_indices"):
        build_evaluation_report(
            config_path=FIXTURE / "config.yaml",
            predictions_path=bad_predictions,
            targets_path=FIXTURE / "targets.csv",
            seen_history_path=FIXTURE / "seen_history.csv",
            output_path=tmp_path / "report.json",
        )


def test_eval_report_accepts_item_idx_seen_history_alias(tmp_path):
    seen_history = tmp_path / "seen_history.csv"
    seen_history.write_text("user_idx,item_idx\n1,10\n2,70\n", encoding="utf-8")

    report = build_evaluation_report(
        config_path=FIXTURE / "config.yaml",
        predictions_path=FIXTURE / "predictions.csv",
        targets_path=FIXTURE / "targets.csv",
        seen_history_path=seen_history,
        output_path=tmp_path / "report.json",
    )

    assert report["metrics"]["recall@10"] == pytest.approx(0.5)
    assert report["diagnostics"]["seen_items_filtered"] == 2


def test_eval_report_accepts_candidates_prediction_alias(tmp_path):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        'user_idx,target_item_idx,candidates\n1,20,"[10, 20, 20]"\n',
        encoding="utf-8",
    )

    seen_history = tmp_path / "seen_history.csv"
    seen_history.write_text("user_idx,seen_item_idx\n1,10\n", encoding="utf-8")

    report = build_evaluation_report(
        config_path=FIXTURE / "config.yaml",
        predictions_path=predictions,
        seen_history_path=seen_history,
        output_path=tmp_path / "report.json",
    )

    assert report["metrics"]["recall@1"] == pytest.approx(1.0)
    assert report["diagnostics"]["duplicate_predictions"] == 1
    assert report["diagnostics"]["seen_items_filtered"] == 1
