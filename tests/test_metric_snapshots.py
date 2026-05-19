import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "reports" / "snapshots"
REQUIRED_TOP_LEVEL = {
    "run_id",
    "method",
    "method_family",
    "split",
    "dataset",
    "git_commit",
    "source",
    "protocol",
    "model",
    "metrics",
    "diagnostics",
    "artifact_policy",
    "limitations",
}
REQUIRED_METRICS = {
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "mrr@10",
    "coverage@10",
}


def test_metric_snapshots_have_expected_schema():
    paths = sorted(SNAPSHOT_DIR.glob("*.metrics.json"))
    assert paths

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED_TOP_LEVEL <= set(data)
        assert data["split"] in {"validation", "test"}
        assert data["dataset"] == "MovieLens-1M"
        assert REQUIRED_METRICS <= set(data["metrics"])
        assert isinstance(data["limitations"], list)

        for value in data["metrics"].values():
            assert value is None or isinstance(value, (int, float))

        policy = data["artifact_policy"]
        assert policy["predictions_committed"] is False
        assert policy["checkpoints_committed"] is False
        assert policy["large_artifacts_committed"] is False


def test_metric_summary_csv_files_are_readable():
    expected_columns = [
        "run_id",
        "method",
        "method_family",
        "split",
        "sid_codebooks",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@10",
        "mrr@10",
        "coverage@10",
        "seen_filter_scope",
        "context_source",
        "notes",
    ]

    for name in ["summary_test.csv", "summary_validation.csv"]:
        path = SNAPSHOT_DIR / name
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert list(rows[0].keys()) == expected_columns


def test_baseline_snapshot_names_are_explicit_about_adaptations():
    bert = json.loads(
        (SNAPSHOT_DIR / "bert4rec_style_multimodal_test.metrics.json").read_text(
            encoding="utf-8"
        )
    )
    sas = json.loads(
        (SNAPSHOT_DIR / "sasrec_multimodal_test.metrics.json").read_text(encoding="utf-8")
    )
    no_cpt = json.loads(
        (SNAPSHOT_DIR / "qwen3_sft_only_validation.metrics.json").read_text(encoding="utf-8")
    )

    assert "BERT4Rec-style" in bert["method"]
    assert "Qwen content" in bert["method"]
    assert "random-mask BERT4Rec" in " ".join(bert["limitations"])
    assert "SASRec-style" in sas["method"]
    assert "Qwen content" in sas["method"]
    assert "ID-only SASRec" in " ".join(sas["limitations"])
    assert no_cpt["split"] == "validation"
    assert "no held-out test" in " ".join(no_cpt["limitations"])
