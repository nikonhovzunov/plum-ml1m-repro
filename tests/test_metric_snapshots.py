import csv
import json
import math
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
    "provenance",
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
SUMMARY_COLUMNS = [
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
SUMMARY_NUMERIC_FIELDS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "mrr@10",
    "coverage@10",
]


def _load_metric_snapshot(run_id: str) -> dict:
    path = SNAPSHOT_DIR / f"{run_id}.metrics.json"
    assert path.exists(), f"Missing metric snapshot for summary row: {run_id}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary_rows(name: str) -> list[dict[str, str]]:
    path = SNAPSHOT_DIR / name
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert list(rows[0].keys()) == SUMMARY_COLUMNS
    return rows


def _assert_csv_metric_matches_json(row: dict[str, str], snapshot: dict) -> None:
    for field in SUMMARY_NUMERIC_FIELDS:
        csv_value = row[field]
        json_value = snapshot["metrics"].get(field)

        if csv_value == "":
            assert json_value is None, (row["run_id"], field, json_value)
            continue

        assert json_value is not None, (row["run_id"], field)
        if field == "coverage@10":
            assert int(csv_value) == json_value
        else:
            assert math.isclose(
                float(csv_value),
                float(json_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ), (row["run_id"], field, csv_value, json_value)


def test_metric_snapshots_have_expected_schema():
    paths = sorted(SNAPSHOT_DIR.glob("*.metrics.json"))
    assert paths

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED_TOP_LEVEL <= set(data)
        assert data["split"] in {"validation", "test"}
        assert data["dataset"] == "MovieLens-1M"
        assert data["git_commit"]
        assert REQUIRED_METRICS <= set(data["metrics"])
        assert isinstance(data["limitations"], list)

        provenance = data["provenance"]
        assert provenance["snapshot_commit"]
        assert provenance["protocol_commit"]
        assert isinstance(provenance["source_artifact_available_in_git"], bool)
        if provenance["source_run_commit"] is None:
            assert provenance["source_run_commit_note"].strip()

        for value in data["metrics"].values():
            assert value is None or isinstance(value, (int, float))

        policy = data["artifact_policy"]
        assert policy["predictions_committed"] is False
        assert policy["checkpoints_committed"] is False
        assert policy["large_artifacts_committed"] is False


def test_qwen_and_sid_snapshots_have_sid_protocol():
    paths = sorted(SNAPSHOT_DIR.glob("*.metrics.json"))
    sid_paths = [
        path
        for path in paths
        if path.name.startswith(("qwen", "sid_"))
        or json.loads(path.read_text(encoding="utf-8"))["method_family"]
        == "plum_style_generative"
    ]
    assert sid_paths

    for path in sid_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "sid_protocol" in data
        protocol = data["sid_protocol"]
        assert protocol["name"]
        if protocol["n_levels"] is not None:
            assert protocol["n_levels"] == len(protocol["codebook_sizes"])
            assert all(size > 0 for size in protocol["codebook_sizes"])
        else:
            assert protocol["codebook_sizes"] is None
            assert "not normalized" in protocol.get("notes", "")


def test_qwen_peft_and_fullft_snapshots_have_adaptation_details():
    expected = {
        "qwen3_0_6b_fullft_test.metrics.json": "full_ft",
        "qwen3_0_6b_qlora32_test.metrics.json": "QLoRA32",
        "qwen3_0_6b_qlora32_3levels_test.metrics.json": "QLoRA32",
        "qwen3_4b_qlora32_3levels_test.metrics.json": "QLoRA32",
        "qwen3_lora16_test.metrics.json": "LoRA16",
        "qwen3_qlora32_test.metrics.json": "QLoRA32",
        "qwen3_qlora32_validation.metrics.json": "QLoRA32",
        "qwen3_sft_only_validation.metrics.json": "LoRA",
    }

    for name, expected_type in expected.items():
        data = json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))
        adaptation = data["adaptation"]
        assert adaptation["type"] == expected_type
        assert adaptation["base_model"].startswith("Qwen3")
        assert adaptation["cpt"]
        assert adaptation["sft"]
        assert adaptation["notes"].strip()


def test_diagnostic_snapshots_have_provenance_and_do_not_commit_dumps():
    for path in sorted(SNAPSHOT_DIR.glob("*.diagnostics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["git_commit"]
        assert "provenance" in data
        assert data["provenance"]["snapshot_commit"]
        assert data["provenance"]["protocol_commit"]
        if data["provenance"]["source_run_commit"] is None:
            assert data["provenance"]["source_run_commit_note"].strip()
        assert data["artifact_policy"]["large_artifacts_committed"] is False
        assert data["artifact_policy"]["prediction_dumps_committed"] is False


def test_metric_summary_csv_files_are_readable():
    for name in ["summary_test.csv", "summary_validation.csv"]:
        _read_summary_rows(name)


def test_summary_csv_metrics_match_json_snapshots():
    for name in ["summary_test.csv", "summary_validation.csv"]:
        for row in _read_summary_rows(name):
            snapshot = _load_metric_snapshot(row["run_id"])
            assert row["method"] == snapshot["method"]
            assert row["method_family"] == snapshot["method_family"]
            assert row["split"] == snapshot["split"]
            assert row["seen_filter_scope"] == snapshot["protocol"]["seen_filter_scope"]
            assert row["context_source"] == snapshot["protocol"]["context_source"]
            _assert_csv_metric_matches_json(row, snapshot)


def test_snapshot_file_names_match_recorded_splits():
    for path in sorted(SNAPSHOT_DIR.glob("*.metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name.endswith("_test.metrics.json"):
            assert data["split"] == "test", path.name
        if path.name.endswith("_validation.metrics.json"):
            assert data["split"] == "validation", path.name

    for row in _read_summary_rows("summary_test.csv"):
        assert row["split"] == "test"
    for row in _read_summary_rows("summary_validation.csv"):
        assert row["split"] == "validation"


def test_snapshots_never_claim_large_artifacts_are_committed():
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = data["artifact_policy"]
        assert policy.get("large_artifacts_committed") is False, path.name
        assert policy.get("predictions_committed", False) is False, path.name
        assert policy.get("prediction_dumps_committed", False) is False, path.name
        assert policy.get("checkpoints_committed", False) is False, path.name


def test_docs_do_not_contain_strong_overclaim_phrases():
    forbidden = [
        "PEFT is always better",
        "QLoRA " + "always" + " beats full fine-tuning",
        "LLMs " + "beat" + " SASRec",
    ]
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, (path, phrase)


def test_baseline_snapshot_names_use_public_method_labels():
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

    assert bert["method"] == "BERT4Rec"
    assert "random-mask BERT4Rec" in " ".join(bert["limitations"])
    assert sas["method"] == "SASRec"
    assert "ID-only SASRec" in " ".join(sas["limitations"])
    assert no_cpt["split"] == "validation"
    assert "no held-out test" in " ".join(no_cpt["limitations"])
