from pathlib import Path

import pandas as pd
import pytest
import yaml

from plum_ml1m.artifacts import ArtifactManifest, sha256_file
from plum_ml1m.config import ConfigError, validate_config
from plum_ml1m.splits import check_chronological_splits

ARTIFACT_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "artifacts"


def test_chronological_split_no_leakage_passes():
    train = pd.DataFrame({"user_idx": [1, 1, 2], "timestamp": [1, 2, 1], "pos": [0, 1, 0]})
    val = pd.DataFrame({"user_idx": [1, 2], "timestamp": [3, 2], "pos": [2, 1]})
    test = pd.DataFrame({"user_idx": [1, 2], "timestamp": [4, 3], "pos": [3, 2]})
    result = check_chronological_splits(train, val, test)
    assert result.ok
    result.raise_if_failed()


def test_chronological_split_detects_leakage():
    train = pd.DataFrame({"user_idx": [1], "timestamp": [5], "pos": [5]})
    val = pd.DataFrame({"user_idx": [1], "timestamp": [4], "pos": [4]})
    test = pd.DataFrame({"user_idx": [1], "timestamp": [6], "pos": [6]})
    result = check_chronological_splits(train, val, test)
    assert not result.ok
    with pytest.raises(ValueError):
        result.raise_if_failed()


def test_config_validation(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "unit",
                "stage": "prepare-data",
                "paths": {
                    "raw_dir": "raw",
                    "processed_dir": "processed",
                    "train": "train.parquet",
                    "val": "val.parquet",
                    "test": "test.parquet",
                },
                "split": {
                    "user_col": "user_idx",
                    "item_col": "item_idx",
                    "time_col": "timestamp",
                    "pos_col": "pos",
                    "protocol": "chronological_leave_one_out",
                },
            }
        ),
        encoding="utf-8",
    )
    assert validate_config(path)["name"] == "unit"

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "bad"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        validate_config(bad)

    invalid_eval = tmp_path / "invalid_eval.yaml"
    invalid_eval.write_text(
        yaml.safe_dump(
            {
                "name": "bad_eval",
                "stage": "evaluate",
                "paths": {
                    "model_or_adapter": "adapter",
                    "split": "data/processed/splits/val.parquet",
                    "sid_array": "sids.npy",
                    "predictions": "preds.parquet",
                    "metrics": "metrics.json",
                },
                "evaluation": {
                    "split_name": "test",
                    "target_id_space": "original item_idx",
                    "constrained_decoding": "trie",
                    "trie_constrained": True,
                    "collision_policy": "expand",
                    "filter_seen": True,
                    "k_values": [1, 10],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        validate_config(invalid_eval)


def test_artifact_manifest_loading_and_required_types(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    artifact_kinds = [
        "raw_movielens",
        "enriched_metadata",
        "movie_overviews",
        "embeddings",
        "sid_checkpoint",
        "sid_assignment_table",
        "cpt_lora_adapter",
        "sft_lora_adapter",
        "predictions",
        "metrics_report",
    ]
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "name": t,
                        "stage": "unit",
                        "kind": t,
                        "path": f"dummy/{t}",
                        "required_for": ["unit"],
                        "safe_to_commit": False,
                        "exists_required_for_ci": False,
                        "expected_type": "file_or_dir",
                    }
                    for t in artifact_kinds
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = ArtifactManifest.load(manifest_path)
    manifest.validate_schema()
    assert manifest.by_kind("metrics_report")[0].path == "dummy/metrics_report"


def test_artifact_local_check_and_checksum(tmp_path: Path):
    artifact = tmp_path / "tiny.txt"
    artifact.write_text("ok", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    kinds = [
        "raw_movielens",
        "enriched_metadata",
        "movie_overviews",
        "embeddings",
        "sid_checkpoint",
        "sid_assignment_table",
        "cpt_lora_adapter",
        "sft_lora_adapter",
        "predictions",
        "metrics_report",
    ]
    rows = []
    for kind in kinds:
        rows.append(
            {
                "name": kind,
                "stage": "unit",
                "kind": kind,
                "path": "tiny.txt",
                "required_for": ["unit"],
                "safe_to_commit": False,
                "exists_required_for_ci": False,
                "expected_type": "file",
                "sha256": sha256_file(artifact),
            }
        )
    manifest_path.write_text(yaml.safe_dump({"artifacts": rows}), encoding="utf-8")
    manifest = ArtifactManifest.load(manifest_path)
    assert manifest.validate_local(root=tmp_path) == []


def test_artifact_schema_only_check_does_not_require_files(tmp_path: Path):
    manifest_path = tmp_path / "manifest_valid.yaml"
    manifest_path.write_text(
        (ARTIFACT_FIXTURE_DIR / "manifest_valid.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest = ArtifactManifest.load(manifest_path)
    manifest.validate_schema()


def test_artifact_local_check_detects_missing_files(tmp_path: Path):
    manifest_path = tmp_path / "manifest_valid.yaml"
    manifest_path.write_text(
        (ARTIFACT_FIXTURE_DIR / "manifest_valid.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    errors = ArtifactManifest.load(manifest_path).validate_local(root=tmp_path)

    assert errors
    assert any("raw_movielens_1m: missing" in error for error in errors)


def test_artifact_local_check_validates_csv_and_parquet_columns():
    errors = ArtifactManifest.load(
        ARTIFACT_FIXTURE_DIR / "manifest_missing_column.yaml"
    ).validate_local(root=ARTIFACT_FIXTURE_DIR)

    assert any("missing_csv_column" in error and "available columns" in error for error in errors)
    assert any(
        "missing_parquet_column" in error and "available columns" in error for error in errors
    )


def test_artifact_local_check_validates_fixture_checksum(tmp_path: Path):
    valid = ArtifactManifest.load(ARTIFACT_FIXTURE_DIR / "manifest_valid.yaml")
    assert valid.validate_local(root=ARTIFACT_FIXTURE_DIR) == []

    manifest_path = tmp_path / "manifest_bad_checksum.yaml"
    data = yaml.safe_load((ARTIFACT_FIXTURE_DIR / "manifest_valid.yaml").read_text(encoding="utf-8"))
    data["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    errors = ArtifactManifest.load(manifest_path).validate_local(root=ARTIFACT_FIXTURE_DIR)
    assert any("sha256 mismatch" in error for error in errors)
