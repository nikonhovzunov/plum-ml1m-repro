from pathlib import Path

import pandas as pd
import pytest
import yaml

from plum_ml1m.artifacts import ArtifactManifest
from plum_ml1m.config import ConfigError, validate_config
from plum_ml1m.splits import check_chronological_splits


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
        yaml.safe_dump({"name": "unit", "stage": "evaluate", "paths": {"metrics": "x.json"}}),
        encoding="utf-8",
    )
    assert validate_config(path)["name"] == "unit"

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "bad"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_artifact_manifest_loading_and_required_types(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    artifact_types = [
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
                    {"type": t, "path": f"dummy/{t}", "description": t, "tracked_by_git": False}
                    for t in artifact_types
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = ArtifactManifest.load(manifest_path)
    manifest.validate_required_types()
    assert manifest.by_type("metrics_report")[0].path == "dummy/metrics_report"
