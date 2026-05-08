from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .protocol import ACTIVE_SID_PROTOCOL


class ConfigError(ValueError):
    """Raised for invalid experiment configuration files."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return data


def require_keys(config: dict[str, Any], keys: list[str], prefix: str = "") -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        label = f"{prefix}: " if prefix else ""
        raise ConfigError(f"{label}missing required keys: {missing}")


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_positive_int(value: Any, label: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean")


def _validate_lora(section: dict[str, Any], prefix: str) -> None:
    if section.get("adaptation") != "LoRA":
        raise ConfigError(f"{prefix}.adaptation must be LoRA")
    _require_positive_int(section.get("lora_r"), f"{prefix}.lora_r")


def _validate_sid_protocol(section: dict[str, Any], *, active_only: bool = True) -> None:
    require_keys(section, ["name", "levels", "codebook_sizes", "collision_policy"], "sid_protocol")
    levels = section["levels"]
    codebook_sizes = section["codebook_sizes"]
    _require_positive_int(levels, "sid_protocol.levels")
    if not isinstance(codebook_sizes, list) or not codebook_sizes:
        raise ConfigError("sid_protocol.codebook_sizes must be a non-empty list")
    if len(codebook_sizes) != levels:
        raise ConfigError("sid_protocol.codebook_sizes length must equal sid_protocol.levels")
    for index, size in enumerate(codebook_sizes):
        _require_positive_int(size, f"sid_protocol.codebook_sizes[{index}]")
    if active_only and section["name"] == ACTIVE_SID_PROTOCOL.name:
        expected = list(ACTIVE_SID_PROTOCOL.codebook_sizes)
        if levels != ACTIVE_SID_PROTOCOL.n_levels or codebook_sizes != expected:
            raise ConfigError(
                f"active {ACTIVE_SID_PROTOCOL.name} must use "
                f"{ACTIVE_SID_PROTOCOL.n_levels} levels and codebooks {expected}"
            )
    if section["collision_policy"] not in {"expand", "representative"}:
        raise ConfigError("sid_protocol.collision_policy must be expand or representative")


def _validate_prepare_data(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["raw_dir", "processed_dir", "train", "val", "test"], "paths")
    split = _require_mapping(config, "split")
    require_keys(split, ["user_col", "item_col", "time_col", "pos_col", "protocol"], "split")
    if split["protocol"] != "chronological_leave_one_out":
        raise ConfigError("split.protocol must be chronological_leave_one_out")


def _validate_metadata(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["movies_dat", "item_meta", "item_profiles", "overviews_csv"], "paths")
    policy = _require_mapping(config, "policy")
    _require_bool(policy.get("keep_missing_descriptions"), "policy.keep_missing_descriptions")


def _validate_embeddings(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["item_profiles", "embeddings", "manifest"], "paths")
    model = _require_mapping(config, "model")
    require_keys(model, ["name", "output_dim"], "model")
    _require_positive_int(model["output_dim"], "model.output_dim")
    require_keys(_require_mapping(config, "texts"), ["metadata", "description"], "texts")


def _validate_train_sid(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["embeddings", "output_dir", "sid_array", "sid_mapping"], "paths")
    if "behavior_pairs" not in paths:
        raise ConfigError("paths.behavior_pairs is required for behavior alignment")
    _validate_sid_protocol(_require_mapping(config, "sid_protocol"))
    model = _require_mapping(config, "model")
    _require_positive_int(model.get("approximate_parameters"), "model.approximate_parameters")


def _validate_train_cpt(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["train", "item_profiles", "sid_array", "output_dir"], "paths")
    model = _require_mapping(config, "model")
    require_keys(model, ["base", "adaptation", "lora_r"], "model")
    if not model["base"]:
        raise ConfigError("model.base must be non-empty")
    _validate_lora(model, "model")
    curriculum = _require_mapping(config, "curriculum")
    require_keys(
        curriculum,
        [
            "synthetic_epochs",
            "train_epochs",
            "behavior_ratio",
            "history_last_k",
            "uses_recsys_val_test_as_behavior",
        ],
        "curriculum",
    )
    _require_bool(
        curriculum["uses_recsys_val_test_as_behavior"],
        "curriculum.uses_recsys_val_test_as_behavior",
    )


def _validate_train_sft(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(paths, ["base_cpt", "train", "val", "test", "sid_array", "output_dir"], "paths")
    task = _require_mapping(config, "task")
    require_keys(task, ["target", "history_window", "target_only_loss", "filter_seen"], "task")
    if task["target"] != "next_watched_item":
        raise ConfigError("task.target must be next_watched_item")
    _require_positive_int(task["history_window"], "task.history_window")
    _require_bool(task["target_only_loss"], "task.target_only_loss")
    _require_bool(task["filter_seen"], "task.filter_seen")
    model = _require_mapping(config, "model")
    require_keys(model, ["adaptation", "lora_r", "beam_size", "top_k"], "model")
    _validate_lora(model, "model")
    _require_positive_int(model["beam_size"], "model.beam_size")
    _require_positive_int(model["top_k"], "model.top_k")


def _validate_evaluate(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    require_keys(
        paths, ["model_or_adapter", "split", "sid_array", "predictions", "metrics"], "paths"
    )
    evaluation = _require_mapping(config, "evaluation")
    require_keys(
        evaluation,
        [
            "split_name",
            "target_id_space",
            "constrained_decoding",
            "collision_policy",
            "filter_seen",
            "k_values",
            "trie_constrained",
        ],
        "evaluation",
    )
    split_name = evaluation["split_name"]
    if split_name not in {"val", "test"}:
        raise ConfigError("evaluation.split_name must be val or test")
    split_path = str(paths["split"]).replace("\\", "/")
    if f"/{split_name}.parquet" not in split_path and not split_path.endswith(
        f"{split_name}.parquet"
    ):
        raise ConfigError("paths.split must match evaluation.split_name")
    if evaluation["constrained_decoding"] != "trie":
        raise ConfigError("evaluation.constrained_decoding must be trie")
    _require_bool(evaluation["trie_constrained"], "evaluation.trie_constrained")
    _require_bool(evaluation["filter_seen"], "evaluation.filter_seen")
    if evaluation["collision_policy"] not in {"expand", "representative"}:
        raise ConfigError("evaluation.collision_policy must be expand or representative")
    if "item_idx" not in str(evaluation["target_id_space"]):
        raise ConfigError("evaluation.target_id_space must explicitly mention item_idx")
    k_values = evaluation["k_values"]
    if not isinstance(k_values, list) or not k_values:
        raise ConfigError("evaluation.k_values must be a non-empty list")
    if len(set(k_values)) != len(k_values):
        raise ConfigError("evaluation.k_values must be unique")
    for index, value in enumerate(k_values):
        _require_positive_int(value, f"evaluation.k_values[{index}]")


STAGE_VALIDATORS = {
    "prepare-data": _validate_prepare_data,
    "build-metadata": _validate_metadata,
    "build-embeddings": _validate_embeddings,
    "train-sid": _validate_train_sid,
    "train-cpt": _validate_train_cpt,
    "train-sft": _validate_train_sft,
    "evaluate": _validate_evaluate,
}


def validate_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_keys(config, ["name", "stage", "paths"])
    _require_mapping(config, "paths")
    stage = str(config["stage"])
    if stage in STAGE_VALIDATORS:
        STAGE_VALIDATORS[stage](config)
    return config


def validate_config_dir(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    config_dir = Path(config_dir)
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.rglob("*.yaml")):
        results[str(path.relative_to(config_dir))] = validate_config(path)
    if not results:
        raise ConfigError(f"No YAML configs found in {config_dir}")
    return results
