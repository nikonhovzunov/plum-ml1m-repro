from __future__ import annotations

import hashlib
import json
from csv import reader as csv_reader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_ARTIFACT_KINDS = {
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
}


class ArtifactError(ValueError):
    """Raised for invalid or missing artifact manifest entries."""


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    stage: str
    kind: str
    path: str
    required_for: list[str]
    produced_by: str | None = None
    consumed_by: list[str] = field(default_factory=list)
    safe_to_commit: bool = False
    exists_required_for_ci: bool = False
    expected_type: str = "file_or_dir"
    expected_size_bytes: int | None = None
    sha256: str | None = None
    schema: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ArtifactRecord:
        required = [
            "name",
            "stage",
            "kind",
            "path",
            "required_for",
            "safe_to_commit",
            "exists_required_for_ci",
            "expected_type",
        ]
        missing = [key for key in required if key not in row]
        if missing:
            raise ArtifactError(f"artifact record is missing fields: {missing}")
        return cls(**row)

    def resolved_path(self, root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ArtifactManifest:
    name: str
    artifacts: list[ArtifactRecord]
    categories: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        rows = data.get("artifacts", [])
        if not isinstance(rows, list):
            raise ArtifactError("artifact manifest field 'artifacts' must be a list")
        name = str(data.get("name", "artifact_manifest"))
        categories = data.get("categories", {})
        if categories and not isinstance(categories, dict):
            raise ArtifactError("artifact manifest field 'categories' must be a mapping")
        artifacts = [ArtifactRecord.from_dict(row) for row in rows]
        return cls(name=name, artifacts=artifacts, categories=categories)

    @classmethod
    def load(cls, path: str | Path) -> ArtifactManifest:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ArtifactError(f"manifest must be a mapping: {path}")
        return cls.from_dict(data)

    def validate_schema(self) -> None:
        names = [artifact.name for artifact in self.artifacts]
        if len(set(names)) != len(names):
            raise ArtifactError("artifact names must be unique")

        present_kinds = {artifact.kind for artifact in self.artifacts}
        missing_kinds = sorted(REQUIRED_ARTIFACT_KINDS - present_kinds)
        if missing_kinds:
            raise ArtifactError(f"Artifact manifest is missing kinds: {missing_kinds}")

        for artifact in self.artifacts:
            if artifact.expected_type not in {"file", "dir", "file_or_dir"}:
                raise ArtifactError(
                    f"{artifact.name}: expected_type must be file, dir, or file_or_dir"
                )
            if artifact.safe_to_commit:
                raise ArtifactError(
                    f"{artifact.name}: generated artifacts must not be safe_to_commit"
                )
            if artifact.expected_size_bytes is not None and artifact.expected_size_bytes < 0:
                raise ArtifactError(f"{artifact.name}: expected_size_bytes must be non-negative")
            if artifact.sha256 is not None and len(artifact.sha256) != 64:
                raise ArtifactError(f"{artifact.name}: sha256 must be a 64-character hex digest")
            if not isinstance(artifact.schema, dict):
                raise ArtifactError(f"{artifact.name}: schema must be a mapping")
            for key in ("columns", "files", "arrays"):
                if key in artifact.schema and not isinstance(artifact.schema[key], list):
                    raise ArtifactError(f"{artifact.name}: schema.{key} must be a list")

    def validate_local(self, root: str | Path = ".") -> list[str]:
        self.validate_schema()
        root = Path(root)
        errors: list[str] = []
        for artifact in self.artifacts:
            path = artifact.resolved_path(root)
            if not path.exists():
                errors.append(f"{artifact.name}: missing {path}")
                continue
            if artifact.expected_type == "file" and not path.is_file():
                errors.append(f"{artifact.name}: expected file, got {path}")
            if artifact.expected_type == "dir" and not path.is_dir():
                errors.append(f"{artifact.name}: expected directory, got {path}")
            if artifact.expected_size_bytes is not None and path.is_file():
                actual_size = path.stat().st_size
                if actual_size != artifact.expected_size_bytes:
                    errors.append(
                        f"{artifact.name}: size mismatch {actual_size} != "
                        f"{artifact.expected_size_bytes}"
                    )
            if artifact.sha256 and path.is_file():
                actual = sha256_file(path)
                if actual != artifact.sha256:
                    errors.append(f"{artifact.name}: sha256 mismatch {actual} != {artifact.sha256}")
            errors.extend(validate_artifact_schema_columns(artifact, path))
            errors.extend(validate_artifact_schema_files(artifact, path))
        return errors

    def by_kind(self, kind: str) -> list[ArtifactRecord]:
        return [artifact for artifact in self.artifacts if artifact.kind == kind]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_schema_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ArtifactError("schema entries must be lists when present")
    return [str(item) for item in value]


def read_table_columns(path: str | Path) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            try:
                header = next(csv_reader(f))
            except StopIteration:
                return []
        return [str(column) for column in header]
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ArtifactError(
                f"pyarrow is required to validate parquet columns for {path}"
            ) from exc
        return [str(name) for name in pq.read_schema(path).names]
    raise ArtifactError(
        f"schema.columns validation supports only CSV and Parquet files, got {path}"
    )


def validate_artifact_schema_columns(artifact: ArtifactRecord, path: Path) -> list[str]:
    if "columns" not in artifact.schema:
        return []
    if not path.is_file():
        return [f"{artifact.name}: schema.columns requires a file artifact"]
    expected_columns = _normalise_schema_list(artifact.schema.get("columns"))
    try:
        actual_columns = read_table_columns(path)
    except ArtifactError as exc:
        return [f"{artifact.name}: {exc}"]
    missing = [column for column in expected_columns if column not in actual_columns]
    if missing:
        return [
            f"{artifact.name}: missing columns {missing}; "
            f"available columns are {actual_columns}"
        ]
    return []


def validate_artifact_schema_files(artifact: ArtifactRecord, path: Path) -> list[str]:
    if "files" not in artifact.schema:
        return []
    if not path.is_dir():
        return [f"{artifact.name}: schema.files requires a directory artifact"]
    expected_files = _normalise_schema_list(artifact.schema.get("files"))
    missing = [file_name for file_name in expected_files if not (path / file_name).exists()]
    if missing:
        return [f"{artifact.name}: missing required files {missing} in {path}"]
    return []
