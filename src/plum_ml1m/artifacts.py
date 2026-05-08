from __future__ import annotations

import hashlib
import json
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

    def validate_local(self, root: str | Path = ".") -> list[str]:
        self.validate_schema()
        root = Path(root)
        missing: list[str] = []
        for artifact in self.artifacts:
            path = artifact.resolved_path(root)
            if not path.exists():
                missing.append(f"{artifact.name}: missing {path}")
                continue
            if artifact.expected_type == "file" and not path.is_file():
                missing.append(f"{artifact.name}: expected file, got {path}")
            if artifact.expected_type == "dir" and not path.is_dir():
                missing.append(f"{artifact.name}: expected directory, got {path}")
            if artifact.sha256 and path.is_file():
                actual = sha256_file(path)
                if actual != artifact.sha256:
                    missing.append(
                        f"{artifact.name}: sha256 mismatch {actual} != {artifact.sha256}"
                    )
        return missing

    def by_kind(self, kind: str) -> list[ArtifactRecord]:
        return [artifact for artifact in self.artifacts if artifact.kind == kind]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
