from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ARTIFACT_TYPES = {
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


@dataclass(frozen=True)
class ArtifactRecord:
    type: str
    path: str
    description: str = ""
    tracked_by_git: bool = False
    required_for: list[str] | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    artifacts: list[ArtifactRecord]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        rows = data.get("artifacts", [])
        if not isinstance(rows, list):
            raise ValueError("artifact manifest field 'artifacts' must be a list")
        artifacts = [ArtifactRecord(**row) for row in rows]
        return cls(artifacts=artifacts)

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def validate_required_types(self) -> None:
        present = {artifact.type for artifact in self.artifacts}
        missing = sorted(REQUIRED_ARTIFACT_TYPES - present)
        if missing:
            raise ValueError(f"Artifact manifest is missing types: {missing}")

    def by_type(self, artifact_type: str) -> list[ArtifactRecord]:
        return [artifact for artifact in self.artifacts if artifact.type == artifact_type]
