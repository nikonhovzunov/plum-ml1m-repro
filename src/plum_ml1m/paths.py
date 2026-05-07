from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    root = Path(start or Path.cwd()).resolve()
    if root.is_file():
        root = root.parent
    while root.parent != root:
        if (root / "README.md").exists() and (root / "src").exists():
            return root
        root = root.parent
    raise FileNotFoundError("Could not locate project root containing README.md and src/")


def resolve_project_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return find_project_root(root) / path
