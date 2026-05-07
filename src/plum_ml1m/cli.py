from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .artifacts import ArtifactManifest
from .config import validate_config, validate_config_dir
from .paths import find_project_root, resolve_project_path
from .smoke import run_smoke_test


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _plan(stage: str, args: argparse.Namespace) -> dict:
    config = validate_config(args.config)
    return {
        "stage": stage,
        "config": str(Path(args.config).resolve()),
        "name": config["name"],
        "execute": bool(args.execute),
        "note": "Heavy execution is opt-in. Without --execute this command only validates and prints the plan.",
    }


def _maybe_run_notebook(plan: dict, args: argparse.Namespace) -> int:
    _print_json(plan)
    if not args.execute:
        return 0
    notebook = getattr(args, "notebook", None)
    if not notebook:
        raise SystemExit("--execute requires --notebook for this command")
    root = find_project_root()
    script = root / "scripts" / "run_notebook_nbclient.py"
    cmd = [sys.executable, str(script), str(resolve_project_path(notebook, root)), "--timeout", "-1"]
    return subprocess.call(cmd, cwd=str(root))


def command_validate_config(args: argparse.Namespace) -> int:
    if args.config_dir:
        payload = {"configs": sorted(validate_config_dir(args.config_dir).keys())}
    else:
        payload = validate_config(args.config)
    _print_json({"status": "ok", **payload})
    return 0


def command_validate_artifacts(args: argparse.Namespace) -> int:
    manifest = ArtifactManifest.load(args.manifest)
    manifest.validate_required_types()
    _print_json({"status": "ok", "artifacts": len(manifest.artifacts)})
    return 0


def command_smoke_test(_args: argparse.Namespace) -> int:
    _print_json(run_smoke_test())
    return 0


def command_stage(stage: str):
    def _run(args: argparse.Namespace) -> int:
        return _maybe_run_notebook(_plan(stage, args), args)

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plum-ml1m")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--config")
    validate.add_argument("--config-dir")
    validate.set_defaults(func=command_validate_config)

    artifacts = sub.add_parser("validate-artifacts")
    artifacts.add_argument("--manifest", default="configs/artifact_manifest.yaml")
    artifacts.set_defaults(func=command_validate_artifacts)

    smoke = sub.add_parser("smoke-test")
    smoke.set_defaults(func=command_smoke_test)

    default_configs = {
        "prepare-data": "configs/prepare_data.yaml",
        "build-metadata": "configs/metadata.yaml",
        "build-embeddings": "configs/embeddings.yaml",
        "train-sid": "configs/rqvae_sid.yaml",
        "train-cpt": "configs/cpt.yaml",
        "train-sft": "configs/sft.yaml",
        "evaluate": "configs/evaluation.yaml",
    }

    for name, default_config in default_configs.items():
        stage_parser = sub.add_parser(name)
        stage_parser.add_argument("--config", default=default_config)
        stage_parser.add_argument("--notebook")
        stage_parser.add_argument("--execute", action="store_true")
        stage_parser.set_defaults(func=command_stage(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config" and not args.config and not args.config_dir:
        parser.error("validate-config requires --config or --config-dir")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
