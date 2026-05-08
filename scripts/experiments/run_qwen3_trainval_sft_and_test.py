"""Experimental Qwen3 train+val SFT runner followed by full test evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_root(start: Path) -> Path:
    root = start.resolve()
    while not (root / "src").exists() and root.parent != root:
        root = root.parent
    return root


ROOT = find_root(Path.cwd())
RUN_NAME = "sft_qwen3_4b_sid_v2_trainval_test_w16_bestepoch_v1"
BASE_CPT_DIR = (
    ROOT / "data/processed/artifacts/cpt_qwen3_4b_base_sid_v2_plum_curriculum_v1/final_merged"
)
OUTPUT_DIR = ROOT / "data/processed/artifacts" / RUN_NAME
NOTEBOOK = ROOT / "notebooks/sft/11_sft_qwen3_4b_sid_v2_trainval_test_w16_bestepoch.ipynb"
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"
TEST_EVAL_DIR = OUTPUT_DIR / "test_full_beam20_return20"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ROOT / "scripts/run_notebook_nbclient.py"),
            str(NOTEBOOK),
            "--timeout",
            "-1",
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/experiments/qwen3_sft_beam_sweep.py"),
            "--split",
            "test",
            "--max-users",
            "0",
            "--configs",
            "20:20",
            "--run-name",
            RUN_NAME,
            "--base-cpt-dir",
            str(BASE_CPT_DIR),
            "--adapter-dir",
            str(FINAL_ADAPTER_DIR),
            "--output-dir",
            str(TEST_EVAL_DIR),
        ]
    )


if __name__ == "__main__":
    main()
