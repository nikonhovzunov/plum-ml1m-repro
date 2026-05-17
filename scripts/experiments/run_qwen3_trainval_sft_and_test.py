"""Legacy Qwen3 LoRA train+val SFT runner followed by full test evaluation.

Kept only to reproduce the earlier LoRA all-history row in the README. The
current active generative launcher is `run_qwen3_qlora32_trainval_e3_allseen_test.py`.
"""

from __future__ import annotations

import argparse
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


def run(args: list[str], dry_run: bool) -> None:
    print("+", " ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        run(
            [
                sys.executable,
                str(ROOT / "scripts/run_notebook_nbclient.py"),
                str(NOTEBOOK),
                "--timeout",
                "-1",
            ],
            args.dry_run,
        )
    if not args.skip_test:
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
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
