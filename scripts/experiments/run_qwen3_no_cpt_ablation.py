"""Run the Qwen3 no-CPT SFT ablation.

Pipeline:
1. SFT Qwen3-4B-Base directly on train and select epoch on validation.
2. Full validation evaluation with trie decoding and all-history seen filtering.
3. SFT Qwen3-4B-Base directly on train+val for the selected epoch count.
4. Full test evaluation with the same decoding/filtering protocol.
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
BASE_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/"
    / "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)

VAL_RUN_NAME = "sft_qwen3_4b_no_cpt_sid_v2_next_watch_w16_pat2_v1"
TEST_RUN_NAME = "sft_qwen3_4b_no_cpt_sid_v2_trainval_test_w16_selected_v1"

VAL_NOTEBOOK = ROOT / "notebooks/sft/12_sft_qwen3_4b_no_cpt_sid_v2_next_watch_w16_pat2.ipynb"
TEST_NOTEBOOK = (
    ROOT / "notebooks/sft/13_sft_qwen3_4b_no_cpt_sid_v2_trainval_test_w16_selected.ipynb"
)

VAL_OUTPUT_DIR = ROOT / "data/processed/artifacts" / VAL_RUN_NAME
TEST_OUTPUT_DIR = ROOT / "data/processed/artifacts" / TEST_RUN_NAME


def run(args: list[str]) -> None:
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--skip-validation-train", action="store_true")
    parser.add_argument("--skip-full-val", action="store_true")
    parser.add_argument("--skip-trainval-test-train", action="store_true")
    parser.add_argument("--skip-full-test", action="store_true")
    args = parser.parse_args()

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Qwen3 base snapshot is missing: {BASE_MODEL_DIR}. "
            "Download/cache Qwen/Qwen3-4B-Base or set QWEN3_4B_BASE_DIR in the notebooks."
        )

    if not args.skip_validation_train:
        run(
            [
                sys.executable,
                ROOT / "scripts/run_notebook_nbclient.py",
                VAL_NOTEBOOK,
                "--timeout",
                "-1",
            ]
        )

    if not args.skip_full_val:
        run(
            [
                sys.executable,
                ROOT / "scripts/experiments/qwen3_sft_beam_sweep.py",
                "--split",
                "val",
                "--max-users",
                "0",
                "--configs",
                "20:20",
                "--run-name",
                VAL_RUN_NAME,
                "--base-cpt-dir",
                BASE_MODEL_DIR,
                "--adapter-dir",
                VAL_OUTPUT_DIR / "best_adapter",
                "--output-dir",
                VAL_OUTPUT_DIR / "full_val_beam20_return20_strict_full_seen_b5",
                "--seen-filter-scope",
                "all",
                "--eval-batch-size",
                str(args.eval_batch_size),
            ]
        )

    if not args.skip_trainval_test_train:
        run(
            [
                sys.executable,
                ROOT / "scripts/run_notebook_nbclient.py",
                TEST_NOTEBOOK,
                "--timeout",
                "-1",
            ]
        )

    if not args.skip_full_test:
        run(
            [
                sys.executable,
                ROOT / "scripts/experiments/qwen3_sft_beam_sweep.py",
                "--split",
                "test",
                "--max-users",
                "0",
                "--configs",
                "20:20",
                "--run-name",
                TEST_RUN_NAME,
                "--base-cpt-dir",
                BASE_MODEL_DIR,
                "--adapter-dir",
                TEST_OUTPUT_DIR / "final_adapter",
                "--output-dir",
                TEST_OUTPUT_DIR / "test_full_beam20_return20_strict_full_seen_b5",
                "--seen-filter-scope",
                "all",
                "--eval-batch-size",
                str(args.eval_batch_size),
            ]
        )


if __name__ == "__main__":
    main()
