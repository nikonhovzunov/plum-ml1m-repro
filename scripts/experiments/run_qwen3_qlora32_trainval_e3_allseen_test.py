from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

RUN_NAME = "sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen_v1"
CPT_RUN = "cpt_qwen3_4b_base_sid_v2_plum_curriculum_qlora32_v1"
NOTEBOOK = ROOT / "notebooks/sft/21_sft_qwen3_4b_qlora32_sid_v2_trainval_test_w16_e3_allseen.ipynb"
RUNNER_DIR = ROOT / "data/processed/artifacts/qwen3_qlora32_runner"


def guarded(
    command: list[str],
    log_name: str,
    limit_mib: int,
    poll_seconds: int,
    dry_run: bool,
) -> None:
    guard = ROOT / "scripts/experiments/run_with_vram_guard.py"
    full_cmd = [
        PYTHON,
        str(guard),
        "--limit-mib",
        str(limit_mib),
        "--poll-seconds",
        str(poll_seconds),
        "--log-path",
        str(RUNNER_DIR / f"{log_name}_vram.log"),
        "--",
        *command,
    ]
    print("+", " ".join(full_cmd), flush=True)
    if dry_run:
        return
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(full_cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--limit-mib", type=int, default=15000)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if not args.skip_train:
        guarded(
            [
                PYTHON,
                str(ROOT / "scripts/run_notebook_nbclient.py"),
                str(NOTEBOOK),
                "--timeout",
                "-1",
                "--kernel-name",
                "python3",
            ],
            "trainval_e3_allseen",
            args.limit_mib,
            args.poll_seconds,
            args.dry_run,
        )

    if not args.skip_test:
        guarded(
            [
                PYTHON,
                str(ROOT / "scripts/experiments/qwen3_sft_beam_sweep.py"),
                "--split",
                "test",
                "--max-users",
                "0",
                "--configs",
                "20:20",
                "--eval-batch-size",
                "1",
                "--run-name",
                RUN_NAME,
                "--base-cpt-dir",
                f"data/processed/artifacts/{CPT_RUN}/final_merged",
                "--adapter-dir",
                f"data/processed/artifacts/{RUN_NAME}/final_adapter",
                "--output-dir",
                f"data/processed/artifacts/{RUN_NAME}/test_full_beam20_return20_all_seen_{stamp}",
                "--seen-filter-scope",
                "all",
                "--load-in-4bit",
            ],
            "trainval_e3_allseen_test",
            args.limit_mib,
            args.poll_seconds,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
