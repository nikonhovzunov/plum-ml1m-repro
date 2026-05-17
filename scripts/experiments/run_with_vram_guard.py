from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    first = output.splitlines()[0].strip()
    return int(float(first))


def kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mib", type=int, default=14500)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.cmd:
        raise SystemExit("No command provided after --")
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    with args.log_path.open("a", encoding="utf-8") as log:
        log.write(f"command={' '.join(cmd)}\n")
        log.write(f"limit_mib={args.limit_mib}\n")
        log.flush()

        proc = subprocess.Popen(cmd)
        peak = 0
        try:
            while proc.poll() is None:
                mem = gpu_memory_mib()
                if mem is not None:
                    peak = max(peak, mem)
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} memory_mib={mem} peak_mib={peak}\n")
                    log.flush()
                    if mem > args.limit_mib:
                        log.write(f"VRAM limit exceeded: {mem} > {args.limit_mib}; killing pid={proc.pid}\n")
                        log.flush()
                        kill_tree(proc.pid)
                        raise SystemExit(99)
                time.sleep(args.poll_seconds)
        finally:
            if proc.poll() is None:
                kill_tree(proc.pid)

        log.write(f"exit_code={proc.returncode} peak_mib={peak}\n")
        log.flush()
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
