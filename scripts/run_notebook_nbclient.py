from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=-1)
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    main()
