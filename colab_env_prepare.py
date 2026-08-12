"""Install this project's dependencies in a notebook or Colab runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).with_name("requirements.txt"),
    )
    args = parser.parse_args()
    if not args.requirements.is_file():
        raise FileNotFoundError(f"Requirements file not found: {args.requirements}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(args.requirements),
        ],
        check=True,
    )
    print("Environment ready. Run finetune_whisper.py with your dataset arguments.")


if __name__ == "__main__":
    main()
