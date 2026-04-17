"""
One script to run the whole V2.2 thing.

Basically does everything so we do not have to run
like 8 commands by hand from the root folder:
- build data
- train encoders
- train query tower
- train reranker
- run eval
- run readiness check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


STEPS = [
    ("Build V2.2 data", [sys.executable, "scripts/build_v2_data.py"]),
    ("Train song encoder", [sys.executable, "src/v2/train/train_song_encoder.py"]),
    ("Train user encoder", [sys.executable, "src/v2/train/train_user_encoder.py"]),
    ("Train context encoder", [sys.executable, "src/v2/train/train_context_encoder.py"]),
    ("Train query tower", [sys.executable, "src/v2/train/train_query_tower.py"]),
    ("Train reranker", [sys.executable, "src/v2/train/train_reranker.py"]),
    ("Run offline eval", [sys.executable, "src/v2/eval/offline_eval.py", "--split", "test", "--candidate-k", "50"]),
    ("Run readiness check", [sys.executable, "src/v2/eval/check_readiness.py"]),
]


def run_step(index: int, total: int, title: str, command: list[str], verbose: bool) -> None:
    print("=" * 96)
    print(f"[{index}/{total}] {title}")
    print("=" * 96)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout and exc.stdout.strip():
            print(exc.stdout.strip())
        if exc.stderr and exc.stderr.strip():
            print(exc.stderr.strip(), file=sys.stderr)
        raise

    if verbose:
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip(), file=sys.stderr)
    else:
        print("done")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full V2.2 pipeline in one go.")
    parser.add_argument("--verbose", action="store_true", help="Show all the noisy logs from each step.")
    args = parser.parse_args()

    total = len(STEPS)
    for idx, (title, command) in enumerate(STEPS, start=1):
        run_step(idx, total, title, command, verbose=bool(args.verbose))
    print("=" * 96)
    print("V2.2 full run complete.")
    print("Use `python eval_v2.py --no-rerun` if you just want the latest summary again.")
    print("Use `python demo_v2.py` for demo output.")
    print("If `python` is weird on your Mac, try `python3`.")
    print("=" * 96)


if __name__ == "__main__":
    main()
