"""Run a random-parameter baseline without goal search or rendered outputs."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"
for import_root in (REPO_ROOT, EXAMPLES_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.analysis_metrics.random_run.baseline import run_random_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--nb_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_random_baseline(
        config_file=args.config_file,
        output_dir=args.output_dir,
        nb_iterations=args.nb_iterations,
        seed=args.seed,
    )
    print(f"Random baseline complete: {summary.discoveries_dir}")
    print(f"Discoveries: {summary.count}")


if __name__ == "__main__":
    main()
