"""Run coverage comparison for random vs. tool discoveries."""
import argparse
from pathlib import Path

from adtool.utils.coverage_comparison import run_coverage_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    summary = run_coverage_comparison(Path(args.config_file))
    print(f"Coverage comparison complete: {summary.run_dir}")
    print(
        f"Random embeddings: {summary.random_count}, Tool embeddings: {summary.tool_count}"
    )


if __name__ == "__main__":
    main()
