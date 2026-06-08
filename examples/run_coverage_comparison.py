"""Compare coverage between two discovery folders."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis_metrics.coverage_comparison import compare_discovery_sets
except ModuleNotFoundError as exc:
    if exc.name != "analysis_metrics":
        raise
    from examples.analysis_metrics.coverage_comparison import compare_discovery_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_a", type=Path)
    parser.add_argument("discovery_b", type=Path)
    parser.add_argument("--config_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("coverage_runs"))
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    parser.add_argument("--points", type=int, default=None)
    args = parser.parse_args()

    summary = compare_discovery_sets(
        args.discovery_a,
        args.discovery_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        config_file=args.config_file,
        points=args.points,
    )
    print(f"Coverage comparison complete: {summary.run_dir}")
    print(f"{summary.label_a}: {summary.count_a}, {summary.label_b}: {summary.count_b}")


if __name__ == "__main__":
    main()
