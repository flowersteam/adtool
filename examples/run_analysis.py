"""Run configured analysis modules between two discovery folders."""

import argparse
import sys
from pathlib import Path
from adtool.examples.analysis_metrics.analysis_run import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_a", type=Path)
    parser.add_argument("discovery_b", type=Path)
    parser.add_argument("--config_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("analysis_runs"))
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    args = parser.parse_args()

    summary = run_analysis(
        args.discovery_a,
        args.discovery_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        config_file=args.config_file,
    )
    print(f"Analysis complete: {summary.run_dir}")
    print(
        f"{summary.dataset_a.label}: {summary.dataset_a.count}, "
        f"{summary.dataset_b.label}: {summary.dataset_b.count}"
    )


if __name__ == "__main__":
    main()
