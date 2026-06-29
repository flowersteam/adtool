"""Run configured analysis modules between one primary and many comparison folders."""

import argparse
import sys
from pathlib import Path
from adtool.examples.analysis_metrics.analysis_run import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_discovery", type=Path)
    parser.add_argument("comparison_discoveries", type=Path, nargs="+")
    parser.add_argument("--config_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("analysis_runs"))
    parser.add_argument("--primary_label", type=str, default=None)
    parser.add_argument("--comparison_label", action="append", default=[])
    args = parser.parse_args()

    summary = run_analysis(
        args.primary_discovery,
        args.comparison_discoveries,
        output_dir=args.output_dir,
        primary_label=args.primary_label,
        comparison_labels=args.comparison_label,
        config_file=args.config_file,
    )
    print(f"Analysis complete: {summary.run_dir}")
    print(", ".join(f"{dataset.label}: {dataset.count}" for dataset in summary.datasets))


if __name__ == "__main__":
    main()
