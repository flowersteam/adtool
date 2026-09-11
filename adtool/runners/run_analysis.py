"""Run configured analysis modules across a list of discovery inputs."""

import argparse
from pathlib import Path

from adtool.user_tools.analysis_metrics.analysis_run import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "discovery_paths",
        type=Path,
        nargs="+",
        help="Discoveries, experiment-root, or checkpoint directories",
    )
    parser.add_argument("--config_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("analysis_runs"))
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()

    summary = run_analysis(
        args.discovery_paths,
        output_dir=args.output_dir,
        labels=args.label,
        config_file=args.config_file,
    )
    print(f"Analysis complete: {summary.run_dir}")
    print(", ".join(f"{dataset.label}: {dataset.count}" for dataset in summary.datasets))


if __name__ == "__main__":
    main()
