"""Experiment execution and end-to-end per-test orchestration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .constants import ROOT, TMP_ROOT
from .json_utils import read_json_file
from .metrics import (
    collect_metrics,
    compare_metrics,
    refresh_baseline,
    select_ranges_for_spec,
)
from .models import ExperimentResult, SmokeSpec
from .specs import baseline_path


def create_run_dir(test_name: str) -> Path:
    """Create a unique temporary run directory for a test."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_dir = TMP_ROOT / f"{test_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_runtime_config(template_path: Path, run_dir: Path) -> Path:
    """Prepare test config by overriding save location with an isolated folder."""
    cfg = json.loads(template_path.read_text(encoding="utf-8"))
    # ExperimentPipeline expects save_location to end with '/'.
    cfg["experiment"]["config"]["save_location"] = f"{run_dir.as_posix()}/"
    runtime_cfg = run_dir / "runtime_config.json"
    runtime_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return runtime_cfg


def run_experiment(spec: SmokeSpec, runtime_cfg: Path) -> ExperimentResult:
    """Execute run_experimentations.py for a single smoke spec."""
    cmd = [
        sys.executable,
        "-m",
        "adtool.runners.run_experimentations",
        "--config_file",
        str(runtime_cfg),
        "--nb_iterations",
        str(spec.nb_iterations),
        "--seed",
        str(spec.seed),
        "--experiment_id",
        str(spec.experiment_id),
    ]

    start = time.perf_counter()
    process = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    return ExperimentResult(
        returncode=process.returncode,
        elapsed_seconds=elapsed,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def run_single_test(
    spec: SmokeSpec,
    refresh_baseline_flag: bool,
    keep_artifacts: bool,
    verbose: bool,
) -> tuple[bool, list[str]]:
    """Run one smoke test end-to-end."""
    run_dir = create_run_dir(spec.name)

    try:
        runtime_cfg = prepare_runtime_config(spec.config_file, run_dir)
        result = run_experiment(spec, runtime_cfg)

        if result.returncode != 0:
            message = (
                f"Experiment execution failed for '{spec.name}'.\n"
                f"Exit code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            return False, [message]

        metrics = collect_metrics(
            run_dir,
            result.elapsed_seconds,
            output_key=spec.output_key,
            dynamic_params_path=spec.dynamic_params_path,
        )

        if verbose:
            print(f"[{spec.name}] metrics:\n{json.dumps(metrics, indent=2)}")

        if refresh_baseline_flag:
            refresh_baseline(spec, metrics)
            return True, [f"Baseline refreshed at {baseline_path(spec)}"]

        baseline_file = baseline_path(spec)
        if not baseline_file.exists():
            return (
                False,
                [
                    f"Missing baseline for '{spec.name}': {baseline_file}",
                    "Run with --refresh-baseline to create it.",
                ],
            )

        baseline = read_json_file(baseline_file)
        if not isinstance(baseline, dict):
            return False, [
                f"Invalid baseline format for '{spec.name}': {baseline_file}"
            ]

        raw_ranges = baseline.get("ranges", {})
        if not isinstance(raw_ranges, dict):
            return False, [f"Invalid ranges format in baseline: {baseline_file}"]

        try:
            ranges = select_ranges_for_spec(spec, raw_ranges)
        except RuntimeError as exc:
            return False, [str(exc)]

        failures = compare_metrics(metrics, ranges)
        return len(failures) == 0, failures if failures else ["All checks passed"]
    finally:
        if keep_artifacts:
            print(f"[{spec.name}] artifacts kept at: {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
