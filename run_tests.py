#!/usr/bin/env python3
"""Smoke-test runner for example experiments.

Each smoke test lives in ``tests/<example>/`` and provides:
- ``test_spec.json``: execution settings (config, seed, iterations)
- ``smoke_config.json``: tiny/fast experiment config template
- ``baseline_ranges.json``: acceptable metric ranges for comparisons

By default this script executes all available smoke tests and verifies that:
1) experiments run without runtime errors;
2) basic result/performance metrics are within loose baseline ranges.
"""

from __future__ import annotations

import argparse
import numbers
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
TESTS_ROOT = ROOT / "tests"
TMP_ROOT = TESTS_ROOT / ".tmp"


@dataclass(frozen=True)
class SmokeSpec:
    """Executable smoke-test specification."""

    name: str
    folder: Path
    config_file: Path
    nb_iterations: int
    seed: int
    experiment_id: int
    output_key: str
    dynamic_params_path: str


def _load_specs(tests_root: Path) -> List[SmokeSpec]:
    specs: List[SmokeSpec] = []
    for spec_path in sorted(tests_root.glob("*/test_spec.json")):
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        folder = spec_path.parent
        name = str(raw["name"])
        config_file = folder / str(raw["config_file"])
        if not config_file.exists():
            raise FileNotFoundError(
                f"Smoke test '{name}' points to missing config: "
                f"{config_file}"
            )

        specs.append(
            SmokeSpec(
                name=name,
                folder=folder,
                config_file=config_file,
                nb_iterations=int(raw["nb_iterations"]),
                seed=int(raw["seed"]),
                experiment_id=int(raw.get("experiment_id", 0)),
                output_key=str(raw.get("output_key", "output")),
                dynamic_params_path=str(
                    raw.get("dynamic_params_path", "params.dynamic_params")
                ),
            )
        )

    return specs


def _create_run_dir(test_name: str) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_dir = TMP_ROOT / f"{test_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _prepare_runtime_config(template_path: Path, run_dir: Path) -> Path:
    cfg = json.loads(template_path.read_text(encoding="utf-8"))
    # ExperimentPipeline expects save_location to end with '/'.
    cfg["experiment"]["config"]["save_location"] = f"{run_dir.as_posix()}/"
    runtime_cfg = run_dir / "runtime_config.json"
    runtime_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return runtime_cfg


def _run_experiment(spec: SmokeSpec, runtime_cfg: Path) -> Tuple[float, str, str]:
    cmd = [
        sys.executable,
        "run.py",
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
    return elapsed, process.stdout, process.stderr if process.returncode != 0 else ""


def _extract_run_idx(path: Path) -> int:
    match = re.search(r"_idx_(\d+)_", path.parent.name)
    return int(match.group(1)) if match else 10**9


def _get_nested_value(container: object, dotted_path: str) -> object:
    """Read a nested value from dictionaries using a dotted path."""
    current = container
    for chunk in dotted_path.split("."):
        if not isinstance(current, dict) or chunk not in current:
            raise KeyError(
                f"Missing path '{dotted_path}' at chunk '{chunk}'"
            )
        current = current[chunk]
    return current


def _flatten_numeric_values(value: object) -> List[float]:
    """Extract all numeric values from nested dict/list/array structures."""
    if isinstance(value, bool):
        return []

    if isinstance(value, numbers.Real):
        return [float(value)]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return [float(x) for x in value.astype(np.float64).reshape(-1)]

    if isinstance(value, dict):
        flattened: List[float] = []
        for nested_value in value.values():
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened

    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for nested_value in value:
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened

    return []


def _collect_metrics(
    run_dir: Path,
    elapsed_seconds: float,
    output_key: str,
    dynamic_params_path: str,
) -> Dict[str, float]:
    discovery_files = list((run_dir / "discoveries").glob("*/discovery.json"))
    if not discovery_files:
        raise RuntimeError(f"No discoveries found in {run_dir / 'discoveries'}")

    discovery_files.sort(key=lambda p: (_extract_run_idx(p), p.parent.name))

    flat_outputs: List[np.ndarray] = []
    first_output_values: List[float] = []
    param_numeric_values: List[float] = []

    for file_path in discovery_files:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        try:
            output = _get_nested_value(data, output_key)
        except KeyError as error:
            raise RuntimeError(
                f"Missing output path '{output_key}' in discovery: {file_path}"
            ) from error

        out_arr = np.asarray(output, dtype=np.float64).reshape(-1)
        if out_arr.size == 0:
            raise RuntimeError(f"Output embedding is empty in discovery: {file_path}")

        flat_outputs.append(out_arr)
        first_output_values.append(float(out_arr[0]))

        try:
            dynamic_params = _get_nested_value(data, dynamic_params_path)
        except KeyError:
            dynamic_params = {}

        if not isinstance(dynamic_params, dict):
            raise RuntimeError(
                "Dynamic parameter path does not resolve to a dictionary: "
                f"'{dynamic_params_path}' in {file_path}"
            )

        param_numeric_values.extend(_flatten_numeric_values(dynamic_params))

    output_sizes = [arr.size for arr in flat_outputs]
    if len(set(output_sizes)) != 1:
        raise RuntimeError(
            "Output embedding size is not stable across discoveries: "
            f"{output_sizes}"
        )

    output_matrix = np.stack(flat_outputs)
    all_output_values = output_matrix.reshape(-1)

    finite_mask = np.isfinite(all_output_values)
    finite_output_values = all_output_values[finite_mask]
    if finite_output_values.size == 0:
        raise RuntimeError("Output embedding contains no finite values")

    finite_first_values = [x for x in first_output_values if np.isfinite(x)]
    if not finite_first_values:
        raise RuntimeError("Output first component is non-finite for all discoveries")

    sanitized_last_output = np.nan_to_num(
        output_matrix[-1], nan=0.0, posinf=0.0, neginf=0.0
    )

    metrics = {
        "discovery_count": float(output_matrix.shape[0]),
        "output_dim": float(output_matrix.shape[1]),
        "output_mean": float(finite_output_values.mean()),
        "output_std": float(finite_output_values.std()),
        "output_last_l2": float(np.linalg.norm(sanitized_last_output)),
        "output_finite_ratio": float(
            finite_output_values.size / all_output_values.size
        ),
        "output_first_mean": float(np.mean(finite_first_values)),
        "runtime_seconds": float(elapsed_seconds),
        "seconds_per_iteration": float(elapsed_seconds / output_matrix.shape[0]),
    }

    if param_numeric_values:
        metrics["param_numeric_mean"] = float(np.mean(param_numeric_values))
        metrics["param_numeric_std"] = float(np.std(param_numeric_values))

    return metrics


def _loose_range(value: float, relative: float, absolute: float) -> Dict[str, float]:
    delta = max(abs(value) * relative, absolute)
    return {"min": float(value - delta), "max": float(value + delta)}


def _clamp_range(
    bounds: Dict[str, float],
    min_value: float | None = None,
    max_value: float | None = None,
) -> Dict[str, float]:
    clamped = dict(bounds)
    if min_value is not None:
        clamped["min"] = max(clamped["min"], min_value)
    if max_value is not None:
        clamped["max"] = min(clamped["max"], max_value)
    return clamped


def _build_default_ranges(reference: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    ranges = {
        "discovery_count": {
            "min": reference["discovery_count"],
            "max": reference["discovery_count"],
        },
        "output_dim": {
            "min": reference["output_dim"],
            "max": reference["output_dim"],
        },
        "output_mean": _loose_range(reference["output_mean"], 0.35, 1.0),
        "output_std": _clamp_range(
            _loose_range(reference["output_std"], 0.35, 1.0),
            min_value=0.0,
        ),
        "output_last_l2": _clamp_range(
            _loose_range(reference["output_last_l2"], 0.4, 2.0),
            min_value=0.0,
        ),
        "output_finite_ratio": _clamp_range(
            _loose_range(reference["output_finite_ratio"], 0.15, 0.05),
            min_value=0.0,
            max_value=1.0,
        ),
        "output_first_mean": _loose_range(
            reference["output_first_mean"], 0.35, 1.0
        ),
        # Performance guardrail: allow slower runs, but catch major regressions.
        "seconds_per_iteration": {
            "min": 0.0,
            "max": float(reference["seconds_per_iteration"] * 4.0 + 0.5),
        },
    }

    if "param_numeric_mean" in reference:
        ranges["param_numeric_mean"] = _loose_range(
            reference["param_numeric_mean"], 0.6, 0.02
        )

    if "param_numeric_std" in reference:
        ranges["param_numeric_std"] = _clamp_range(
            _loose_range(reference["param_numeric_std"], 0.6, 0.02),
            min_value=0.0,
        )

    return ranges


def _baseline_path(spec: SmokeSpec) -> Path:
    return spec.folder / "baseline_ranges.json"


def _refresh_baseline(spec: SmokeSpec, metrics: Dict[str, float]) -> None:
    payload = {
        "name": spec.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": metrics,
        "ranges": _build_default_ranges(metrics),
        "notes": (
            "Loose acceptance ranges for smoke testing. "
            "Regenerate when intentionally changing behavior."
        ),
    }
    _baseline_path(spec).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compare_metrics(
    metrics: Dict[str, float],
    ranges: Dict[str, Dict[str, float]],
) -> List[str]:
    failures: List[str] = []

    for metric_name, bounds in ranges.items():
        if metric_name not in metrics:
            failures.append(f"Missing metric '{metric_name}' in test output")
            continue

        value = metrics[metric_name]
        if not np.isfinite(value):
            failures.append(f"{metric_name}: value is non-finite ({value})")
            continue

        min_val = bounds.get("min")
        max_val = bounds.get("max")

        if min_val is not None and not np.isfinite(float(min_val)):
            failures.append(f"{metric_name}: lower bound is non-finite ({min_val})")
            continue

        if max_val is not None and not np.isfinite(float(max_val)):
            failures.append(f"{metric_name}: upper bound is non-finite ({max_val})")
            continue

        if min_val is not None and value < min_val:
            failures.append(
                f"{metric_name}: {value:.6g} < lower bound {float(min_val):.6g}"
            )

        if max_val is not None and value > max_val:
            failures.append(
                f"{metric_name}: {value:.6g} > upper bound {float(max_val):.6g}"
            )

    return failures


def _run_single_test(
    spec: SmokeSpec,
    refresh_baseline: bool,
    keep_artifacts: bool,
    verbose: bool,
) -> Tuple[bool, List[str]]:
    run_dir = _create_run_dir(spec.name)

    try:
        runtime_cfg = _prepare_runtime_config(spec.config_file, run_dir)
        elapsed, stdout, stderr = _run_experiment(spec, runtime_cfg)

        if stderr:
            message = (
                f"Experiment execution failed for '{spec.name}'.\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
            return False, [message]

        metrics = _collect_metrics(
            run_dir,
            elapsed,
            output_key=spec.output_key,
            dynamic_params_path=spec.dynamic_params_path,
        )
        if verbose:
            pretty_metrics = json.dumps(metrics, indent=2)
            print(f"[{spec.name}] metrics:\n{pretty_metrics}")

        if refresh_baseline:
            _refresh_baseline(spec, metrics)
            return True, [f"Baseline refreshed at {_baseline_path(spec)}"]

        baseline_file = _baseline_path(spec)
        if not baseline_file.exists():
            return (
                False,
                [
                    f"Missing baseline for '{spec.name}': {baseline_file}",
                    "Run with --refresh-baseline to create it.",
                ],
            )

        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        ranges = baseline.get("ranges", {})
        failures = _compare_metrics(metrics, ranges)
        return len(failures) == 0, failures if failures else ["All checks passed"]
    finally:
        if keep_artifacts:
            print(f"[{spec.name}] artifacts kept at: {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def _filter_specs(specs: List[SmokeSpec], only: List[str]) -> List[SmokeSpec]:
    if not only:
        return specs

    selected = {name.strip().lower() for name in only}
    filtered = [spec for spec in specs if spec.name.lower() in selected]
    missing = selected - {spec.name.lower() for spec in filtered}
    if missing:
        raise ValueError(f"Unknown smoke test(s): {', '.join(sorted(missing))}")
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Run adtool smoke tests")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Run only selected smoke tests by name (e.g. --only grayscott)",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Regenerate baseline ranges from current results",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-test runtime artifacts under tests/.tmp",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed test metrics",
    )
    args = parser.parse_args()

    specs = _load_specs(TESTS_ROOT)
    if not specs:
        print("No smoke tests found under tests/*/test_spec.json")
        return 1

    selected_specs = _filter_specs(specs, args.only)

    all_passed = True
    for spec in selected_specs:
        print(f"[RUN] {spec.name}")
        passed, details = _run_single_test(
            spec,
            refresh_baseline=args.refresh_baseline,
            keep_artifacts=args.keep_artifacts,
            verbose=args.verbose,
        )

        if passed:
            print(f"[PASS] {spec.name}")
            for detail in details:
                print(f"  - {detail}")
            continue

        all_passed = False
        print(f"[FAIL] {spec.name}")
        for detail in details:
            print(f"  - {detail}")

    if not all_passed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
