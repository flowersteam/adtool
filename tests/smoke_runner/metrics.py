"""Metric extraction, baseline generation, and comparison logic."""

from __future__ import annotations

import json
import numbers
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from .json_utils import read_json_file
from .models import SmokeSpec
from .specs import baseline_path


def _extract_run_idx(path: Path) -> int:
    """Extract run index from discovery folder name if present."""
    match = re.search(r"_idx_(\d+)_", path.parent.name)
    return int(match.group(1)) if match else 10**9


def _get_nested_value(container: object, dotted_path: str) -> object:
    """Read a nested value from dictionaries using a dotted path."""
    if not dotted_path:
        return container

    current = container
    for chunk in dotted_path.split("."):
        if not isinstance(current, dict) or chunk not in current:
            raise KeyError(f"Missing path '{dotted_path}' at chunk '{chunk}'")
        current = current[chunk]
    return current


def _flatten_numeric_values(value: object) -> list[float]:
    """Extract all numeric values from nested dict/list/array structures."""
    if isinstance(value, (bool, np.bool_)):
        return []

    if isinstance(value, numbers.Real):
        return [float(value)]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []

        if np.issubdtype(value.dtype, np.number):
            return [float(x) for x in value.astype(np.float64).reshape(-1)]

        flattened: list[float] = []
        for nested_value in value.reshape(-1).tolist():
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened

    if isinstance(value, dict):
        flattened: list[float] = []
        for nested_value in value.values():
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened

    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for nested_value in value:
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened

    return []


def _to_flat_numeric_array(output: object, file_path: Path) -> np.ndarray:
    """Convert an output payload to a non-empty 1D float array."""
    try:
        out_arr = np.asarray(output, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Output at {file_path} is not numerically coercible"
        ) from exc

    if out_arr.size == 0:
        raise RuntimeError(f"Output embedding is empty in discovery: {file_path}")
    return out_arr


def collect_metrics(
    run_dir: Path,
    elapsed_seconds: float,
    output_key: str,
    dynamic_params_path: str,
) -> dict[str, float]:
    """Collect generic smoke metrics from discovery files."""
    discovery_files = list((run_dir / "discoveries").glob("*/discovery.json"))
    if not discovery_files:
        raise RuntimeError(f"No discoveries found in {run_dir / 'discoveries'}")

    discovery_files.sort(key=lambda p: (_extract_run_idx(p), p.parent.name))

    flat_outputs: list[np.ndarray] = []
    first_output_values: list[float] = []
    param_numeric_values: list[float] = []

    for file_path in discovery_files:
        data = read_json_file(file_path)
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid discovery format: {file_path}")

        try:
            output = _get_nested_value(data, output_key)
        except KeyError as error:
            raise RuntimeError(
                f"Missing output path '{output_key}' in discovery: {file_path}"
            ) from error

        out_arr = _to_flat_numeric_array(output, file_path)
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


def _loose_range(value: float, relative: float, absolute: float) -> dict[str, float]:
    """Build a symmetric loose range around a reference value."""
    delta = max(abs(value) * relative, absolute)
    return {"min": float(value - delta), "max": float(value + delta)}


def _clamp_range(
    bounds: dict[str, float],
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict[str, float]:
    """Clamp optional min/max values in a range definition."""
    clamped = dict(bounds)
    if min_value is not None:
        clamped["min"] = max(clamped["min"], min_value)
    if max_value is not None:
        clamped["max"] = min(clamped["max"], max_value)
    return clamped


def build_default_ranges(reference: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build baseline acceptance ranges from reference metrics."""
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
        "output_first_mean": _loose_range(reference["output_first_mean"], 0.35, 1.0),
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


def refresh_baseline(spec: SmokeSpec, metrics: dict[str, float]) -> None:
    """Generate and persist baseline ranges for a spec."""
    payload = {
        "name": spec.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": metrics,
        "ranges": build_default_ranges(metrics),
        "notes": (
            "Loose acceptance ranges for smoke testing. "
            "Regenerate when intentionally changing behavior."
        ),
    }
    baseline_path(spec).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_ranges_for_spec(
    spec: SmokeSpec,
    ranges: Mapping[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Filter baseline ranges according to optional spec.compare_metrics."""
    if not spec.compare_metrics:
        return dict(ranges)

    selected: dict[str, dict[str, float]] = {}
    missing_metrics: list[str] = []

    for metric_name in spec.compare_metrics:
        if metric_name not in ranges:
            missing_metrics.append(metric_name)
            continue
        selected[metric_name] = ranges[metric_name]

    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise RuntimeError(
            "Requested compare_metrics are missing in baseline "
            f"for '{spec.name}': {missing}. "
            "Run with --refresh-baseline after updating metrics."
        )

    return selected


def compare_metrics(
    metrics: dict[str, float],
    ranges: dict[str, dict[str, float]],
) -> list[str]:
    """Compare current metrics against baseline ranges."""
    failures: list[str] = []

    for metric_name, bounds in ranges.items():
        if not isinstance(bounds, dict):
            failures.append(f"{metric_name}: invalid bounds format (expected object)")
            continue

        if metric_name not in metrics:
            failures.append(f"Missing metric '{metric_name}' in test output")
            continue

        value = metrics[metric_name]
        if not np.isfinite(value):
            failures.append(f"{metric_name}: value is non-finite ({value})")
            continue

        min_raw = bounds.get("min")
        max_raw = bounds.get("max")

        try:
            min_val = float(min_raw) if min_raw is not None else None
        except (TypeError, ValueError):
            failures.append(f"{metric_name}: invalid lower bound ({min_raw})")
            continue

        try:
            max_val = float(max_raw) if max_raw is not None else None
        except (TypeError, ValueError):
            failures.append(f"{metric_name}: invalid upper bound ({max_raw})")
            continue

        if min_val is not None and not np.isfinite(min_val):
            failures.append(f"{metric_name}: lower bound is non-finite ({min_val})")
            continue

        if max_val is not None and not np.isfinite(max_val):
            failures.append(f"{metric_name}: upper bound is non-finite ({max_val})")
            continue

        if min_val is not None and max_val is not None and min_val > max_val:
            failures.append(
                f"{metric_name}: lower bound {min_val:.6g} is above "
                f"upper bound {max_val:.6g}"
            )
            continue

        if min_val is not None and value < min_val:
            failures.append(f"{metric_name}: {value:.6g} < lower bound {min_val:.6g}")

        if max_val is not None and value > max_val:
            failures.append(f"{metric_name}: {value:.6g} > upper bound {max_val:.6g}")

    return failures
