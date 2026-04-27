"""Smoke-test specification loading and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_DYNAMIC_PARAMS_PATH,
    DEFAULT_OUTPUT_KEY,
    TESTS_ROOT,
)
from .json_utils import parse_string_list, read_json_file
from .models import SmokeSpec


def load_specs(tests_root: Path = TESTS_ROOT) -> list[SmokeSpec]:
    """Load and validate smoke specs from tests/*/test_spec.json."""
    specs: list[SmokeSpec] = []

    for spec_path in sorted(tests_root.glob("*/test_spec.json")):
        raw = read_json_file(spec_path)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid spec format in {spec_path}: expected object")

        folder = spec_path.parent
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"Missing or empty 'name' in {spec_path}")

        config_name = str(raw.get("config_file", "")).strip()
        if not config_name:
            raise ValueError(f"Missing or empty 'config_file' in {spec_path}")

        config_file = folder / config_name
        if not config_file.exists():
            raise FileNotFoundError(
                f"Smoke test '{name}' points to missing config: {config_file}"
            )

        specs.append(
            SmokeSpec(
                name=name,
                folder=folder,
                config_file=config_file,
                nb_iterations=int(raw.get("nb_iterations", 0)),
                seed=int(raw.get("seed", 0)),
                experiment_id=int(raw.get("experiment_id", 0)),
                output_key=str(raw.get("output_key", DEFAULT_OUTPUT_KEY)).strip(),
                dynamic_params_path=str(
                    raw.get("dynamic_params_path", DEFAULT_DYNAMIC_PARAMS_PATH)
                ).strip(),
                enabled=bool(raw.get("enabled", True)),
                skip_reason=str(raw.get("skip_reason", "")).strip(),
                compare_metrics=parse_string_list(
                    raw.get("compare_metrics", []),
                    field_name="compare_metrics",
                    file_path=spec_path,
                ),
            )
        )

    for spec in specs:
        if spec.nb_iterations <= 0:
            raise ValueError(
                f"Smoke test '{spec.name}' has invalid nb_iterations: "
                f"{spec.nb_iterations}"
            )
        if not spec.output_key:
            raise ValueError(
                f"Smoke test '{spec.name}' has empty output_key in test_spec.json"
            )

    return specs


def filter_specs(specs: Sequence[SmokeSpec], only: Sequence[str]) -> list[SmokeSpec]:
    """Filter selected specs from --only CLI values."""
    if not only:
        return list(specs)

    selected = {name.strip().lower() for name in only}
    filtered = [spec for spec in specs if spec.name.lower() in selected]
    missing = selected - {spec.name.lower() for spec in filtered}
    if missing:
        raise ValueError(f"Unknown smoke test(s): {', '.join(sorted(missing))}")
    return filtered


def baseline_path(spec: SmokeSpec) -> Path:
    """Return the baseline JSON path for a smoke spec."""
    return spec.folder / "baseline_ranges.json"
