"""Data models for smoke-test execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
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
    enabled: bool
    skip_reason: str
    compare_metrics: list[str]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Result of executing one smoke-test experiment process."""

    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str
