"""Load persisted discoveries independently of their storage format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_history import Checkpoint, load_branch_discoveries


@dataclass(frozen=True)
class LoadedDiscoveries:
    """Discovery payloads and their stable source references."""

    path: Path
    sources: list[Path]
    payloads: list[dict[str, Any]]
    checkpoint: Checkpoint | None = None

    @property
    def source_label(self) -> str:
        if self.checkpoint is not None:
            return f"Checkpoint {self.checkpoint.name}"
        return f"Discovery folder {self.path}"


def _json_discovery_paths(discovery_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in discovery_path.rglob("discovery.json")
            if path.is_file() and "analysis_runs" not in path.relative_to(discovery_path).parts
        ),
        key=lambda path: path.stat().st_mtime,
    )


def load_discoveries(
    discovery_path: str | Path,
    checkpoint_name: str | None = None,
) -> LoadedDiscoveries:
    """Load discoveries from a checkpoint branch or JSON discovery folder.

    A directory containing ``checkpoints/`` is loaded from the selected
    checkpoint branch. Otherwise, recursively saved ``discovery.json`` files
    are loaded in modification-time order.
    """
    discovery_path = Path(discovery_path).resolve()
    if (discovery_path / "checkpoints").is_dir():
        checkpoint_discoveries = load_branch_discoveries(
            discovery_path,
            checkpoint_name=checkpoint_name,
        )
        payloads = checkpoint_discoveries.payloads
        sources = [
            checkpoint_discoveries.checkpoint.path / f"history-record-{index:08d}"
            for index in range(len(payloads))
        ]
        return LoadedDiscoveries(
            path=discovery_path,
            sources=sources,
            payloads=payloads,
            checkpoint=checkpoint_discoveries.checkpoint,
        )

    sources = _json_discovery_paths(discovery_path)
    if not sources:
        raise ValueError(f"No discoveries found in {discovery_path}")

    payloads: list[dict[str, Any]] = []
    for source in sources:
        try:
            with source.open("r") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load discovery {source}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Discovery {source} must contain a JSON object")
        payloads.append(payload)

    return LoadedDiscoveries(
        path=discovery_path,
        sources=sources,
        payloads=payloads,
    )


def numeric_discovery_output(
    payload: dict[str, Any],
    *,
    source_label: str = "Discovery",
) -> np.ndarray:
    """Return a discovery's output as a numeric NumPy array."""
    try:
        return np.asarray(payload["output"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source_label} contains a discovery without a numeric output"
        ) from error


def numeric_discovery_output_matrix(discoveries: LoadedDiscoveries) -> np.ndarray:
    """Return flattened numeric outputs for a loaded discovery collection."""
    outputs = [
        numeric_discovery_output(payload, source_label=discoveries.source_label).reshape(-1)
        for payload in discoveries.payloads
    ]
    try:
        return np.vstack(outputs)
    except ValueError as error:
        raise ValueError(
            f"{discoveries.source_label} contains numeric outputs with incompatible shapes"
        ) from error
