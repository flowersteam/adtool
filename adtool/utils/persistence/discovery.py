"""Load persisted discoveries independently of their storage format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_history import (
    Checkpoint,
    checkpoint_input_chain,
    load_branch_discoveries,
    load_checkpoint_records,
)


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

    An experiment root containing ``checkpoints/`` or an explicit checkpoint
    directory is loaded cumulatively from the selected branch. Otherwise,
    recursively saved ``discovery.json`` files are loaded in modification-time
    order.
    """
    discovery_path = Path(discovery_path).resolve()
    if (discovery_path / "manifest.json").is_file():
        groups = load_discovery_groups(discovery_path)
        return LoadedDiscoveries(
            path=discovery_path,
            sources=[source for group in groups for source in group.sources],
            payloads=[payload for group in groups for payload in group.payloads],
            checkpoint=groups[-1].checkpoint,
        )

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


def load_discovery_groups(
    discovery_path: str | Path,
    checkpoint_name: str | None = None,
) -> list[LoadedDiscoveries]:
    """Load one discovery group or checkpoint-local groups for analysis.

    Checkpoint groups are returned from the oldest ancestor to the selected
    checkpoint. Each group contains only the history records owned by that
    checkpoint, so callers can give every checkpoint a separate series.
    """
    discovery_path = Path(discovery_path).resolve()
    chain = checkpoint_input_chain(discovery_path, checkpoint_name)
    if chain is None:
        return [load_discoveries(discovery_path)]

    groups: list[LoadedDiscoveries] = []
    for checkpoint in chain:
        payloads = load_checkpoint_records(checkpoint)
        if not payloads:
            raise ValueError(f"No discoveries found in checkpoint {checkpoint.name}")
        first_run_idx = int(checkpoint.manifest.get("step", 0)) - len(payloads)
        branch_id = str(checkpoint.manifest.get("branch_id") or checkpoint.name)
        normalized_payloads = []
        for offset, payload in enumerate(payloads):
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("run_idx", first_run_idx + offset)
            if not metadata.get("branch_id"):
                metadata["branch_id"] = branch_id
            normalized_payloads.append({**payload, "metadata": metadata})
        groups.append(
            LoadedDiscoveries(
                path=checkpoint.path,
                sources=[
                    checkpoint.path / f"history-record-{index:08d}"
                    for index in range(len(normalized_payloads))
                ],
                payloads=normalized_payloads,
                checkpoint=checkpoint,
            )
        )
    return groups


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
