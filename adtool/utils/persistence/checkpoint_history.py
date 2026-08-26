"""Read-only helpers for checkpoint ancestry and persisted discoveries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name


def checkpoints_for(root: str | Path) -> dict[str, Checkpoint]:
    """Return all complete checkpoints below an experiment save location."""
    container = Path(root).resolve() / "checkpoints"
    checkpoints: dict[str, Checkpoint] = {}
    if not container.is_dir():
        return checkpoints
    for manifest_path in container.glob("*/manifest.json"):
        try:
            with manifest_path.open() as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict):
            checkpoint = Checkpoint(manifest_path.parent.resolve(), manifest)
            checkpoints[checkpoint.name] = checkpoint
    return checkpoints


def latest_checkpoint(checkpoints: dict[str, Checkpoint]) -> Checkpoint:
    if not checkpoints:
        raise ValueError("No checkpoints found")

    def key(checkpoint: Checkpoint) -> tuple[float, int, str]:
        raw_created_at = checkpoint.manifest.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00")).timestamp()
        except (AttributeError, ValueError):
            created_at = float("-inf")
        return created_at, int(checkpoint.manifest.get("step", -1)), checkpoint.name

    return max(checkpoints.values(), key=key)


def select_checkpoint(
    checkpoints: dict[str, Checkpoint], checkpoint_name: str | None = None
) -> Checkpoint:
    if checkpoint_name is None:
        return latest_checkpoint(checkpoints)
    name = Path(checkpoint_name).name
    try:
        return checkpoints[name]
    except KeyError as error:
        available = ", ".join(sorted(checkpoints)) or "none"
        raise ValueError(f"Unknown checkpoint {checkpoint_name!r}. Available: {available}") from error


def ancestor_chain(
    checkpoint: Checkpoint, checkpoints: dict[str, Checkpoint]
) -> list[Checkpoint]:
    """Return the selected checkpoint's branch from root to leaf."""
    chain: list[Checkpoint] = []
    current: Checkpoint | None = checkpoint
    seen: set[Path] = set()
    while current is not None:
        if current.path in seen:
            raise ValueError(f"Checkpoint parent cycle detected at {current.name}")
        seen.add(current.path)
        chain.append(current)
        parent = current.manifest.get("parent_checkpoint")
        current = checkpoints.get(Path(parent).name) if parent else None
        if parent and current is None:
            raise ValueError(f"Checkpoint {chain[-1].name} has missing parent {parent!r}")
    return list(reversed(chain))


def checkpoint_tree_payload(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    checkpoints = checkpoints_for(root)
    nodes = []
    for checkpoint in checkpoints.values():
        parent = checkpoint.manifest.get("parent_checkpoint")
        nodes.append({
            "name": checkpoint.name,
            "parent": Path(parent).name if parent else None,
            "step": checkpoint.manifest.get("step"),
            "created_at": checkpoint.manifest.get("created_at"),
        })
    return {"nodes": sorted(nodes, key=lambda node: (node["step"] or -1, node["name"]))}


def load_checkpoint_records(checkpoint: Checkpoint) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in checkpoint.manifest.get("history_files", []):
        history_path = checkpoint.path / filename
        try:
            with history_path.open("rb") as file:
                chunk = pickle.load(file)
        except (OSError, pickle.UnpicklingError, EOFError) as error:
            raise ValueError(f"Could not load checkpoint history {history_path}: {error}") from error
        if not isinstance(chunk, list) or not all(isinstance(record, dict) for record in chunk):
            raise ValueError(f"Invalid checkpoint history format in {history_path}")
        records.extend(chunk)
    return records


def load_branch_records(root: str | Path, checkpoint_name: str | None = None) -> tuple[Checkpoint, list[dict[str, Any]]]:
    checkpoints = checkpoints_for(root)
    selected = select_checkpoint(checkpoints, checkpoint_name)
    records: list[dict[str, Any]] = []
    for checkpoint in ancestor_chain(selected, checkpoints):
        checkpoint_records = load_checkpoint_records(checkpoint)
        first_run_idx = int(checkpoint.manifest.get("step", 0)) - len(checkpoint_records)
        for offset, record in enumerate(checkpoint_records):
            metadata = dict(record.get("metadata") or {})
            metadata.setdefault("run_idx", first_run_idx + offset)
            metadata.setdefault("branch_id", checkpoint.manifest.get("branch_id"))
            records.append({**record, "metadata": metadata})
    return selected, records


def checkpoint_branch_index(root: str | Path) -> dict[str, list[tuple[int, str]]]:
    """Map a pipeline branch ID to its checkpoints, ordered by saved step.

    Discoveries carry the full ``metadata.branch_id`` set by the pipeline.  It
    is the stable identity that distinguishes identical results from separate
    runs; the checkpoint folder name contains the same branch ID prefix.
    """
    index: dict[str, list[tuple[int, str]]] = {}
    for checkpoint in checkpoints_for(root).values():
        branch_id = checkpoint.manifest.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            continue
        step = int(checkpoint.manifest.get("step", 0))
        index.setdefault(branch_id, []).append((step, checkpoint.name))

    for checkpoints in index.values():
        checkpoints.sort()
    return index
