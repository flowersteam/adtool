from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib

import numpy as np


@dataclass(frozen=True)
class ProjectedSeries:
    values: np.ndarray
    label: str
    branch_id: str | None = None


def branch_color(branch_id: str) -> str:
    """Return a stable, vivid color derived from a checkpoint branch ID."""
    digest = hashlib.sha256(branch_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    saturation = 0.58 + digest[4] / 255 * 0.20
    value = 0.72 + digest[5] / 255 * 0.20
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def branch_labels(datasets, labels) -> dict[str, str]:
    """Resolve stable labels, promoting explicitly selected branch labels."""
    resolved: dict[str, str] = {}
    for dataset, input_label in zip(datasets, labels):
        branch_order = list(dict.fromkeys(
            checkpoint.branch_id for checkpoint in dataset.checkpoints
        ))
        if not branch_order:
            continue
        selected_branch = branch_order[-1]
        ancestor_index = 0
        for branch_id in branch_order:
            is_explicit = branch_id == selected_branch
            if is_explicit:
                candidate = input_label
            else:
                ancestor_index += 1
                candidate = f"{input_label} ancestor {ancestor_index}"
            if branch_id not in resolved or is_explicit:
                resolved[branch_id] = candidate
    return resolved


def projected_branch_series(datasets, labels, projected_values) -> list[ProjectedSeries]:
    """Group projected discoveries by branch, not by checkpoint."""
    labels_by_branch = branch_labels(datasets, labels)
    entries = []
    branch_entries = {}
    seen_checkpoints = set()

    for dataset, input_label, values in zip(datasets, labels, projected_values):
        if not dataset.checkpoints:
            entries.append({
                "parts": [values],
                "label": input_label,
                "branch_id": None,
            })
            continue

        for checkpoint in dataset.checkpoints:
            checkpoint_key = checkpoint.path.resolve()
            if checkpoint_key in seen_checkpoints:
                continue
            seen_checkpoints.add(checkpoint_key)
            entry = branch_entries.get(checkpoint.branch_id)
            if entry is None:
                entry = {
                    "parts": [],
                    "label": labels_by_branch[checkpoint.branch_id],
                    "branch_id": checkpoint.branch_id,
                }
                branch_entries[checkpoint.branch_id] = entry
                entries.append(entry)
            entry["parts"].append(values[checkpoint.start:checkpoint.stop])

    return [
        ProjectedSeries(
            values=np.concatenate(entry["parts"], axis=0),
            label=(
                labels_by_branch.get(entry["branch_id"], entry["label"])
                if entry["branch_id"] is not None
                else entry["label"]
            ),
            branch_id=entry["branch_id"],
        )
        for entry in entries
    ]
