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
    color: str | None = None


def branch_color(branch_id: str) -> str:
    """Return a stable, vivid color derived from a checkpoint branch ID."""
    digest = hashlib.sha256(branch_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    saturation = 0.58 + digest[4] / 255 * 0.20
    value = 0.72 + digest[5] / 255 * 0.20
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def branch_labels(datasets, labels) -> dict[str, str]:
    """Resolve stable labels while preserving the input dataset priority."""
    resolved: dict[str, str] = {}
    for dataset, input_label in zip(datasets, labels):
        branch_order = list(dict.fromkeys(
            checkpoint.branch_id for checkpoint in dataset.checkpoints
        ))
        if not branch_order:
            continue
        selected_branch = branch_order[-1]
        for ancestor_index, branch_id in enumerate(branch_order, start=1):
            if branch_id == selected_branch:
                candidate = input_label
            else:
                candidate = f"{input_label} ancestor {ancestor_index}"
            # The first input that contains a shared branch owns its label.
            if branch_id not in resolved:
                resolved[branch_id] = candidate
    return resolved


def displayed_branch_id(dataset, branch_id: str) -> str:
    """Collapse non-requested ancestor branches into the selected branch."""
    if not dataset.checkpoints or dataset.display_ancestors:
        return branch_id
    selected_branch_id = dataset.checkpoints[-1].branch_id
    if (
        branch_id != selected_branch_id
        and branch_id not in dataset.selected_branch_ids
    ):
        return selected_branch_id
    return branch_id


def displayed_branch_color(dataset, branch_id: str) -> str | None:
    """Return an explicit dataset color for a displayed branch, if any."""
    selected_color = dataset.selected_branch_colors.get(branch_id)
    if selected_color is not None:
        return selected_color
    if not dataset.display_ancestors:
        return dataset.color
    return None


def projected_branch_series(datasets, labels, projected_values) -> list[ProjectedSeries]:
    """Group projected discoveries by branch in input/youngest-first order."""
    labels_by_branch = branch_labels(datasets, labels)
    entries = []
    seen_checkpoints = set()

    for dataset, input_label, values in zip(datasets, labels, projected_values):
        if not dataset.checkpoints:
            entries.append({
                "parts": [values],
                "label": input_label,
                "branch_id": None,
                "color": dataset.color,
            })
            continue

        branch_entries = {}
        branch_order = []
        for checkpoint in dataset.checkpoints:
            checkpoint_key = checkpoint.path.resolve()
            if checkpoint_key in seen_checkpoints:
                continue
            seen_checkpoints.add(checkpoint_key)
            displayed_branch = displayed_branch_id(dataset, checkpoint.branch_id)
            entry = branch_entries.get(displayed_branch)
            if entry is None:
                entry = {
                    "parts": [],
                    "label": labels_by_branch[displayed_branch],
                    "branch_id": displayed_branch,
                    "color": displayed_branch_color(dataset, displayed_branch),
                }
                branch_entries[displayed_branch] = entry
                branch_order.append(displayed_branch)
            entry["parts"].append(values[checkpoint.start:checkpoint.stop])

        # Checkpoint chains are stored oldest first. Plot children first so
        # they appear above their ancestors in legends and curve order.
        entries.extend(branch_entries[branch_id] for branch_id in reversed(branch_order))

    return [
        ProjectedSeries(
            values=np.concatenate(entry["parts"], axis=0),
            label=(
                labels_by_branch.get(entry["branch_id"], entry["label"])
                if entry["branch_id"] is not None
                else entry["label"]
            ),
            branch_id=entry["branch_id"],
            color=entry["color"],
        )
        for entry in entries
    ]
