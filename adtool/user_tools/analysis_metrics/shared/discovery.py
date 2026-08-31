from pathlib import Path

import numpy as np

from adtool.utils.persistence.discovery import (
    load_discovery_groups,
    numeric_discovery_output_matrix,
)

from .summary import CheckpointSlice, DiscoverySet


def load_discovery_set(discovery_path, checkpoint_name=None):
    groups = load_discovery_groups(
        discovery_path,
        checkpoint_name=checkpoint_name,
    )
    if len(groups) == 1 and groups[0].checkpoint is None:
        discoveries = groups[0]
        return DiscoverySet(
            path=discoveries.path,
            files=discoveries.sources,
            payloads=discoveries.payloads,
            outputs=numeric_discovery_output_matrix(discoveries),
        )

    files = []
    payloads = []
    outputs = []
    checkpoints = []
    for discoveries in groups:
        start = len(payloads)
        files.extend(discoveries.sources)
        payloads.extend(discoveries.payloads)
        outputs.append(numeric_discovery_output_matrix(discoveries))
        checkpoint = discoveries.checkpoint
        checkpoints.append(
            CheckpointSlice(
                path=checkpoint.path,
                name=checkpoint.name,
                branch_id=str(checkpoint.manifest.get("branch_id") or checkpoint.name),
                step=int(checkpoint.manifest["step"]),
                start=start,
                stop=len(payloads),
            )
        )
    return DiscoverySet(
        path=Path(discovery_path).resolve(),
        files=files,
        payloads=payloads,
        outputs=np.vstack(outputs),
        checkpoints=checkpoints,
    )


def order_sequence_by_run_idx(values, payloads, files):
    run_indices = []
    for file_path, payload in zip(files, payloads):
        metadata = payload.get("metadata")
        if metadata is None or "run_idx" not in metadata:
            raise ValueError(
                "Space coverage progression requires discovery metadata.run_idx "
                f"in {file_path}"
            )
        run_indices.append(int(metadata["run_idx"]))

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    run_indices = np.asarray(run_indices, dtype=int)
    order = np.argsort(run_indices, kind="stable")
    return matrix[order], run_indices[order]
