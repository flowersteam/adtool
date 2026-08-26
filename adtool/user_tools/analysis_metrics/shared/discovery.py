import numpy as np

from adtool.utils.persistence.discovery import (
    load_discoveries,
    numeric_discovery_output_matrix,
)

from .summary import DiscoverySet


def load_discovery_set(discovery_path, checkpoint_name=None):
    discoveries = load_discoveries(
        discovery_path,
        checkpoint_name=checkpoint_name,
    )
    return DiscoverySet(
        path=discoveries.path,
        files=discoveries.sources,
        payloads=discoveries.payloads,
        outputs=numeric_discovery_output_matrix(discoveries),
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
