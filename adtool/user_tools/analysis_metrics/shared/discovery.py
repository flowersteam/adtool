import json
from pathlib import Path

import numpy as np

from .summary import DiscoverySet


def load_discovery_set(discovery_path):
    discovery_path = Path(discovery_path).resolve()
    files = sorted(
        (
            path
            for path in discovery_path.rglob("discovery.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise ValueError(f"No discoveries found in {discovery_path}")

    payloads = []
    outputs = []
    for file_path in files:
        with file_path.open("r") as handle:
            payload = json.load(handle)
        payloads.append(payload)
        outputs.append(np.asarray(payload["output"], dtype=float).reshape(-1))

    return DiscoverySet(
        path=discovery_path,
        files=files,
        payloads=payloads,
        outputs=np.vstack(outputs),
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
