import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np


def _is_valid_embedding(embedding: np.ndarray) -> bool:
    if embedding.size == 0:
        return False
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return False
    return True


def collect_random_embeddings(system: Any, explorer: Any, count: int) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for _ in range(count):
        params_payload = explorer.parameter_map.sample()
        data = {"params": params_payload}
        data = system.map(data)
        data = explorer.behavior_map.map(data)
        embedding = np.asarray(data.get("output", []), dtype=float).reshape(-1)
        if _is_valid_embedding(embedding):
            embeddings.append(embedding)
    return embeddings


def iter_discovery_files(discovery_root: Path) -> Iterable[Path]:
    files = list(discovery_root.rglob("discovery.json"))
    files.sort(key=lambda path: path.stat().st_mtime)
    return files


def load_discovery_embeddings(discovery_root: Path) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for path in iter_discovery_files(discovery_root):
        try:
            with open(path, "r") as handle:
                discovery = json.load(handle)
        except json.JSONDecodeError:
            continue
        output = discovery.get("output", [])
        embedding = np.asarray(output, dtype=float).reshape(-1)
        if _is_valid_embedding(embedding):
            embeddings.append(embedding)
    return embeddings
