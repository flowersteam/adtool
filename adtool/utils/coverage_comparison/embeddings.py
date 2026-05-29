import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


def _is_valid_embedding(embedding: np.ndarray) -> bool:
    if embedding.size == 0:
        return False
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return False
    return True


def _get_parameter_key(explorer: Any) -> str:
    key = getattr(explorer, "postmap_key", None)
    if isinstance(key, str) and key:
        return key

    parameter_map = getattr(explorer, "parameter_map", None)
    key = getattr(parameter_map, "premap_key", None)
    if isinstance(key, str) and key:
        return key

    return "params"


def _get_behavior_key(explorer: Any) -> str:
    key = getattr(explorer, "premap_key", None)
    if isinstance(key, str) and key:
        return key
    return "output"


def collect_random_embeddings(
    system: Any,
    explorer: Any,
    count: int,
    build_embedding: Callable[[Dict[str, Any]], np.ndarray],
) -> List[np.ndarray]:
    """Collect independent random trials from parameter_map.sample().

    This intentionally does not call explorer.bootstrap(), explorer.map(),
    mutation, target sampling, or history saving. The only explorer components
    used are the parameter map for random candidate generation and the behavior
    map for projecting simulator outputs into the same behavior space as saved
    discoveries.
    """
    embeddings: List[np.ndarray] = []
    parameter_key = _get_parameter_key(explorer)
    behavior_key = _get_behavior_key(explorer)
    base_behavior_map = getattr(explorer, "behavior_map", None)

    for _ in range(count):
        params_payload = explorer.parameter_map.sample()
        data = {parameter_key: params_payload}
        data = system.map(data)

        if base_behavior_map is not None and data.get(behavior_key, None) is not None:
            try:
                behavior_map = deepcopy(base_behavior_map)
            except Exception as exc:
                raise RuntimeError(
                    "Could not copy behavior_map for independent random baseline."
                ) from exc
            data = behavior_map.map(data)

        embedding = build_embedding(data)
        if _is_valid_embedding(embedding):
            embeddings.append(embedding)
    return embeddings


def iter_discovery_files(discovery_root: Path) -> Iterable[Path]:
    files = list(discovery_root.rglob("discovery.json"))
    files.sort(key=lambda path: path.stat().st_mtime)
    return files


def load_discovery_embeddings(
    discovery_root: Path,
    build_embedding: Callable[[Dict[str, Any]], np.ndarray],
) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for path in iter_discovery_files(discovery_root):
        try:
            with open(path, "r") as handle:
                discovery = json.load(handle)
        except json.JSONDecodeError:
            continue
        embedding = build_embedding(discovery)
        if _is_valid_embedding(embedding):
            embeddings.append(embedding)
    return embeddings
