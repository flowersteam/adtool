import json
from dataclasses import asdict
from datetime import datetime

import numpy as np


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def build_discovery(explorer, observed):
    param_key = getattr(explorer, "postmap_key", "params")
    behavior_key = getattr(explorer, "premap_key", "output")
    discovery = {
        param_key: to_jsonable(observed.get(param_key)),
        behavior_key: to_jsonable(observed[behavior_key]),
        "equil": 1,
    }
    raw_behavior_key = f"raw_{behavior_key}"
    if raw_behavior_key in observed:
        discovery[raw_behavior_key] = to_jsonable(observed[raw_behavior_key])
    return discovery


def write_discovery(discoveries_dir, run_idx, seed, discovery):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    discovery_dir = discoveries_dir / f"{timestamp}_idx_{run_idx}_seed_{seed}"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    discovery["metadata"] = {
        "run_idx": run_idx,
        "experiment_id": 0,
        "seed": seed,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    with (discovery_dir / "discovery.json").open("w") as handle:
        json.dump(discovery, handle)
