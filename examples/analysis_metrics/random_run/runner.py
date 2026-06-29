import json
import random
from dataclasses import dataclass
from pathlib import Path
from pydoc import locate

import numpy as np

from adtool.examples.analysis_metrics.random_run.discovery import (
    build_discovery,
    write_discovery,
)


@dataclass(frozen=True)
class RandomRunSummary:
    output_dir: Path
    discoveries_dir: Path
    count: int
    seed: int


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _load_json(path):
    with Path(path).open("r") as handle:
        return json.load(handle)


def _load_object(path):
    obj = locate(path)
    if obj is None:
        raise ValueError(f"Could not import {path}")
    return obj


def _build_system_and_explorer(config):
    system_config = config["system"]
    explorer_config = config["explorer"]
    system_cls = _load_object(system_config["path"])
    explorer_factory_cls = _load_object(explorer_config["path"])

    system = system_cls(**dict(system_config.get("config") or {}))
    explorer_factory = explorer_factory_cls(**dict(explorer_config.get("config") or {}))
    return system, explorer_factory(system)


def _observe_behavior(system, explorer, params):
    param_key = getattr(explorer, "postmap_key", "params")
    data = {
        param_key: params,
        "equil": 1,
    }
    data = system.map(data)
    if hasattr(explorer, "observe_results"):
        return explorer.observe_results(data)

    behavior_map = getattr(explorer, "behavior_map", None)
    if behavior_map is not None and data.get("output") is not None:
        return behavior_map.map(data)
    return data


def run_random_baseline(
    config_file,
    output_dir,
    nb_iterations,
    seed=42,
):
    _set_seed(seed)
    config = _load_json(config_file)
    system, explorer = _build_system_and_explorer(config)

    output_dir = Path(output_dir).resolve()
    discoveries_dir = output_dir / "discoveries"
    discoveries_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(int(nb_iterations)):
        params = explorer.parameter_map.sample()
        observed = _observe_behavior(system, explorer, params)
        write_discovery(
            discoveries_dir,
            run_idx,
            seed,
            build_discovery(explorer, observed),
        )

    summary = RandomRunSummary(
        output_dir=output_dir,
        discoveries_dir=discoveries_dir,
        count=int(nb_iterations),
        seed=seed,
    )
    with (output_dir / "random_run_summary.json").open("w") as handle:
        json.dump(
            {
                "output_dir": str(summary.output_dir),
                "discoveries_dir": str(summary.discoveries_dir),
                "count": summary.count,
                "seed": summary.seed,
            },
            handle,
            indent=2,
        )
    return summary
