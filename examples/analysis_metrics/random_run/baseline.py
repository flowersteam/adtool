from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pydoc import ErrorDuringImport, locate
from typing import Any

import numpy as np

from ..import_paths import ensure_adtool_examples_alias


@dataclass(frozen=True)
class RandomRunSummary:
    output_dir: Path
    discoveries_dir: Path
    count: int
    seed: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _locate_class(path: str) -> type:
    ensure_adtool_examples_alias()
    cls = None
    try:
        cls = locate(path)
    except ErrorDuringImport:
        raise

    if cls is None and path.startswith("adtool.examples."):
        fallback_path = f"examples.{path[len('adtool.examples.'):]}"
        cls = locate(fallback_path)

    if cls is None:
        raise ValueError(f"Could not retrieve class from path: {path}.")
    if not isinstance(cls, type):
        raise ValueError(f"Path must resolve to a class: {path}.")
    return cls


def _build_system_and_explorer(config: dict[str, Any]) -> tuple[Any, Any]:
    system_config = config["system"]
    explorer_config = config["explorer"]

    system_cls = _locate_class(system_config["path"])
    explorer_factory_cls = _locate_class(explorer_config["path"])

    system = system_cls(**system_config.get("config", {}))
    explorer_factory = explorer_factory_cls(**explorer_config.get("config", {}))
    return system, explorer_factory(system)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    return value


def _observe_behavior(system: Any, explorer: Any, params: Any) -> dict[str, Any]:
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


def _minimal_discovery(explorer: Any, observed: dict[str, Any]) -> dict[str, Any]:
    param_key = getattr(explorer, "postmap_key", "params")
    behavior_key = getattr(explorer, "premap_key", "output")
    if behavior_key not in observed:
        raise ValueError(f"Behavior key '{behavior_key}' missing after behavior mapping")

    discovery = {
        param_key: _to_jsonable(observed.get(param_key)),
        behavior_key: _to_jsonable(observed[behavior_key]),
        "equil": 1,
    }
    raw_behavior_key = f"raw_{behavior_key}"
    if raw_behavior_key in observed:
        discovery[raw_behavior_key] = _to_jsonable(observed[raw_behavior_key])
    return discovery


def _save_discovery(discoveries_dir: Path, run_idx: int, seed: int, discovery: dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    discovery_dir = discoveries_dir / f"{timestamp}_idx_{run_idx}_seed_{seed}"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    with (discovery_dir / "discovery.json").open("w") as handle:
        json.dump(discovery, handle)


def run_random_baseline(
    config_file: Path,
    output_dir: Path,
    nb_iterations: int,
    seed: int = 42,
) -> RandomRunSummary:
    _set_seed(seed)
    config = _load_json(config_file)
    system, explorer = _build_system_and_explorer(config)

    output_dir = Path(output_dir).resolve()
    discoveries_dir = output_dir / "discoveries"
    discoveries_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(int(nb_iterations)):
        params = explorer.parameter_map.sample()
        observed = _observe_behavior(system, explorer, params)
        _save_discovery(
            discoveries_dir,
            run_idx,
            seed,
            _minimal_discovery(explorer, observed),
        )

    summary = RandomRunSummary(
        output_dir=output_dir,
        discoveries_dir=discoveries_dir,
        count=int(nb_iterations),
        seed=seed,
    )
    with (output_dir / "random_run_summary.json").open("w") as handle:
        json.dump({
            "output_dir": str(summary.output_dir),
            "discoveries_dir": str(summary.discoveries_dir),
            "count": summary.count,
            "seed": summary.seed,
        }, handle, indent=2)
    return summary
