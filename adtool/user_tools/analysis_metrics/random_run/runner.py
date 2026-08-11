import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adtool.utils.factory import instantiate_object
from .discovery import (
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


def _load_json(path):
    with Path(path).open("r") as handle:
        return json.load(handle)


def analysis_runs_directory(config: dict) -> Path:
    """Return the analysis output root configured for an experiment."""
    try:
        save_location = config["experiment"]["config"]["save_location"]
    except KeyError as exc:
        raise ValueError(
            "Random analysis requires experiment.config.save_location."
        ) from exc

    if not isinstance(save_location, str) or not save_location.strip():
        raise ValueError("experiment.config.save_location must be a non-empty path.")

    return (Path(save_location).expanduser().resolve() / "analysis_runs")


def create_random_run_directory(config: dict) -> Path:
    """Create an isolated random-analysis directory below ``analysis_runs``."""
    runs_root = analysis_runs_directory(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = runs_root / f"random_run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _build_system_and_explorer(config):
    system_config = config["system"]
    explorer_config = config["explorer"]
    system = instantiate_object(system_config, object_name="system")
    explorer_factory = instantiate_object(
        explorer_config,
        object_name="explorer factory",
    )
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
    nb_iterations,
    seed=42,
):
    _set_seed(seed)
    config = _load_json(config_file)
    system, explorer = _build_system_and_explorer(config)

    output_dir = create_random_run_directory(config)
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
