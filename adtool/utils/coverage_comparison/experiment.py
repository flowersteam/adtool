import json
from pathlib import Path
from typing import Any, Dict, Tuple
from pydoc import locate as _locate


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as handle:
        return json.load(handle)


def build_system_and_explorer(experiment_config: Dict[str, Any]) -> Tuple[Any, Any]:
    system_cls = _locate(experiment_config["system"]["path"])
    if system_cls is None:
        raise ValueError(
            f"Could not retrieve class from path: {experiment_config['system']['path']}."
        )

    explorer_factory_cls = _locate(experiment_config["explorer"]["path"])
    if explorer_factory_cls is None:
        raise ValueError(
            f"Could not retrieve class from path: {experiment_config['explorer']['path']}."
        )

    system = system_cls(**experiment_config["system"]["config"])
    explorer_factory = explorer_factory_cls(**experiment_config["explorer"]["config"])
    explorer = explorer_factory(system)
    return system, explorer
