import json
import os
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, Type

from adtool.callbacks.on_discovery_callbacks.save_discovery import (
    SaveDiscovery,
)


class SaveDiscoveryOnDisk(SaveDiscovery):
    def __call__(
        self,
        config: Dict[str, Any],
        resource_uri: str,
        experiment_id: int,
        run_idx: int,
        seed: int,
        discovery: Dict[str, Any],
        rendered_outputs=None
    ) -> None:
        # save binary file
        dir_path = self._initialize_save_path(
            resource_uri, experiment_id, run_idx, seed
        )
        self._write_shared_config(resource_uri, config)
        if rendered_outputs is not None:
            discovery["rendered_outputs"] = []
            for rendered_output in rendered_outputs:
                rendered_output_name = self._save_binary_callback(
                    rendered_output[0],
                    dir_path,
                    rendered_output[1],
                    name="visu",
                )
                discovery["rendered_outputs"].append(rendered_output_name)

        return super().__call__(resource_uri, experiment_id, run_idx, seed, discovery)

    @staticmethod
    def _write_shared_config(resource_uri: str, config: Dict[str, Any]) -> None:
        discoveries_path = os.path.join(resource_uri, "discoveries")
        os.makedirs(discoveries_path, exist_ok=True)
        config_path = os.path.join(discoveries_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing_config = json.load(f)
            if existing_config == config:
                return
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def _dump_json(
        discovery: Dict[str, Any],
        dir_path: str,
        json_encoder: Type[json.JSONEncoder],
        run_idx: int = None,
        experiment_id: int = None,
        seed: int = None,
        **kwargs,
    ) -> None:
        discovery["metadata"] = {
            "run_idx": run_idx,
            "experiment_id": experiment_id,
            "seed": seed,
            "created_at": datetime.now().isoformat(),
        }
        # save dict_data to disk as JSON object
        file_path = os.path.join(dir_path, "discovery.json")
        with open(file_path, "w") as f:
            json.dump(discovery, f, cls=json_encoder)

    @staticmethod
    def _initialize_save_path(
        resource_uri: str, experiment_id: int, run_idx: int, seed: int
    ) -> str:
        dt = datetime.now()
        date_str = dt.isoformat(timespec="minutes")
        disc_path = os.path.join(resource_uri, "discoveries")
        os.makedirs(disc_path, exist_ok=True)
        dir_str = f"{date_str}_exp_{experiment_id}_idx_{run_idx}_seed_{seed}"
        dir_path = os.path.join(disc_path, dir_str)

        # initialize
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        return dir_path

    def _save_binary_callback(
        cls: Type,
        binary: bytes,
        save_dir: str,
        extension: str,
        name=None,
    ) -> str:
        if name is None:
            file_name = sha1(binary).hexdigest() + f".{extension}"
        else:
            file_name = f"{name}.{extension}"
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(binary)
        return file_name
