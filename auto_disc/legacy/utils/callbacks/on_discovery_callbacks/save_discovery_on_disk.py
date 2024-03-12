import json
import os
import pickle
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, Type

import numpy as np
import torch
from auto_disc.legacy.utils.callbacks.on_discovery_callbacks.save_discovery import (
    SaveDiscovery,
)
from auto_disc.utils.leaf.Leaf import Leaf


class SaveDiscoveryOnDisk(SaveDiscovery):
    def __call__(
        self,
        resource_uri: str,
        experiment_id: int,
        run_idx: int,
        seed: int,
        discovery: Dict[str, Any],
    ) -> None:
        return super().__call__(resource_uri, experiment_id, run_idx, seed, discovery)

    @staticmethod
    def _dump_json(
        discovery: Dict[str, Any],
        dir_path: str,
        json_encoder: Type[json.JSONEncoder],
        **kwargs,
    ) -> None:
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
        if not os.path.exists(disc_path):
            os.mkdir(disc_path)
        dir_str = f"{date_str}_exp_{experiment_id}_idx_{run_idx}_seed_{seed}"
        dir_path = os.path.join(disc_path, dir_str)

        # initialize
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        return dir_path

    @classmethod
    def _save_binary_callback(cls: Type, binary: bytes, save_dir: str) -> str:
        file_name = sha1(binary).hexdigest()
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(binary)
        return file_name
