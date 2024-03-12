import json
import os
import pickle

import matplotlib.pyplot as plt
import requests
from auto_disc.legacy.utils.callbacks.on_save_callbacks import BaseOnSaveCallback
from torch import Tensor


class OnSaveModulesOnDiskCallback(BaseOnSaveCallback):
    """
    class for save autodisc modules on disk.
    """

    def __init__(self, folder_path: str, **kwargs) -> None:
        """
        init the callback with a path to save modules on disk

        Args:
            folder_path: path to folder where we want save the discovery
        """
        super().__init__(**kwargs)
        self.folder_path = folder_path

    def __call__(self, **kwargs) -> None:
        """
        Save modules on disk

        Args:
            kwargs: run_idx, experiment_id, seed, system, input_wrappers...
        """
        # TODO convert to_save_modules --> self.to_save_modules (like on_discovery_*_callback)
        to_save_modules = [
            "system",
            "explorer",
            "input_wrappers",
            "output_representations",
            "in_memory_db",
        ]

        for save_module in to_save_modules:
            if isinstance(kwargs[save_module], list):
                to_pickle = []
                for element in kwargs[save_module]:
                    to_pickle.append(element.save())
            else:
                to_pickle = kwargs[save_module].save()

            folder = "{}{}/{}/{}".format(
                self.folder_path, kwargs["experiment_id"], kwargs["seed"], save_module
            )
            filename = "{}/idx_{}.pickle".format(folder, kwargs["run_idx"])

            if not os.path.isdir(folder):
                print(folder)
                os.makedirs(folder)
            with open(filename, "wb") as out_file:
                pickle.dump(to_pickle, out_file)

        folder = "{}{}/{}/".format(
            self.folder_path, kwargs["experiment_id"], kwargs["seed"]
        )
        self.logger.info(
            "New modules saved : {} : {} :{}".format(
                folder, to_save_modules, kwargs["run_idx"]
            )
        )
