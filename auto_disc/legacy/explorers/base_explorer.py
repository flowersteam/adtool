import typing
from typing import Any, Callable, Dict, Type

import torch
from auto_disc.legacy import BaseAutoDiscModule
from auto_disc.legacy.utils.spaces import BoxSpace, DictSpace


class BaseExplorer(BaseAutoDiscModule):
    """
    Base class for explorers.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def initialize(
        self,
        input_space: DictSpace,
        output_space: DictSpace,
        input_distance_fn: Callable,
    ) -> None:
        """
        Defines input and output space for the explorer (as well as a distance function for the input space).
        """
        self.input_space = input_space
        self.output_space = output_space
        self.input_distance_fn = input_distance_fn

    def sample(self):
        """
        Emits a new set of parameters to test in the system
        """
        raise NotImplementedError()

    def observe(self, parameters: Dict, observations: torch.Tensor):
        """
        Stores parameters and the output the system produced using them
        """
        raise NotImplementedError()

    def optimize(self):
        """
        Optimizes the explorer's sample policy given the discoveries arhcived
        """
        raise NotImplementedError()

    # def save(self, filepath):
    #     """
    #     Saves the explorer object using torch.save function in pickle format
    #     /!\ We intentionally empty explorer.db from the pickle
    #     because the database is already automatically saved in external files each time the explorer call self.db.add_run_data
    #     """

    #     file = open(filepath, 'wb')

    #     # do not pickle the data as already saved in extra files
    #     tmp_data = self.db
    #     self.db.reset_empty_db()

    #     # pickle exploration object
    #     torch.save(self, file)

    #     # attach db again to the exploration object
    #     self.db = tmp_data

    # @staticmethod
    # def load(explorer_filepath, load_data=True, run_ids=None, map_location='cuda'):

    #     explorer = torch.load(explorer_filepath, map_location=map_location) #TODO: deal with gpu/cpu and relative/absolute path for explorer.db.config.db_directory

    #     if load_data:
    #         explorer.db = ExplorationDB(config=explorer.db.config)
    #         explorer.db.load(run_ids=run_ids, map_location=map_location)

    #     return explorer
