import json
import os
from datetime import datetime
from uuid import uuid1

from adtool.auto_disc.utils.callbacks.on_save_callbacks.save_leaf_callback import (
    SaveLeaf,
)
from adtool.utils.leaf.Leaf import Leaf, LeafUID


class SaveLeafExpeDB(SaveLeaf):
    """
    Simple adapter which calls the save_leaf method from a callback
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(
        self, experiment_id: int, seed: int, module_to_save: Leaf, resource_uri: str
    ) -> None:
        # add ExpeDB entry point
        resource_uri = resource_uri + "/checkpoint_saves"
        return super().__call__(experiment_id, seed, module_to_save, resource_uri)
