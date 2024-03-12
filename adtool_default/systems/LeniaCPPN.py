from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
from adtool_default.systems.Lenia import Lenia
from auto_disc.auto_disc.systems.System import System
from auto_disc.auto_disc.wrappers.CPPNWrapper import CPPNWrapper
from auto_disc.legacy.utils.config_parameters import (
    IntegerConfigParameter,
    StringConfigParameter,
)
from auto_disc.utils.leaf.locators.locators import BlobLocator


@StringConfigParameter(
    name="version",
    possible_values=["pytorch_fft", "pytorch_conv2d"],
    default="pytorch_fft",
)
@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="scale_init_state", default=1, min=1)
@IntegerConfigParameter(name="cppn_n_passes", default=2, min=1)
class LeniaCPPN(System):
    CONFIG_DEFINITION = {}

    def __init__(self):
        super().__init__()
        self.locator = BlobLocator()

        self.lenia = Lenia(
            version=self.config["version"],
            SX=self.config["SX"],
            SY=self.config["SY"],
            final_step=self.config["final_step"],
            scale_init_state=self.config["scale_init_state"],
        )

        self.cppn = CPPNWrapper(
            postmap_shape=(self.lenia.config["SY"], self.lenia.config["SX"]),
            n_passes=self.config["cppn_n_passes"],
        )

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        # turns genome into init_state
        # as CPPNWrapper is a wrapper, it operates on the lowest level
        intermed_dict["params"] = self.cppn.map(intermed_dict["params"])

        # pass params to Lenia
        intermed_dict = self.lenia.map(intermed_dict)

        return intermed_dict

    def render(self, data_dict, mode: str = "PIL_image") -> Optional[bytes]:
        return self.lenia.render(data_dict, mode=mode)
