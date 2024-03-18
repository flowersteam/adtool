from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
from examples.systems.Lenia import Lenia
from adtool.systems.System import System
from adtool.wrappers.CPPNWrapper import CPPNWrapper

from adtool.utils.leaf.locators.locators import BlobLocator

import sys
# @StringConfigParameter(
#     name="version",
#     possible_values=["pytorch_fft", "pytorch_conv2d"],
#     default="pytorch_fft",
# )
# @IntegerConfigParameter(name="SX", default=256, min=1)
# @IntegerConfigParameter(name="SY", default=256, min=1)
# @IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
# @IntegerConfigParameter(name="scale_init_state", default=1, min=1)
# @IntegerConfigParameter(name="cppn_n_passes", default=2, min=1)


from adtool.utils.expose_config.defaults import Defaults, defaults

@dataclass
class LeniaCPPNConfig(Defaults):
    version: str=defaults("pytorch_fft", domain=["pytorch_fft", "pytorch_conv2d"])
    SX: int=defaults(256, min=1)
    SY: int=defaults(256, min=1)
    final_step: int=defaults(200, min=1, max=1000)
    scale_init_state: int=defaults(1, min=1)
    cppn_n_passes: int=defaults(2, min=1)

# now import enum
from enum import Enum, auto, unique, StrEnum

LeniaCPPNVersionEnum = StrEnum("version","pytorch_fft pytorch_conv2d")


 
@LeniaCPPNConfig.expose_config()
class LeniaCPPN(System):

    def __init__(self):
        super().__init__()
        self.locator = BlobLocator()

        self.lenia = Lenia(
            version=self.config.version,
            SX=self.config.SX,
            SY=self.config.SY,
            final_step=self.config.final_step,
            scale_init_state=self.config.scale_init_state,
        )

        print("self.config.cppn_n_passes",self.config.cppn_n_passes, file=sys.stderr)

        self.cppn = CPPNWrapper(
            postmap_shape=(self.lenia.config.SY, self.lenia.config.SX),
            n_passes=self.config.cppn_n_passes,
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
