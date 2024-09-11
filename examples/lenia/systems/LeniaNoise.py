from copy import deepcopy

import torch
from examples.lenia.systems.Lenia import Lenia
from adtool.systems.System import System



from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from examples.lenia.systems.Lenia import Lenia
from adtool.systems.System import System
from adtool.wrappers.CPPNWrapper import CPPNWrapper

from adtool.utils.leaf.locators.locators import BlobLocator

import sys


# now import enum
from enum import Enum, auto, unique, StrEnum



from pydantic import BaseModel
from pydantic.fields import Field
from adtool.utils.expose_config.expose_config import expose
 

class LeniaNoiseConfig(BaseModel):
    SX: int = Field(256, ge=1)
    SY: int = Field(256, ge=1)
    final_step: int = Field(200, ge=1, le=1000)
    scale_init_state: float = Field(1, ge=1)

@expose
class LeniaNoise(Lenia):

    config=LeniaNoiseConfig

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
      #  print(args,kwargs)
     #   print("LeniaCPPN", self.SX)
        self.locator = BlobLocator()

    #    self.super().__init__(*args, **kwargs)

        # self.lenia = Lenia(
        #     SX=self.config.SX,
        #     SY=self.config.SY,
        #     final_step=self.config.final_step,
        # )



    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)
        # turns genome into init_state
        # as CPPNWrapper is a wrapper, it operates on the lowest level
     #   intermed_dict["params"] = self.cppn.map(intermed_dict["params"])
        #random tensor of size (SY//scale_init_state, SX//scale_init_state, C)
        intermed_dict['params']["init_state"] = torch.rand((
            int(self.SY/self.scale_init_state),
             int(self.SX/self.scale_init_state)
           ))
        
        # pass params to Lenia
        intermed_dict = super().map(intermed_dict)
        return intermed_dict
        

    def render(self, data_dict, mode: str = "PIL_image") -> Tuple[bytes, str]:
        return super().render(data_dict, mode=mode)


