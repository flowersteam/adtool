#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pydantic import BaseModel
from PIL import Image
from adtool.utils.expose_config.expose_config import expose
from pydantic import Field

from examples.draft.maps.DraftParameterMap import DraftDynamicParams, DraftParams





@expose
class BlockSimulation:

    config = DraftParams

    def __init__(
        self,
        # parameters that define the simulation
         *args, **kwargs
    ) -> None:
        # initialize the simulation
        pass


    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        self.params = DraftDynamicParams(**input["params"]["dynamic_params"])
        # run your simulation with the given parameters and dynamic parameters

        input["output"] =  None# store  the final state of the simulation
        return input

    def render(self, data_dict, mode: str = "PIL_image") -> Tuple[bytes,str]:
        # used to render the simulation, return a list of bytes and the format like [(*bytes*, "png"), (*bytes*, "mp4")]
        return []







