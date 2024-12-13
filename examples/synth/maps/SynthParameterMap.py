import dataclasses
from copy import deepcopy
from typing import Dict
from adtool.utils.leaf.Leaf import Leaf
from examples.synth.systems.Synth import Synth
from examples.synth.systems.utils import Synth, WaveformType, Generator
import random

class SynthParameterMap(Leaf):
    def __init__(
        self,
        system: Synth,

        premap_key: str = "params",
     #   param_obj: SynthParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()


        # if len(config_decorator_kwargs) > 0:
        #     param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key


    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        return intermed_dict

    def sample(self) -> Dict:

        # just generator

        synth=Synth(output=Generator(
        waveform= random.choice(list(WaveformType)),
        frequency= random.uniform(1, 20) if random.random() < 0.5 else random.uniform(100, 1000),
        amplitude=1.0,
    ))
         
        json=synth.to_json()



        p_dict = {
            "dynamic_params": json


        }

        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)

        synth=Synth.from_json(intermed_dict["dynamic_params"])

        synth.mutate()

        json=synth.to_json()

        intermed_dict["dynamic_params"] = json

        return intermed_dict
    
