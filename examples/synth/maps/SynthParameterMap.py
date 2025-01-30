from copy import deepcopy
from typing import Dict
from adtool.utils.leaf.Leaf import Leaf
from examples.synth.systems.Synth import SynthSimulation
import random

class SynthParameterMap(Leaf):
    def __init__(
        self,
        system: SynthSimulation,

        premap_key: str = "params",
     #   param_obj: SynthParams = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        self.nb_freqs = system.nb_freqs


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
        
        # random frequencies and amplitudes
        frequencies = [random.uniform(80, 10_000) for _ in range(self.nb_freqs)]
        amplitudes = [random.uniform(0, 1) for _ in range(self.nb_freqs)]


        json = {
            "frequencies": frequencies,
            "amplitudes": amplitudes,
        }

        p_dict = {
            "dynamic_params": json
        }

        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)

        # intermed_dict["dynamic_params"]

        json = intermed_dict["dynamic_params"]

        # mutate frequencies by adding a random noise as a ratio of the frequency
        frequencies = json["frequencies"]
        for i in range(len(frequencies)):
            frequencies[i] += random.uniform(-0.05, 0.05) * frequencies[i]

        # mutate amplitudes by adding a random noise as a ratio of the amplitude
        amplitudes = json["amplitudes"]
        for i in range(len(amplitudes)):
            amplitudes[i] += random.uniform(-0.05, 0.05) * amplitudes[i]


        intermed_dict["dynamic_params"] = json

        return intermed_dict
    
