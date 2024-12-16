import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import imageio
import io
import matplotlib.pyplot as plt

from examples.synth.systems.utils import Synth


import soundfile as sf


class GenerationParams(BaseModel):
    max_generators: int = Field(5, ge=1, le=20)

from adtool.utils.expose_config.expose_config import expose


@expose
class SynthSimulation:
    config = GenerationParams

    def __init__(
        self,
        max_generators: int,
    ) -> None:
        self.max_generators = max_generators



    def map(self, input: Dict, fix_seed: bool = True) -> Dict:

        self.synth=Synth.from_json(input["params"]["dynamic_params"])

        print('input["params"]["dynamic_params"]',input["params"]["dynamic_params"])

        signal=self.synth.generate()

     #   sf.write("synth_output.wav", signal, sample_rate)

        input["output"] = signal
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:

        signal=self.synth.generate()
        
        image=self.synth.image()
        # get byte array of image
        byte_img = io.BytesIO()
        imageio.imwrite(byte_img, image, format="png")


        byte_signal = io.BytesIO()

        sf.write(byte_signal, signal, 44100, format="wav")


        return [(byte_img.getvalue(), "png"),
                (byte_signal.getvalue(), "wav")]
                
