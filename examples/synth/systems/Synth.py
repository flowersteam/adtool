import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
import io



import soundfile as sf


class GenerationParams(BaseModel):
    nb_freqs: int = Field(10, ge=3, le=1000)

from adtool.utils.expose_config.expose_config import expose


def additive_synthesis(frequencies, amplitudes, duration=2, samplerate=16000):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes))
    return signal

@expose
class SynthSimulation:
    config = GenerationParams

    def __init__(
        self,
        nb_freqs: int,
    ) -> None:
        self.nb_freqs = nb_freqs



    def map(self, input: Dict, fix_seed: bool = True) -> Dict:


        frequencies = input["params"]['dynamic_params']['frequencies']
        amplitudes = input["params"]['dynamic_params']['amplitudes']

        self.signal = additive_synthesis(frequencies, amplitudes)

     #   sf.write("synth_output.wav", signal, sample_rate)

        input["output"] = self.signal
        return input

    def render(self, data_dict: Dict[str, Any]) -> Tuple[bytes, str]:

        byte_signal = io.BytesIO()

        sf.write(byte_signal, self.signal, 16000, format="wav")


        return [
                (byte_signal.getvalue(), "wav")]
                
