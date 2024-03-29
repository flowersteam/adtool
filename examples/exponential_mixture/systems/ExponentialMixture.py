import io
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from adtool.systems.System import System

from adtool.utils.leaf.locators.locators import BlobLocator
import sys

matplotlib.use("Agg")

from pydantic import BaseModel
#import the Fields class from pydantic
from pydantic.fields import Field


class SystemParams(BaseModel):
    sequence_max: float = Field(100.0, ge=0.0, le=1000.0)
    sequence_density: int = Field(100, ge=1, le=1000)





print("SystemParams", SystemParams, file=sys.stderr)

@SystemParams.expose_config()
class ExponentialMixture(System):
    def __init__(self, sequence_max=100.0, sequence_density=100):
        print("ExponentialMixture.__init__", 
              sequence_max, type(sequence_max),
              file=sys.stderr)
        super().__init__()
        self.sequence_max = sequence_max
        self.sequence_density = sequence_density

        # this module is stateless
        self.locator = BlobLocator()

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        param_tensor = self._process_dict(input)

        _, y_tensor = self._tensor_map(
            param_tensor, self.sequence_max, self.sequence_density
        )

        intermed_dict["output"] = y_tensor

        return intermed_dict

    def render(self, input_dict: Dict) -> bytes:
        """
        Renders an image given a dict with the `output` key and relevant config
        """
        x_tensor = torch.linspace(
            start=0.0, end=self.sequence_max, steps=self.sequence_density
        )
        y_tensor = input_dict["output"]

        output_binary = io.BytesIO()
        plt.plot(x_tensor, y_tensor)
        plt.savefig(output_binary)
        plt.clf()

        return output_binary.getvalue()

    def _process_dict(self, input_dict: Dict) -> torch.Tensor:
        # extract param tensor
        param_tensor = input_dict["params"]
        assert len(param_tensor.size()) == 1

        return param_tensor

    def _tensor_map(
        self, param_tensor: torch.Tensor, sequence_max: float, sequence_density: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.linspace(start=0.0, end=sequence_max, steps=sequence_density)

        mixture_tensor = torch.exp(torch.outer(param_tensor, -1 * x_tensor))

        y_tensor = torch.sum(mixture_tensor, dim=0)

        return x_tensor, y_tensor


def test():
    assert len(ExponentialMixture.CONFIG_DEFINITION) > 0



if __name__ == "__main__":
    test()
