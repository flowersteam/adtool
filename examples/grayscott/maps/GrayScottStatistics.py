import typing
from copy import deepcopy

import torch
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector

from adtool.utils.misc.torch_utils import roll_n
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from functools import partial

from examples.grayscott.systems.GrayScott import GrayScottSimulation

from adtool.wrappers.mutators import add_gaussian_noise

# UniformParameterMap
from adtool.maps.UniformParameterMap import UniformParameterMap

import dataclasses

EPS = 0.0001


class GrayScottStatistics(Leaf):
    """
    Outputs 3-dimensional embedding for Gray-Scott simulation.
    """

    def __init__(
        self,
        system: GrayScottSimulation,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key
        self.postmap_key = postmap_key

        self.SX = system.width
        self.SY = system.height

        self._statistic_names = ["mean", "contrast", "entropy"]
        self._n_latents = len(self._statistic_names)

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        intermed_dict = deepcopy(input)

        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(tensor[0])

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()

    def _calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        return (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

    def _calc_static_statistics(self, final_obs: torch.Tensor) -> torch.Tensor:
        feature_vector = torch.zeros(self._n_latents)
        cur_idx = 0

        feature_vector[cur_idx] = final_obs.mean()
        cur_idx += 1

        contrast = final_obs.max() - final_obs.min()
        feature_vector[cur_idx] = contrast
        cur_idx += 1

        pre_entropy = final_obs
        pre_entropy = pre_entropy - pre_entropy.min()
        pre_entropy = pre_entropy / pre_entropy.max()
        pre_entropy = pre_entropy + EPS
        entropy = -torch.sum(pre_entropy * torch.log(pre_entropy))

        feature_vector[cur_idx] = entropy

        return feature_vector


@dataclass
class GrayScottHyperParameters:
    height: int = 512
    width: int = 512
    num_inference_steps: int = 1000
    F: float = 0.04
    k: float = 0.06
    Du: float = 0.16
    Dv: float = 0.08
    initial_condition_seed: int = 0

    def to_tensor(self):
        return torch.tensor(
            [self.height, self.width, self.num_inference_steps, self.F, self.k, self.Du, self.Dv, self.initial_condition_seed]
        ).float()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(
            height=int(tensor[0].item()),
            width=int(tensor[1].item()),
            num_inference_steps=int(tensor[2].item()),
            F=tensor[3].item(),
            k=tensor[4].item(),
            Du=tensor[5].item(),
            Dv=tensor[6].item(),
            initial_condition_seed=int(tensor[7].item()),
        )


class GrayScottParameterMap(Leaf):
    def __init__(
        self,
        system: GrayScottSimulation,
        premap_key: str = "params",
        param_obj: GrayScottHyperParameters = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = GrayScottHyperParameters()

        self.locator = BlobLocator()
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=param_obj.to_tensor() - 0.1,
            tensor_high=param_obj.to_tensor() + 0.1,
        )

        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=param_obj.to_tensor(),
            std=torch.tensor([10.0, 10.0, 50.0, 0.01, 0.01, 0.01, 0.01, 1.0]).float(),
        )

        self.SX = param_obj.width
        self.SY = param_obj.height

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)

        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            intermed_dict[self.premap_key] = self.sample()
        else:
            pass

        return intermed_dict

    def sample(self) -> Dict:
        p_dyn_tensor = self.uniform.sample()

        dp = GrayScottHyperParameters.from_tensor(p_dyn_tensor)
        p_dict = {
            "dynamic_params": asdict(dp),
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        intermed_dict = deepcopy(parameter_dict)

        dp = GrayScottHyperParameters(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_tensor()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)

        intermed_dict["dynamic_params"] = asdict(
            GrayScottHyperParameters.from_tensor(mutated_dp_tensor)
        )

        return intermed_dict
