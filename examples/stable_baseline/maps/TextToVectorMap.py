#!/usr/bin/env python3
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from adtool.maps.Map import Map
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.leaf.locators.locators import BlobLocator
from transformers import CLIPTextModel, CLIPTokenizer

TORCH_DEVICE = "mps"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)



from pydantic import BaseModel, Field

class PromptParams(BaseModel):
    seed_prompt: str = Field("a photo of a dog on the beach")
    perturbation_scale: float = Field(0.001, ge=0.00001, le=0.1)

@expose
class TextToVectorMap(Map):

    config=PromptParams

    def __init__(
        self,
        premap_key: str = "prompt",
        postmap_key: str = "params",
        seed_prompt="a photo of a dog on the beach",
        perturbation_scale=0.001,
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.locator = BlobLocator()

        self.seed_prompt = seed_prompt
        self.perturbation_scale = perturbation_scale

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "segmind/tiny-sd", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "segmind/tiny-sd", subfolder="text_encoder"
        )
        self.text_encoder.to(TORCH_DEVICE)

        self.uncond_vector, self.seed_vector = self._transform_text_to_vector(
            self.seed_prompt
        )

    def map(
        self, input: Dict, override_existing: bool = True, use_seed_vector: bool = False
    ) -> Dict:
        # rename for code clarity
        data = input

        # idempotently ensure the seed_prompt is in the payload
        data[self.premap_key] = self.premap_key

        # generate the postmap text_vector
        if (override_existing and self.postmap_key in data) or (
            self.postmap_key not in data
        ):
            if use_seed_vector:
                data[self.postmap_key] = (
                    torch.cat([self.uncond_vector, self.seed_vector])
                    .detach()
                    .clone()
                    .to("cpu")
                )
            else:
                # overrides "text_vector" with new sample
                data[self.postmap_key] = self.sample().detach().clone().to("cpu")
        else:
            # passes "genome" through if it exists
            pass

        return data

    def sample(self) -> torch.Tensor:
        # random gaussian perturbation of the seed vector
        delta = (
            self.perturbation_scale
            * torch.norm(self.seed_vector)
            * torch.randn(size=self.seed_vector.size(), device=TORCH_DEVICE)
        )
        return (
            torch.cat([self.uncond_vector, self.seed_vector + delta]).detach().clone()
        )

    def mutate(self, parameter_dict: torch.Tensor) -> torch.Tensor:
        # NOTE: the parameter_dict is actually just a tensor (i.e., the trivial
        # dictionary with a single element)
        concat_dim = parameter_dict.size()[-2]
        text_vector = parameter_dict.clone()
        conditioned_text_vector = text_vector[: (concat_dim // 2)]

        delta = (
            self.perturbation_scale
            * torch.norm(conditioned_text_vector)
            * torch.randn(size=conditioned_text_vector.size())
        )

        conditioned_text_vector = conditioned_text_vector + delta
        text_vector[: (concat_dim) // 2] = conditioned_text_vector

        return text_vector

    def _transform_text_to_vector(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # get vector of the seed_prompt
        # TODO: add support/error handling for multiple prompts
        prompt = [text]
        batch_size = len(prompt)

        # generate embedding of the text prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(TORCH_DEVICE))[0]

        # for classifier guidance tuning
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(TORCH_DEVICE))[
            0
        ]

        return uncond_embeddings, text_embeddings


def test():
    m = TextToVectorMap(seed_prompt="something")
    input = {}
    output = m.map(input)
    output_tensor = output["params"]
    assert torch.norm(output_tensor) > 0

    sampled_output = m.map(output)
    sampled_output_tensor = sampled_output["params"]
    assert not torch.equal(sampled_output_tensor, output_tensor)

    backup_tensor = sampled_output_tensor
    sampled_output = m.mutate(sampled_output)
    assert not torch.equal(sampled_output["params"], backup_tensor)


if __name__ == "__main__":
    test()
