#!/usr/bin/env python3
import io
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import imageio
import numpy
from pydantic import BaseModel
import torch
from adtool.systems.System import System
from adtool.utils.expose_config.expose_config import expose
from adtool.utils.leaf.locators.locators import BlobLocator
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

TORCH_DEVICE = "mps"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from typing import Dict

from pydantic import BaseModel
from pydantic.fields import Field
from adtool.systems.System import System


class GenerationParams(BaseModel):
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    num_inference_steps: int = Field(16, ge=8, le=256)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    initial_condition_seed: int = Field(0, ge=0, le=999999)


@expose
class StableDiffusionPropagator(System):
    
    config=GenerationParams

    def __init__(
        self,
        premap_key="params",
        postmap_key="output",
        height=512,
        width=512,
        num_inference_steps=16,
        guidance_scale=7.5,
        seed_prompt="a photo of a miniature schnauzer on the beach",
        initial_condition_seed=0,
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.locator = BlobLocator()

        # pipeline hyperparameters
        self.height = height  # default height of Stable Diffusion
        self.width = width  # default width of Stable Diffusion

        self.num_inference_steps = num_inference_steps  # Number of denoising steps

        self.guidance_scale = guidance_scale  # Scale for classifier-free guidance

        self.seed_prompt = seed_prompt

        self.initial_condition_seed = initial_condition_seed

        # latent space trajectory
        self.latents_over_t = []
        ## Load models
        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "segmind/tiny-sd", subfolder="vae"
        )

        # The UNet model for reverse diffusing the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "segmind/tiny-sd", subfolder="unet"
        )
        self.unet.set_attention_slice(slice_size="auto")

        # Scheduler for determining the denoising schedule
        self.scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
            set_alpha_to_one=False,
        )
        self.scheduler.set_timesteps(self.num_inference_steps)

        ## Move to GPU
        self.vae.to(TORCH_DEVICE)
        self.unet.to(TORCH_DEVICE)

    def map(self, input: Dict, fix_seed: bool = True) -> Dict:
        text_embeddings = input[self.premap_key].to(TORCH_DEVICE)
        # TODO: add support for multiple prompts in the parameter map
        batch_size = 1

        # generate initial condition
        if fix_seed:
            latents = torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=torch.Generator(TORCH_DEVICE).manual_seed(
                    self.initial_condition_seed
                ),
                device=TORCH_DEVICE,
            )
        else:
            latents = torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=torch.Generator(TORCH_DEVICE).manual_seed(
                    self.initial_condition_seed
                ),
                device=TORCH_DEVICE,
            )

        # begin reverse diffusion
        print("Starting reverse diffusion...")
        self.scheduler.set_timesteps(self.num_inference_steps)
        latents_over_t = torch.empty(
            size=tuple(list(self.scheduler.timesteps.size()) + list(latents.size())),
            device=TORCH_DEVICE,
        )

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            with torch.no_grad():
                latents_over_t[i] = latents

        self.latents_over_t = latents_over_t

        # make output dict, saving the graph pytorch node
        input[self.postmap_key] = latents_over_t[-1].detach().clone().to("cpu")
        return input

    def _decode_image(self, latent):
        # some magic scale which controls blurriness of output
        latent = latent.to(TORCH_DEVICE)
        latent = 1 / 0.18215 * latent
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    def render(self, data_dict: dict) -> bytes:
        # ignores data_dict, as the render is based on self.latents_over_t
        # in which only the last state is stored in data_dict["output"]
        imgs = []
        for i, _ in enumerate(tqdm(self.scheduler.timesteps)):
            img = self._decode_image(self.latents_over_t[i])
            imgs.append(img)
        final_result = imgs[-1]
        imgs = imgs + [final_result] * 50
        byte_img = io.BytesIO()
        imageio.mimwrite(byte_img, imgs, "mp4", fps=5, output_params=["-f", "mp4"])
        return byte_img.getvalue()
