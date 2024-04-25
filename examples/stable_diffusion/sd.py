import torch
from adtool.maps.IdentityBehaviorMap import IdentityBehaviorMap
from adtool.maps.UniformParameterMap import UniformParameterMap
from examples.stable_diffusion.maps.TextToVectorMap import TextToVectorMap
from examples.stable_diffusion.systems.StableDiffusionPropagator import StableDiffusionPropagator

from diffusers import DiffusionPipeline
from peft import PeftModel
from diffusers import LCMScheduler, AutoPipelineForText2Image

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from transformers import AutoTokenizer

from transformers import CLIPTextModel

from diffusers import UNet2DConditionModel

from diffusers import AutoencoderKL

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def default_unet():
    from peft import PeftModel
    unet=UNet2DConditionModel.from_pretrained("segmind/tiny-sd", subfolder="unet")
    PeftModel.from_pretrained(unet, "akameswa/lcm-lora-tiny-sd")
    
    return unet

model_id = "segmind/tiny-sd"
adapter_id = "akameswa/lcm-lora-tiny-sd"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)

PeftModel.from_pretrained(pipe.unet, adapter_id)
pipe.fuse_lora()

def default_scheduler():
    scheduler= LCMScheduler.from_pretrained("segmind/tiny-sd", subfolder="scheduler")
    scheduler=LCMScheduler.from_config(pipe.scheduler.config)
    return scheduler


model_id = "segmind/tiny-sd"
adapter_id = "akameswa/lcm-lora-tiny-sd"

class FakePipe:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("segmind/tiny-sd", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("segmind/tiny-sd", subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained("segmind/tiny-sd", subfolder="vae")
        self.unet = default_unet()
        self.scheduler = default_scheduler()


class FakePipe2:
    def __init__(self):
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler


pipe = FakePipe2()

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)


PeftModel.from_pretrained(pipe.unet, adapter_id)


def test():
    #DiffusionPipeline.from_pretrained(model_name)
    txt_embedder = TextToVectorMap(seed_prompt="a realistic purple cat",tokenizer=pipe.tokenizer,text_encoder=pipe.text_encoder)
    sd = StableDiffusionPropagator(height=512,width=512,num_inference_steps=2, guidance_scale=1,vae=pipe.vae,unet=pipe.unet, scheduler=pipe.scheduler)
    id = IdentityBehaviorMap()
   # id=UniformParameterMap(tensor_low=[-1.0], tensor_high=[1.0], tensor_bound_low=[-1.0], tensor_bound_high=[1.0])

    # get initial seed image
    data = txt_embedder.map({}, use_seed_vector=True)
    #save as pickle
    # with open('txt_seed.pkl', 'wb+') as f:
    #     pickle.dump(data['params'], f)
    # with open('truetxtseed.pkl', 'rb') as f:
    #     data['params'] = pickle.load(f)
    data = sd.map(data)
    data = id.map(data)
    #save as pickle
    # with open('output.pkl', 'wb+') as f:
    #     pickle.dump(data['output'], f)
    img = sd._decode_image(data["output"])
    img.save(f"out_seed.png")

    data = sd.map(data)
    #save as pickle
    # with open('output.pkl', 'wb+') as f:
    #     pickle.dump(data['output'], f)
    img = sd._decode_image(data["output"])

    img.save(f"out_1.png")

    # # generate random samples
    for i in range(10):
        # this first step will sample around the seed_prompt
        data = txt_embedder.map(data)

        # reverse diffuse here
        data = sd.map(data, fix_seed=False)
        # save the video
        byte_vid,ext = sd.render({})
        with open(f"outvid_{i}.mp4", "wb") as f:
            f.write(byte_vid)

        # does nothing, because there is no IMGEP
        data = id.map(data)

        img = sd._decode_image(data["output"])
        img.save(f"out_{i}.png")


if __name__ == "__main__":
    test()
