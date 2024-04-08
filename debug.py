from peft import PeftModel
from diffusers import LCMScheduler, AutoPipelineForText2Image

#SET seed
import torch
torch.manual_seed(0)

model_id = "segmind/tiny-sd"
adapter_id = "akameswa/lcm-lora-tiny-sd"

pipe = AutoPipelineForText2Image.from_pretrained(model_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe

PeftModel.from_pretrained(pipe.unet, adapter_id)



prompt = "red circle"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.0).images[0]

