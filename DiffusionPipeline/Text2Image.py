import torch
from diffusers import DiffusionPipeline
import os
import random

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

class Diffusion:
  def __init__(self,model_name="CompVis/stable-diffusion-v1-4"):

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.pipe = DiffusionPipeline.from_pretrained(model_name,
                                                  torch_dtype=torch.float16).to(self.device)

    self.pos_txt = ", high quality, photorealistic, 3D render, ultra-detailed"
    self.neg_txt = "low quality, blurry, distorted, bad anatomy, watermark, cropped, jpeg artifacts"
    
  
  def compile(self):
    torch.set_float32_matmul_precision('high')
    self.pipe.unet = torch.compile(self.pipe.unet,fullgraph=True)

    generator = torch.Generator(device=self.pipe.device).manual_seed(42)
    
    _ = self.pipe(
            prompt="Test image",
            negative_prompt="blurry",
            num_inference_steps=25,
            width=512,
            height=512,
            generator=generator
        )[0]
    
  @torch.no_grad
  def __call__(self,seed,prompt):
    seed_everything(seed)
    generator = torch.Generator(self.pipe.device).manual_seed(seed)

    output_image = self.pipe(
      prompt = prompt,
      negative_prompt = self.neg_txt,
      num_inference_steps = 25,
      width = 512,
      height = 512,
      generator=generator
    ).images[0]

    return output_image