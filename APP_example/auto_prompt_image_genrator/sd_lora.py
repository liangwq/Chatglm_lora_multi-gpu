import torch
from diffusers import StableDiffusionPipeline
import torch
model_id = "/root/autodl-tmp/models--Lykon--DreamShaper/snapshots/7b50c4970b0df577d82c1d9e886b17b1da170584"

model_path = "sayakpaul/sd-model-finetuned-lora-t4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path,cache_dir= './')
pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(12736364)

prompt = ["The future world of our planet, with beautiful, immersive cities and landscapes, but with a dark, dangerous secret that could change everything.",
"A world of advanced technology, with sleek, sleek cars, spaceships, and other sleek, advanced objects that travel through space.",
"A world of机器人, with sleek, advanced机器人 that can perform all manner of tasks, from cleaning to fighting to protecting.",
"A world of virtual reality, with immersive virtual reality experiences that allow users to explore and interact with the world in a way that is both fascinating and challenging.",
"A world of energy, with sleek, advanced energy sources that power everything from sleek, advanced cars to sleek, advanced homes.",
"A world of transportation, with sleek, advanced transportation systems that allow users to easily travel from place to place, whether it's by car, plane, or ship.",
"A world of communication, with sleek, advanced communication systems that allow users to easily communicate with each other, from across the world to across the galaxy.",
"A world of medicine, with sleek, advanced medicine that can treat everything from simple health problems to complex, advanced diseases.",
"A world of education, with sleek, advanced education systems that allow users to access the latest learning resources and technologies."]

negative_prompt = 'cartoon, 3d, ((bad art)), ((b&w)), wierd colors, blurry, (((duplicate))), [out of frame], ugly, ((((watermarks))))'
image = pipe(''.join(prompt),negative_prompt =negative_prompt , generator=generator,height = 512,width=768).images[0] 
image
