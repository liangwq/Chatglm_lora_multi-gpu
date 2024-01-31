from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
import torch
import os

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from PIL import Image
import numpy as np
import gradio as gr
import psutil
import time
import math

SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None)
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# check if MPS is available OSX only M1/M2/M3 chips
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "xpu" if xpu_available else "cpu"
)
torch_device = device
torch_dtype = torch.float16

print(f"SAFETY_CHECKER: {SAFETY_CHECKER}")
print(f"TORCH_COMPILE: {TORCH_COMPILE}")
print(f"device: {device}")

if mps_available:
    device = torch.device("mps")
    torch_device = "cpu"
    torch_dtype = torch.float32

if SAFETY_CHECKER == "True":
    i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        cache_dir = "../Qwen",
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else "fp32",
    )
    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        cache_dir = "../Qwen",
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else "fp32",
    )
else:
    i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        safety_checker=None,
        cache_dir = "../Qwen",
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else "fp32",
    )
    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        safety_checker=None,
        cache_dir = "../Qwen",
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else "fp32",
    )


t2i_pipe.to(device=torch_device, dtype=torch_dtype).to(device)
t2i_pipe.set_progress_bar_config(disable=True)
i2i_pipe.to(device=torch_device, dtype=torch_dtype).to(device)
i2i_pipe.set_progress_bar_config(disable=True)


def resize_crop(image, size=512):
    image = image.convert("RGB")
    w, h = image.size
    image = image.resize((size, int(size * (h / w))), Image.BICUBIC)
    return image


async def predict(init_image, prompt, strength, steps, seed=1231231):
    if init_image is not None:
        init_image = resize_crop(init_image)
        generator = torch.manual_seed(seed)
        last_time = time.time()
    
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))
            
        results = i2i_pipe(
            prompt=prompt,
            image=init_image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=0.0,
            strength=strength,
            width=512,
            height=512,
            output_type="pil",
        )
    else:
        generator = torch.manual_seed(seed)
        last_time = time.time()
        results = t2i_pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=512,
            height=512,
            output_type="pil",
        )
    print(f"Pipe took {time.time() - last_time} seconds")
    nsfw_content_detected = (
        results.nsfw_content_detected[0]
        if "nsfw_content_detected" in results
        else False
    )
    if nsfw_content_detected:
        gr.Warning("NSFW content detected.")
        return Image.new("RGB", (512, 512))
    return results.images[0]


css = """
#container{
    margin: 0 auto;
    max-width: 80rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    init_image_state = gr.State()
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# SDXL Turbo Image to Image/Text to Image
            ## Unofficial Demo
            SDXL Turbo model can generate high quality images in a single pass read more on [stability.ai post](https://stability.ai/news/stability-ai-sdxl-turbo).  
            **Model**: https://huggingface.co/stabilityai/sdxl-turbo
            """,
            elem_id="intro",
        )
        with gr.Row():
            prompt = gr.Textbox(
                placeholder="Insert your prompt here:",
                scale=5,
                container=False,
            )
            generate_bt = gr.Button("Generate", scale=1)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=["upload", "webcam", "clipboard"],
                    label="Webcam",
                    type="pil",
                )
            with gr.Column():
                image = gr.Image(type="filepath")
                with gr.Accordion("Advanced options", open=False):
                    strength = gr.Slider(
                        label="Strength",
                        value=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                    )
                    steps = gr.Slider(
                        label="Steps", value=2, minimum=1, maximum=10, step=1
                    )
                    seed = gr.Slider(
                        randomize=True,
                        minimum=0,
                        maximum=12013012031030,
                        label="Seed",
                        step=1,
                    )

        with gr.Accordion("Run with diffusers"):
            gr.Markdown(
                """## Running SDXL Turbo with `diffusers`
            ```bash
            pip install diffusers==0.23.1
            ```
            ```py
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/sdxl-turbo"
            ).to("cuda")
            results = pipe(
                prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe",
                num_inference_steps=1,
                guidance_scale=0.0,
            )
            imga = results.images[0]
            imga.save("image.png")
            ```
            """
            )

        inputs = [image_input, prompt, strength, steps, seed]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        strength.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        image_input.change(
            fn=lambda x: x,
            inputs=image_input,
            outputs=init_image_state,
            show_progress=False,
            queue=False,
        )

demo.queue()
demo.launch(server_port=6006)

