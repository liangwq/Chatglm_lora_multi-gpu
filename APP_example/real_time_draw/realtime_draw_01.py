#!pip install -q diffusers transformers accelerate peft gradio==3.50.2

from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", cache_dir = "./",revision="fb9c5d167af11fd84454ae6493878b10bb63b067", safety_checker=None, custom_pipeline="latent_consistency_img2img")
pipe.to(torch_device="cuda", torch_dtype=torch.float16)
pipe.set_progress_bar_config(disable=True)

import gradio as gr

def generate(prompt, input_image):
    image = pipe(prompt, image=input_image, num_inference_steps=4, guidance_scale=8.0, lcm_origin_steps=50, strength=0.8).images[0]
    return image.resize((768, 768))

with gr.Blocks(title=f"Realtime Latent Consistency Model") as demo:
    with gr.Row():
        with gr.Column(scale=23):
            textbox = gr.Textbox(show_label=False, value="a close-up picture of a fluffy cat")

    with gr.Row(variant="default"):
        input_image = gr.Image(
            show_label=False,
            type="pil",
            tool="color-sketch",
            source="canvas",
            height=742,
            width=742,
            brush_radius=10.0,
        )
        output_image = gr.Image(
            show_label=False,
            type="pil",
            interactive=False,
            height=742,
            width=742,
            elem_id="output_image",
        )

    textbox.change(fn=generate, inputs=[textbox, input_image], outputs=[output_image], show_progress=False)
    input_image.change(fn=generate, inputs=[textbox, input_image], outputs=[output_image], show_progress=False)

demo.launch(inline=False, share=True,server_port=6006)
