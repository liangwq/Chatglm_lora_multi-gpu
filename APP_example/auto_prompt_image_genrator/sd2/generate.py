import datetime
import os
import re
from typing import Literal, Union

import streamlit as st
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)

PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img"]


@st.cache_resource(max_entries=1)
def get_pipeline(
    name: PIPELINE_NAMES,
) -> Union[
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
]:
    if name in ["txt2img", "img2img"]:
        model_id = "Lykon/DreamShaper"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id,
            cache_dir = './',
            subfolder="scheduler",
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir = './',
            scheduler=scheduler,
            #revision="fp16",
            #torch_dtype=torch.float16,
        )

        if name == "img2img":
            pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
        pipe = pipe.to("cuda")
        return pipe
    elif name == "inpaint":
        model_id = "stabilityai/stable-diffusion-2-inpainting"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            cache_dir = './',
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        return pipe


def generate(
    prompt,
    pipeline_name: PIPELINE_NAMES,
    image_input=None,
    mask_input=None,
    negative_prompt=None,
    steps=50,
    width=512,
    height=512,
    guidance_scale=7.5,
    enable_attention_slicing=False,
    enable_xformers=True
):
    """Generates an image based on the given prompt and pipeline name"""
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)
    callback = lambda step, *_: p.progress(step / steps)

    pipe = get_pipeline(pipeline_name)
    torch.cuda.empty_cache()

    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    if enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        callback=callback,
        guidance_scale=guidance_scale,
    )

    print("kwargs", kwargs)

    if pipeline_name == "inpaint" and image_input and mask_input:
        kwargs.update(image=image_input, mask_image=mask_input)
    elif pipeline_name == "txt2img":
        kwargs.update(width=width, height=height)
    elif pipeline_name == "img2img" and image_input:
        kwargs.update(
            image=image_input,
        )
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    image = pipe(**kwargs).images[0]

    os.makedirs("outputs", exist_ok=True)

    filename = (
        "outputs/"
        + re.sub(r"\s+", "_", prompt)[:50]
        + f"_{datetime.datetime.now().timestamp()}"
    )
    image.save(f"{filename}.png")
    with open(f"{filename}.txt", "w") as f:
        f.write(prompt)
    return image

