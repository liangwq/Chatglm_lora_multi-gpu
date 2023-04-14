from __future__ import annotations

import gc
import json
import pathlib
import sys

import gradio as gr
import PIL.Image
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraModel, LoraConfig, set_peft_model_state_dict


class InferencePipeline:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_path = None

    def clear(self) -> None:
        self.weight_path = None
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_lora_weight_path(name: str) -> pathlib.Path:
        curr_dir = pathlib.Path(__file__).parent
        return curr_dir / name, curr_dir / f'{name.replace(".pt", "_config.json")}'

    def load_and_set_lora_ckpt(self, pipe, weight_path, config_path, dtype):
        with open(config_path, "r") as f:
            lora_config = json.load(f)
        lora_checkpoint_sd = torch.load(weight_path, map_location=self.device)
        unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
        text_encoder_lora_ds = {
            k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
        }

        unet_config = LoraConfig(**lora_config["peft_config"])
        pipe.unet = LoraModel(unet_config, pipe.unet)
        set_peft_model_state_dict(pipe.unet, unet_lora_ds)

        if "text_encoder_peft_config" in lora_config:
            text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
            pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
            set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()

        pipe.to(self.device)
        return pipe

    def load_pipe(self, model_id: str, lora_filename: str) -> None:
        weight_path, config_path = self.get_lora_weight_path(lora_filename)
        if weight_path == self.weight_path:
            return

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        pipe = pipe.to(self.device)
        pipe = self.load_and_set_lora_ckpt(pipe, weight_path, config_path, torch.float16)
        self.pipe = pipe

    def run(
        self,
        base_model: str,
        lora_weight_name: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        n_steps: int,
        guidance_scale: float,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error("CUDA is not available.")

        self.load_pipe(base_model, lora_weight_name)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt if negative_prompt else None,
        )  # type: ignore
        return out.images[0]
