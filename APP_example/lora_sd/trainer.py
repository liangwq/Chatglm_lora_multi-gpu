from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess

import gradio as gr
import PIL.Image
import torch


def pad_image(image: PIL.Image.Image) -> PIL.Image.Image:
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = PIL.Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = PIL.Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


class Trainer:
    def __init__(self):
        self.is_running = False
        self.is_running_message = "Another training is in progress."

        self.output_dir = pathlib.Path("results")
        self.instance_data_dir = self.output_dir / "training_data"

    def check_if_running(self) -> dict:
        if self.is_running:
            return gr.update(value=self.is_running_message)
        else:
            return gr.update(value="No training is running.")

    def cleanup_dirs(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def prepare_dataset(self, concept_images: list, resolution: int) -> None:
        self.instance_data_dir.mkdir(parents=True)
        for i, temp_path in enumerate(concept_images):
            image = PIL.Image.open(temp_path.name)
            image = pad_image(image)
            image = image.resize((resolution, resolution))
            image = image.convert("RGB")
            out_path = self.instance_data_dir / f"{i:03d}.jpg"
            image.save(out_path, format="JPEG", quality=100)

    def run(
        self,
        base_model: str,
        resolution_s: str,
        n_steps: int,
        concept_images: list | None,
        concept_prompt: str,
        learning_rate: float,
        gradient_accumulation: int,
        fp16: bool,
        use_8bit_adam: bool,
        gradient_checkpointing: bool,
        train_text_encoder: bool,
        with_prior_preservation: bool,
        prior_loss_weight: float,
        class_prompt: str,
        num_class_images: int,
        lora_r: int,
        lora_alpha: int,
        lora_bias: str,
        lora_dropout: float,
        lora_text_encoder_r: int,
        lora_text_encoder_alpha: int,
        lora_text_encoder_bias: str,
        lora_text_encoder_dropout: float,
    ) -> tuple[dict, list[pathlib.Path]]:
        if not torch.cuda.is_available():
            raise gr.Error("CUDA is not available.")

        if self.is_running:
            return gr.update(value=self.is_running_message), []

        if concept_images is None:
            raise gr.Error("You need to upload images.")
        if not concept_prompt:
            raise gr.Error("The concept prompt is missing.")

        resolution = int(resolution_s)

        self.cleanup_dirs()
        self.prepare_dataset(concept_images, resolution)

        command = f"""
        accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path={base_model}  \
            --instance_data_dir={self.instance_data_dir} \
            --output_dir={self.output_dir} \
            --train_text_encoder \
            --instance_prompt="{concept_prompt}" \
            --resolution={resolution} \
            --gradient_accumulation_steps={gradient_accumulation} \
            --learning_rate={learning_rate} \
            --max_train_steps={n_steps} \
            --train_batch_size=1 \
            --lr_scheduler=constant \
            --lr_warmup_steps=0 \
            --num_class_images={num_class_images} \
        """
        if train_text_encoder:
            command += f" --train_text_encoder"
        if with_prior_preservation:
            command += f""" --with_prior_preservation \
                --prior_loss_weight={prior_loss_weight} \
                --class_prompt="{class_prompt}" \
                --class_data_dir={self.output_dir / 'class_data'}
                """

        command += f""" --use_lora \
            --lora_r={lora_r} \
            --lora_alpha={lora_alpha} \
            --lora_bias={lora_bias} \
            --lora_dropout={lora_dropout}
            """

        if train_text_encoder:
            command += f""" --lora_text_encoder_r={lora_text_encoder_r} \
                --lora_text_encoder_alpha={lora_text_encoder_alpha} \
                --lora_text_encoder_bias={lora_text_encoder_bias} \
                --lora_text_encoder_dropout={lora_text_encoder_dropout}
                """
        if fp16:
            command += " --mixed_precision fp16"
        if use_8bit_adam:
            command += " --use_8bit_adam"
        if gradient_checkpointing:
            command += " --gradient_checkpointing"

        with open(self.output_dir / "train.sh", "w") as f:
            command_s = " ".join(command.split())
            f.write(command_s)

        self.is_running = True
        res = subprocess.run(shlex.split(command))
        self.is_running = False

        if res.returncode == 0:
            result_message = "Training Completed!"
        else:
            result_message = "Training Failed!"
        weight_paths = sorted(self.output_dir.glob("*.pt"))
        config_paths = sorted(self.output_dir.glob("*.json"))
        return gr.update(value=result_message), weight_paths + config_paths
