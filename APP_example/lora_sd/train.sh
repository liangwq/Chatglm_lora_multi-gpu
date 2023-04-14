#export MODEL_NAME= "stabilityai/stable-diffusion-2-1-base"  #"CompVis/stable-diffusion-v1-4" #
#export INSTANCE_DIR="results/training_data"
#export CLASS_DIR="results/class_data"
#export OUTPUT_DIR="results"

#CUDA_VISIBLE_DEVICES=0 
accelerate launch  train_dreambooth.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base  \
  --instance_data_dir=results/training_data \
  --class_data_dir=results/class_data \
  --output_dir=results \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --lora_text_encoder_r 16 \
  --lora_text_encoder_alpha 17 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800
