# Chatglm_lora_multi-gpu
chatglm多gpu用deepspeed和
包括两种方式多gpu运行：
1.deepspeed
torchrun --nproc_per_node=2 multi_gpu_fintune_belle.py --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 1000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.json

2.accelerate+deepspeed 
accelerate launch --config_file accelerate_ds_zero3_cpu_offload_config.yaml  multi_gpu_fintune_belle.py --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --max_steps 10000 --save_steps 1000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --remove_unused_columns false --logging_steps 50 --output_dir output

3.ddp方式还没试