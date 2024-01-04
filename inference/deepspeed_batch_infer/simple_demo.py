import os
import torch
import deepspeed
import numpy as np
import transformers

from time import perf_counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference

from torch.nn import Module
from typing import Dict, Union, Optional
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizer


model_name = "/root/autodl-tmp/GLM-API/model/chatglm3-6b-32k"
payload = ["DeepSpeed is a machine learning framework","一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？"]
payload0 = "DeepSpeed is a machine learning framework"

'''
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)

ds_model = deepspeed.init_inference(
    model=model,      # Transformers模型
    mp_size=2,        # GPU数量
    dtype=torch.float16, # 权重类型(fp16)
    replace_method="auto", # 让DS自动替换层
    replace_with_kernel_inject=True, # 使用kernel injector替换
)
print(f"模型加载至设备{ds_model.module.device}\n")

# 执行模型推理
input_ids = tokenizer.batch_encode_plus(payload, return_tensors="pt",padding=True)#.to(model.device)
for t in input_ids:
    if torch.is_tensor(input_ids[t]):
        input_ids[t] = input_ids[t].to(model.device)


#input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(model.device)
torch.cuda.synchronize()
logits = ds_model.generate(**input_ids, do_sample=True, max_length=100)


outputs = tokenizer.batch_decode(logits, skip_special_tokens=True)
torch.cuda.synchronize()

if int(os.getenv("LOCAL_RANK", "0")) == 0:
    for i, o in zip(payload, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
'''
        
def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM3
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map

def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model

available_gpus = torch.cuda.device_count()
model = load_model_on_gpus(model_name, available_gpus)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = model.eval()
#response, history = model.chat(tokenizer, "你好", history=[])
response, history= model.chat(tokenizer,payload0, history=[])
#response = tokenizer.decode(response0)
print(response)
