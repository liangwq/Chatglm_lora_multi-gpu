from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

# Model Repository on huggingface.co





#加载数据
df = pd.read_csv("yitu_origin.csv")
df = pd.DataFrame(df)

#数据加载到Dataloader，按批装载，batch_zise根据自己显卡大小改写，num_workers根据自己gpu卡数改写
test_loader = DataLoader(dataset=df['tittle'] , batch_size=6, shuffle=False,num_workers=2)

#装载模型，用deepspeed提速
model_id = "THUDM/chatglm-6b"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir ='./', trust_remote_code=True)

#多GPU模型加载
def load_model_on_gpus(checkpoint_path, num_gpus=2):
    # 总共占用13GB显存,28层transformer每层0.39GB左右
    # 第一层 word_embeddings和最后一层 lm_head 层各占用1.2GB左右
    num_trans_layers = 28
    vram_per_layer = 0.39
    average = 13/num_gpus
    used = 1.2
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': num_gpus-1, 'lm_head': num_gpus-1}
    gpu_target = 0
    for i in range(num_trans_layers):
        if used > average-vram_per_layer/2 and gpu_target < num_gpus:
            gpu_target += 1
            used = 0
        else:
            used += vram_per_layer
        device_map['transformer.layers.%d' % i] = gpu_target

    model = ChatGLMForConditionalGeneration.from_pretrained(
        checkpoint_path,trust_remote_code=True).half()
    model = model.eval()
    model = load_checkpoint_and_dispatch(
        model, checkpoint_path, device_map=device_map, offload_folder="offload", offload_state_dict=True).half()
    return model

model = load_model_on_gpus("/ossfs/workspace/ChatGLM-Tuning/models--THUDM--chatglm-6b/snapshots/chatglm", num_gpus=2)


#批量生成文案
test_preds = []
max_length = 1024
def yitu_generator(batch):
    batch_0 = []
    for text in batch:
        instruct = f"你是写手，把'{text}'把这句话做10种续写\n"
        batch_0.append(instruct)
    input_tokens = tokenizer.batch_encode_plus(batch_0, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    return input_tokens
    
for  batch in enumerate(test_loader):
    batch = batch[1]
    '''input_tokens = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())'''
   
    input_tokens = yitu_generator(batch)
    outputs = model.generate(**input_tokens, do_sample=True,max_length=1024,temperature=0.4)

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    test_preds.extend(outputs)
    if len(test_preds) > 100:
        break
test_preds
