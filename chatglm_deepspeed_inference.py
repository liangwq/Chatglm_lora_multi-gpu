from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from modeling_chatglm import ChatGLMForConditionalGeneration
# Model Repository on huggingface.co





#加载数据
df = pd.read_csv("yitu_origin.csv")
df = pd.DataFrame(df)

#数据加载到Dataloader，按批装载，batch_zise根据自己显卡大小改写，num_workers根据自己gpu卡数改写
test_loader = DataLoader(dataset=df['tittle'] , batch_size=6, shuffle=False,num_workers=1)

#装载模型，用deepspeed提速
model_id = "THUDM/chatglm-6b"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir ='./', trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(model_id, cache_dir ='./', trust_remote_code=True,torch_dtype=torch.float16)


# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
print(f"model is loaded on device {ds_model.module.device}")

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
