import peft
import loralib as lora

from transformers import AutoModel, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

from peft import get_peft_model, LoraConfig, TaskType
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()
import torch
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=True,
    r=32,
    lora_alpha=32, 
    target_modules=["q", "v"],
    lora_dropout=0.1
)


class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data

        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data
    
    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)


# reload the model
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",cache_dir='./' ,trust_remote_code=True)

# convert it again
for key, module in model.named_modules():
    if key.endswith('attention'):
        try:
            qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
            qkv_layer.update(module.query_key_value)
            module.query_key_value = qkv_layer
        except:
            pass
        module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)


# load the LoRA checkpoint
model.load_state_dict(torch.load('../chatglm-lora.pt'), strict=False)


# merge weights
for layer in model.transformer.layers:
    if hasattr(layer.attention.query_key_value.model.linear_q,'merge_weights'):
        layer.attention.query_key_value.model.linear_q.merge_weights = True
    if hasattr(layer.attention.query_key_value.model.linear_v,'merge_weights'):
        layer.attention.query_key_value.model.linear_v.merge_weights = True

params = {
    "bos_token_id": 150004,
  "eos_token_id": 150005,
  "pad_token_id": 20003,
  "hidden_size": 4096,
  "inner_hidden_size": 16384,
  "layernorm_epsilon": 1e-05,
  "max_sequence_length": 2048,
  "model_type": "chatglm",
  "num_attention_heads": 32,
  "num_layers": 28,
  "torch_dtype": "float16",
  "transformers_version": "4.23.1",
  "vocab_size": 150528
}
n_layers = params["num_layers"]
n_heads = params["num_attention_heads"]
dim = params["hidden_size"]
def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )


new_state_dict = {}
model.train(False)

model_sd = model.state_dict()

for k, v in model_sd.items():
    new_k = k
    if k.endswith("rotary_emb.inv_freq") or "lora" in k:
        new_k =None
    if new_k  is not None:
        if ".attention.query_key_value.model.linear_q.weight" in new_k  or 'attention.query_key_value.model.linear_v.weight' in new_k :
            '''print(k)
            print(v)
            print(unpermute(v))'''
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

import os
import json
output_dir = '../checkpoint-192000'
os.makedirs(output_dir, exist_ok=True)

torch.save(new_state_dict, output_dir + "/pytorch_model.bin")

with open(output_dir + "/params.json", "w") as f:
    json.dump(params, f)
