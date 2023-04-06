from modeling_chatglm import ChatGLMForConditionalGeneration
import torch

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", cache_dir = '.',trust_remote_code=True).half().cuda()

from transformers import AutoTokenizer
from tokenization_chatglm import ChatGLMTokenizer

tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir ='./', trust_remote_code=True)

#peft_model_id = 'mymusise/chatglm-6b-alpaca-lora'
peft_model_id = '//mntnlp/qian.lwq/Chatglm_t/output/checkpoint-3000'
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model
model.save_pretrained('output/mymusise/chatglm-6b-adapter-merged')
