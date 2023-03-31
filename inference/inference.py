from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

#设置环境，CPU还是GPU
torch.set_default_tensor_type(torch.cuda.HalfTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
#加载base model
model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", cache_dir='./',trust_remote_code=True).half().to(device)

#设置tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir ='./', trust_remote_code=True)

#加载训练好的lora
peft_path = "output/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

#模型生成
text = "你是现代诗人，用'红包、美好、表白、夕阳、月光、慢慢'关键词生成2首表白唯美打油诗\n"
input_ids = tokenizer.encode(text, return_tensors='pt')
#model.cuda()
out = model.generate(input_ids=input_ids,max_length=2048,temperature=0.4)
answer = tokenizer.decode(out[0])
print(answer)
