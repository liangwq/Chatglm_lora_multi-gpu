from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


tokenizer =AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", cache_dir = '.',
    trust_remote_code=True
)

model = ChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b", cache_dir = '.',
        device_map="auto",
        trust_remote_code=True).half().cuda()#.to('cuda:0')
model.to('cuda:0')
#model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", cache_dir = '/mntnlp/qian.lwq/Chatglm_t',trust_remote_code=True).half()

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        #model= ChatGLMForConditionalGeneration.from_pretrained("/mntnlp/qian.lwq/Chatglm_t/models--THUDM--chatglm-6b/snapshots/chatglm",trust_remote_code=True).half().cuda()
        response, updated_history = model.chat(
            tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        print("history: ", self.history)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = updated_history
        return response
