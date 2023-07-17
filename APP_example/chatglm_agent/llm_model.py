"""
DATE: 2023/5/28
AUTHOR: ZLYANG
CONTACT: zhlyang95@hotmail.com
"""

### define llm ###

from typing import List, Optional, Mapping, Any
from functools import partial

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import AutoModel, AutoTokenizer


### chatglm-6B llm ###
class ChatGLM(LLM):

    model_path: str
    max_length: int = 4096
    temperature: float = 0.1
    top_p: float = 0.7
    history: List = []
    streaming: bool = True
    model: object = None
    tokenizer: object = None

    @property
    def _llm_type(self) -> str:
        return "chatglm2-6B"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history": [],
            "streaming": self.streaming
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        add_history: bool = False
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Must call `load_model()` to load model and tokenizer!")

        if self.streaming:
            text_callback = partial(StreamingStdOutCallbackHandler().on_llm_new_token, verbose=True)
            resp = self.generate_resp(prompt, text_callback, add_history=add_history)
        else:
            resp = self.generate_resp(self, prompt, add_history=add_history)

        return resp

    def generate_resp(self, prompt, text_callback=None, add_history=True):
        resp = ""
        index = 0
        if text_callback:
            for i, (resp, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    self.history,
                    max_length=self.max_length,
                    top_p=self.top_p,
                    temperature=self.temperature
            )):
                if add_history:
                    if i == 0:
                        self.history += [[prompt, resp]]
                    else:
                        self.history[-1] = [prompt, resp]
                text_callback(resp[index:])
                index = len(resp)
        else:
            resp, _ = self.model.chat(
                self.tokenizer,
                prompt,
                self.history,
                max_length=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature
            )
            if add_history:
                self.history += [[prompt, resp]]
        return resp

    def load_model(self):
        if self.model is not None or self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda().eval()

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._identifying_params:
                self.k = v