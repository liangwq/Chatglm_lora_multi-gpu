from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from typing import Optional, Type
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from peft import PeftModelForCausalLM
from transformers import GenerationConfig
from prompter import Prompter

class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_func(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_func(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")

class LLM_KG_Generator(functional_Tool):
    model:str 
    lora:str
    prompter:str

    
    def evaluate(self,instruction,temperature,top_p,top_k,num_beams,max_new_tokens,repetition_penalty,model,tokenizer,prompter,**kwargs,):
        input = None
        if '[input]' in instruction:
            """only for ie"""
            input=instruction[instruction.find('[input]')+7:]
            instruction=instruction[:instruction.find('[input]')]
            print(f"instruction: {instruction}")
            print(f"input: {input}")
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,        
            repetition_penalty=repetition_penalty,     # add
            **kwargs,
        )
        print(prompt)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        return prompter.get_response(output)
    
    def get_llm_chain(self,model,lora,prompter):
        tokenizer = LlamaTokenizer.from_pretrained(model,)
        model =  LlamaForCausalLM.from_pretrained(
            #"/root/autodl-tmp/KnowLM/zhixi",'/root/autodl-tmp/KnowLM/lora',"../finetune/lora/templates/alpaca.json"
            model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model,lora,torch_dtype=torch.float16, )
        prompter = Prompter(prompter)
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        model.eval()
        model = torch.compile(model)
        return model,tokenizer,prompter
        
                 
    def _call_func(self, instruction,cfg) -> str:
        model,lora,prompter=self.model,self.lora,self.prompter
        model,tokenizer,prompter = self.get_llm_chain(model,lora,prompter)
        top_p = 0.75
        top_k = 40
        max_new_tokens = 512

        s = self.evaluate(instruction,cfg["temperature"],0.75,40,cfg["num_beams"],512,cfg["repetition_penalty"],model,tokenizer,prompter)
        return s