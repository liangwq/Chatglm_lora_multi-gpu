from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets

from accelerate import Accelerator
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import gc
import os
import psutil
import threading

import tqdm
import joblib
import numpy as np
import pandas as pd
import loralib as lora


from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

EOS_ID = 150005
accumulate_step = 4
mixed_precision = 'bf16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir ='./', trust_remote_code=True)

class ClearCacheCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def __init__(self, steps_to_call_clear_cache=1000):
        self.steps_to_call_clear_cache = steps_to_call_clear_cache

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.global_step % self.steps_to_call_clear_cache == 0:
            logger.info(f'pid {os.getpid()} prepare to call empty_cache at global_step: {state.global_step}')
            torch.cuda.empty_cache()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        torch.cuda.empty_cache()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        torch.cuda.empty_cache()
        
@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/belle")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/belle_output', )


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)



class AlpacaDataset(Dataset):
    def __init__(self, pairs, tokenizer) -> None:
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
 
    def __getitem__(self, index):
        if self.pairs[index]['completion'][-4:] == '</s>':
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'][:-4], add_special_tokens=False)
            completion += [EOS_ID]
        else:
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'], add_special_tokens=False)

        return {'prompt':prompt, 'completion':completion}

    def __len__(self):
        return len(self.pairs)



def collate_fn(batch):
    input_ids = []
    labels = []
    position_ids = []

    _max_length = max([len(obj['prompt'])+len(obj['completion']) for obj in batch])
    attention_mask = torch.ones((len(batch), _max_length, _max_length), device=device)
    attention_mask.tril_()

    for i, obj in enumerate(batch):
        context_length = obj['prompt'].index(150004)
        attention_mask[i, :, :context_length] = 1

        to_pad = _max_length - len(obj['prompt']) - len(obj['completion'])

        input_ids.append(obj['prompt'] + obj['completion'] + [tokenizer.pad_token_id] * to_pad)

        position_ids.append(torch.stack([torch.arange(0, _max_length, device=device), 
                                         torch.concat([torch.zeros(context_length - 1, device=device), 
                                                       torch.arange(0, _max_length - context_length + 1, device=device)])]).long())

        labels.append(torch.tensor([-100] * len(obj['prompt']) + 
                                   obj['completion'] +
                                   [-100] * to_pad, device=device).long())

    
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': attention_mask, 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}



def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main():
    LR = 2e-5
    NUM_EPOCHS = 2
    warm_up_ratio = 0.1
    
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # init model , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "THUDM/chatglm-6b", cache_dir ='./',  trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    #model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )


    # setup peft
    peft_config = LoraConfig(
        peft_type="LORA", 
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    if finetune_args.is_resume and finetune_args.resume_path:
        print("=====>load lora pt from =====》:", finetune_args.is_resume, finetune_args.resume_path)
        model.load_state_dict(torch.load(finetune_args.resume_path), strict=False)

    ### Dataset
    
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{input}\n\n### Response:"
        ),}

    with open('data/belle_open_source_1M.train.json', 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；
            #print(json.loads(line)['input'])    
            content.append(json.loads(line))


    pairs = []

    for line in content:
        if line['input'] == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(line)
        completion = line['target']+'</s>'
        if len(prompt) + len(completion) < MAX_LENGTH:
            pairs.append({'prompt':prompt, 'completion':completion})           

    train_dataset = AlpacaDataset(pairs,tokenizer=tokenizer,)
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, shuffle=True, batch_size=2)


    ### Training

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
        num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    
    accelerator.print(model)
    
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    
    for epoch in range(NUM_EPOCHS):
        with TorchTracemalloc() as tracemalloc:
            #model = ModifiedTrainer(model=model,train_dataset=dataset,args=training_args,data_collator=data_collator)
            #model.add_callback(ClearCacheCallback(1000))
            model.to(device).train()
            total_loss = 0
            i = 0

            # Save the starting state
            #accelerate.save_state("my/save/path")
            for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
                if i%2000 ==0 and accelerator.is_main_process:
                    #accelerator.wait_for_everyone()
                    path = training_args.output_dir+'/checkpoint_{}'.format(i)
                    os.makedirs(path) 
                    accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), os.path.join(path, "chatglm-lora.pt"))
                    #save_tunable_parameters(model, os.path.join(path, "chatglm-lora.pt"))
                i +=1
                outputs = model(**batch)
                loss_detach = outputs.loss.detach().cpu().float()
                t.set_description(f"loss: {loss_detach}")
                total_loss += loss_detach
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss 
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
    accelerator.wait_for_everyone()
    
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        peft_model_id = f"finetune_{epoch}"
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), './saved/'+peft_model_id+'.pt')


if __name__ == "__main__":
    main()

