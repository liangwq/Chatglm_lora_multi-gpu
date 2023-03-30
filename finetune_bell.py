from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    

#mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-5-srp45.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp

#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.jsonfrom transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer,AutoConfig,AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field

import datasets
import os
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
import gc
import psutil
import threading
from accelerate import notebook_launcher
from alps.util import logger
from alps.pytorch.api.utils.web_access import patch_requests
patch_requests()

EOS_ID = 150005
accumulate_step = 2
mixed_precision = 'fp16'
MAX_LENGTH = 1024

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device


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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir='./', trust_remote_code=True)


@dataclass 
class FinetuneArguments:
    dataset_path: str = field(default="data/Belle_0_1.train.json")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )



class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    #attention_mask = torch.ones((1, context_length, context_length))
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                #torch.zeros(seq_length, dtype=torch.long)
                torch.zeros(seq_length, dtype=torch.long, device=device),
            
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        #position_ids = torch.arange(context_length, dtype=torch.long)
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids



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



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)

def main():
    #TrainingArguments.overwrite_output_dir = 'output'
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    #training_args.output_dir = 'output'


    # init model load_in_8bit=True, , device_map="cuda:0". , device_map="auto"
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b",cache_dir ='./',trust_remote_code=True)
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


    #with open('data/belle_open_source_1M.train.json', 'r') as f:
    with open(training_args.dataset_path, 'r') as f:
        content =[]
        for line in f.readlines():##readlines(),函数把所有的行都读取进来；  
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


    
    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.add_callback(ClearCacheCallback(1000))

    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )
    model.save_pretrained("chatglm-lora-6b")

if __name__ == "__main__":
    main()                    


#torchrun --nproc_per_node=8 multi_gpu_fintune_belle.py --dataset_path data/Belle_0_1.train.json --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1  --save_steps 2000 --save_total_limit 2 --learning_rate 2e-5 --fp16 --num_train_epochs 2 --remove_unused_columns false --logging_steps 50 --gradient_accumulation_steps 2 --output_dir output --deepspeed ds_config_zero3.json
