import random
import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choices
import matplotlib.pyplot as plt
tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline, AutoModel

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

#生成模型参数设置
sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

config = PPOConfig(
    model_name="THUDM/chatglm-6b", steps=51200, learning_rate=1.41e-5, remove_unused_columns=False, log_with="wandb"
)

txt_in_len = 5
txt_out_len = 20
seed = 1
np.random.seed(seed)

gpt2_model = AutoModel.from_pretrained(config.model_name,trust_remote_code=True)
gpt2_model_ref = create_reference_model(gpt2_model)
gpt2_tokenizer = AutoTokenizer.from_pretrained(config.model_name,trust_remote_code=True)

gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# 加载数据
#
dataset = load_dataset("imdb", split="train")
dataset = dataset.rename_columns({"text": "review", "label": "sentiment"})
# make sure the comments are are at least 500 and trim to 1000
dataset = dataset.filter(lambda x: len(x["review"]) > 500, batched=False)
dataset = dataset.map(lambda x: {"review": x["review"][:1000]}, batched=False)

dataset

#数据token处理
dataset = dataset.map(
    lambda x: {"input_ids": gpt2_tokenizer.encode(" " + x["review"], return_tensors="pt")[0, :txt_in_len]},
    batched=False,
)
dataset = dataset.map(lambda x: {"query": gpt2_tokenizer.decode(x["input_ids"])}, batched=False)
dataset = dataset[:20480]

from datasets import Dataset

dataset = Dataset.from_dict(dataset)
dataset.set_format("pytorch")
dataset[3]["input_ids"]
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#模型做ppo训练
ppo_trainer = PPOTrainer(config, gpt2_model, gpt2_model_ref, gpt2_tokenizer, dataset, data_collator=collator)


#用fintune的bert做监督器
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
else:
    device = ppo_trainer.accelerator.device
sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device)

text = "this movie was really bad!!"
output = sentiment_pipe(text, **sentiment_pipe_kwargs)
output

def extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "POSITIVE":
                positive_logits.append(torch.tensor(element["score"]))
    return positive_logits

ctrl_str = ["[negative]", "[neutral]", "[positive]"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

#reward
def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[neutral]":
            logit[i] = -2 * torch.abs(logit[i]) + 4
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": -1,
}


for epoch in range(2):
    for batch in tqdm(ppo_trainer.dataloader):
        logs, game_data, = (
            dict(),
            dict(),
        )

        #### prepend a random control token
        task_list = choices(ctrl_str, k=config.batch_size)
        game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
        query_tensors = [torch.cat((ctrl_tokens[t], input_ids)) for t, input_ids in zip(task_list, batch["input_ids"])]

        #### get response from gpt2
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-txt_out_len:])
        game_data["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### sentiment analysis
        texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
        logits = extract_pipe_output(sentiment_pipe(texts, **sentiment_pipe_kwargs))
        rewards = pos_logit_to_reward(logits, task_list)

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

for ctrl_s in ctrl_str:
    plt.hist(
        [r for r, t in zip(logs["env/reward_dist"], task_list) if t == ctrl_s], density=True, alpha=0.5, label=ctrl_s
    )
plt.legend(loc="best")
plt.title("reward distribution")
plt.grid(True)
plt.show()