from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from knowledge_extract_tool import LLM_KG_Generator

origin_sentence = "John昨天在纽约的咖啡馆见到了他的朋友Merry。他们一起喝咖啡聊天，计划着下周去加利福尼亚（California）旅行。他们决定一起租车并预订酒店。他们先计划在下周一去圣弗朗西斯科参观旧金山大桥，下周三去洛杉矶拜访Merry的父亲威廉。"
ie_cases = {
    "我将给你个输入，请根据事件类型列表：['旅游行程']，论元角色列表：['旅游地点', '旅游时间', '旅游人员']，从输入中抽取出可能包含的事件，并以(事件触发词,事件类型,[(事件论元,论元角色)])的形式回答。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
    "从给定的文本中提取出可能的实体和实体类型，可选的实体类型为['地点', '人名']，以（实体，实体类型）的格式回答。"+"[input]"+origin_sentence: {"num_beams":2, "temperature": 0.4, "repetition_penalty":1.3},
    "我希望你根据关系列表从给定的输入中抽取所有可能的关系三元组，并以JSON字符串[{'head':'', 'relation':'', 'tail':''}, ]的格式回答，relation可从列表['位于', '别名', '朋友', '父亲', '女儿']中选取。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
    "我希望你根据关系列表从给定的输入中抽取所有可能的关系三元组，并以\"输入中包含的关系三元组是：关系1：头实体1，尾实体1；关系2：头实体2，尾实体2。\"的格式回答，关系列表=['位于', '别名', '朋友', '父亲', '女儿']。"+"[input]"+origin_sentence: {"num_beams":4, "temperature": 0.2, "repetition_penalty":1.3},
}
llm = LLM_KG_Generator(model = "/root/autodl-tmp/KnowLM/zhixi",lora ='/root/autodl-tmp/KnowLM/lora', prompter = "../finetune/lora/templates/alpaca.json" )
for instruction in ie_cases:
    cfg = ie_cases[instruction]
    print(instruction,cfg)
    print(llm._call_func(instruction,cfg ))
    
'''
<s> Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
我将给你个输入，请根据事件类型列表：['旅游行程']，论元角色列表：['旅游地点', '旅游时间', '旅游人员']，从输入中抽取出可能包含的事件，并以(事件触发词,事件类型,[(事件论元,论元角色)])的形式回答。

### Input:
John昨天在纽约的咖啡馆见到了他的朋友Merry。他们一起喝咖啡聊天，计划着下周去加利福尼亚（California）旅行。他们决定一起租车并预订酒店。他们先计划在下周一去圣弗朗西斯科参观旧金山大桥，下周三去洛杉矶拜访Merry的父亲威廉。

### Response:
输入中包含的事件是：
(旅行,旅游行程,[(John,旅游人员),(下周一,旅游时间),(圣弗朗西斯科,旅游地点),(下周三,旅游时间),(洛杉矶,旅游地点)])</s>
输入中包含的事件是：
(旅行,旅游行程,[(John,旅游人员),(下周一,旅游时间),(圣弗朗西斯科,旅游地点),(下周三,旅游时间),(洛杉矶,旅游地点)])</s>
'''