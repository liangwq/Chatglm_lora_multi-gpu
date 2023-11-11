import warnings
warnings.filterwarnings('ignore')

# 设置OpenAI API密钥
import os
#os.environ["OPENAI_API_KEY"] = 'Your Key'

# 构建两个场景的模板
flower_care_template = """
你是一个无人机集群设计专家，擅长解答关于无人机集群不同场景下设计方面的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位无人机集群维护专家，擅长解答关于无人机集群养护保养维护的问题。
下面是需要你来回答的问题:
{input}
"""

# 构建提示信息
prompt_infos = [
    {
        "key": "flower_care",
        "description": "适合回答关于无人机集群设计的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_decoration",
        "description": "适合回答关于无人机集群养护的问题",
        "template": flower_deco_template,
    }
]

# 初始化语言模型
from langchain.llms import OpenAI
llm = OpenAI()

# 构建目标链
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

chain_map = {}

for info in prompt_infos:
    prompt = PromptTemplate(
        template=info['template'],
        input_variables=["input"]
    )
    print("目标提示:\n", prompt)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )
    chain_map[info["key"]] = chain

# 构建路由链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate

destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RounterTemplate.format(destinations="\n".join(destinations))
print("路由模板:\n", router_template)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
print("路由提示:\n", router_prompt)

router_chain = LLMRouterChain.from_llm(
    llm,
    router_prompt,
    verbose=True
)

# 构建默认链
from langchain.chains import ConversationChain

default_chain = ConversationChain(
    llm=llm,
    output_key="text",
    verbose=True
)

# 构建多提示链
from langchain.chains.router import MultiPromptChain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True
)

# 测试1
print(chain.run("如何为高海拔地区设计无人机集群方案？"))
# 测试2              
print(chain.run("如何养护才能延长东南沿海无人机集群使用寿命？"))
# 测试3         
print(chain.run("如何区分蜂群无人机集群和混编多功能无人机集群？"))
目标提示:
 input_variables=['input'] template='\n你是一个无人机集群设计专家，擅长解答关于无人机集群不同场景下设计方面的问题。\n下面是需要你来回答的问题:\n{input}\n'
目标提示:
 input_variables=['input'] template='\n你是一位无人机集群维护专家，擅长解答关于无人机集群养护保养维护的问题。\n下面是需要你来回答的问题:\n{input}\n'
路由模板:
 Given a raw text input to a language model select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}
REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts. REMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >> flower_care: 适合回答关于无人机集群设计的问题 flower_decoration: 适合回答关于无人机集群养护的问题

<< INPUT >> {input}

<< OUTPUT (must include json at the start of the response) >> << OUTPUT (must end with) >>

路由提示: input_variables=['input'] output_parser=RouterOutputParser() template='Given a raw text input to a language model select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.\n\n<< FORMATTING >>\nReturn a markdown code snippet with a JSON object formatted to look like:\njson\n{{\n "destination": string \\ name of the prompt to use or "DEFAULT"\n "next_inputs": string \\ a potentially modified version of the original input\n}}\n\n\nREMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.\nREMEMBER: "next_inputs" can just be the original input if you don\'t think any modifications are needed.\n\n<< CANDIDATE PROMPTS >>\nflower_care: 适合回答关于无人机集群设计的问题\nflower_decoration: 适合回答关于无人机集群养护的问题\n\n<< INPUT >>\n{input}\n\n<< OUTPUT (must include json at the start of the response) >>\n<< OUTPUT (must end with) >>\n'

Entering new MultiPromptChain chain...
Entering new LLMRouterChain chain...
Finished chain. flower_care: {'input': '如何为高海拔地区设计无人机集群方案？'}
Entering new LLMChain chain... Prompt after formatting:
你是一个无人机集群设计专家，擅长解答关于无人机集群不同场景下设计方面的问题。 下面是需要你来回答的问题: 如何为高海拔地区设计无人机集群方案？

Finished chain.
Finished chain.
针对高海拔地区的无人机集群设计方案，应考虑以下几点：

1.电池：无人机在高海拔地区的电池将会更容易耗尽，因此，要特别注意电池的类型和容量，以保证可靠的航行时间。

2.外部温度：高海拔地区的外部温度可能会影响无人机的运行性能，因此，要选择具有

Entering new MultiPromptChain chain...
Entering new LLMRouterChain chain...
Finished chain. flower_care: {'input': '如何养护才能延长东南沿海无人机集群的使用寿命？'}
Entering new LLMChain chain... Prompt after formatting:
你是一个无人机集群设计专家，擅长解答关于无人机集群不同场景下设计方面的问题。 下面是需要你来回答的问题: 如何养护才能延长东南沿海无人机集群的使用寿命？

Finished chain.
Finished chain.
1.定期检查无人机集群，并定期校准其飞行控制系统，以确保其功能正常。 2.使用专业的清洁液或清洁棉球清洁无人机集群的外壳，以防止污垢堆积。 3.定期更换无人机集群电池，以防止电池过度耗尽。 4.定期更换无人机集群的推进器，以确保其飞行效率

Entering new MultiPromptChain chain...
Entering new LLMRouterChain chain...
Finished chain. None: {'input': '如何区分蜂群无人机集群和混编多功能无人机集群？'}
Entering new ConversationChain chain... Prompt after formatting: The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:

Human: 如何区分蜂群无人机集群和混编多功能无人机集群？ AI:

Finished chain.
Finished chain. 哦，这个问题很有趣。蜂群无人机集群是由多架无人机组成的网络，它们可以完成各种复杂的任务，比如监视、搜索和拦截。而混编多功能无人机集群则是由一组拥有不同功能的无人机组成，比如侦察、搜索和拦截。它们可以协同工作，实现复杂的任务。 ```