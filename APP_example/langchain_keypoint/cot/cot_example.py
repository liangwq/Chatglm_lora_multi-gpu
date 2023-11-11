from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


template ="""
步骤1：

我有一个与{input}相关的问题。你能否提出三个不同的解决方案？请考虑各种因素，如{perfect_factors}

答案："""

prompt = PromptTemplate(
    input_variables=["input","perfect_factors"],
    template = template                      
)

chain1 = LLMChain(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    prompt=prompt,
    output_key="solutions"
)

template ="""
步骤2：

对于这三个提出的解决方案，评估它们的潜力。考虑它们的优点和缺点，初始所需的努力，实施的难度，潜在的挑战，以及预期的结果。根据这些因素，为每个选项分配成功的概率和信心水平。

{solutions}

答案："""

prompt = PromptTemplate(
    input_variables=["solutions"],
    template = template                      
)

chain2 = LLMChain(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    prompt=prompt,
    output_key="review"
)

template ="""
步骤3：

对于每个解决方案，深化思考过程。生成潜在的场景，实施策略，任何必要的合作伙伴或资源，以及如何克服潜在的障碍。此外，考虑任何潜在的意外结果以及如何处理它们。

{review}

答案："""

prompt = PromptTemplate(
    input_variables=["review"],
    template = template                      
)

chain3 = LLMChain(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    prompt=prompt,
    output_key="deepen_thought_process"
)

template ="""
步骤4：

根据评估和场景，按照承诺的顺序对解决方案进行排名。输出最优方案并为每个排名提供理由，并为每个解决方案提供任何最后的想法或考虑。

{deepen_thought_process}

答案：[assistant]"""

prompt = PromptTemplate(
    input_variables=["deepen_thought_process"],
    template = template                      
)

chain4 = LLMChain(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    prompt=prompt,
    output_key="ranked_solutions"
)

from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["input", "perfect_factors"],
    output_variables=["ranked_solutions"],
    verbose=True
)
#“输入”:“设计沙漠地形作战无人机集群”, “完美因素”:“沙漠风沙大飞行稳定性和高速飞行下飞机容易灰尘沙子损坏，飞机集群立体作战视野受限”。

result = overall_chain({"input":"设计沙漠地形作战无人机集群", "perfect_factors":"沙漠风沙大飞行稳定性和高速飞行下飞机容易灰尘沙子损坏，飞机集群立体作战视野受限"})
print(result)