# 导入所需要的库
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# 第一个LLMChain：生成材料的介绍
llm = OpenAI(temperature=.7)
template = """
你是一个材料科学家。给定材料的名称和类型，你需要为这种材料写一个200字左右的介绍。
材料: {name}
工艺: {color}
材料科学家: 这是关于上述材料的介绍:"""
prompt_template = PromptTemplate(
    input_variables=["name", "color"],
    template=template
)
introduction_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="introduction"
)

# 第二个LLMChain：根据材料的介绍写出材料的应用场景
template = """
你是一位材料应用专家。给定一种材料的介绍，你需要为这种材料写一篇200字左右的材料特性和应用场景介绍。
材料介绍:
{introduction}
材料应用专家对上述材料的应用介绍:"""
prompt_template = PromptTemplate(
    input_variables=["introduction"],
    template=template
)
review_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="review"
)

# 第三个LLMChain：根据材料的介绍和材料应用场景介绍写出一篇材料科普的文案
template = """
你是一家材料科普的媒体经理。给定一种材料的介绍和应用场景介绍，你需要为这种花写一篇材料科普的帖子，300字左右。
材料介绍:
{introduction}
材料应用专家对上述材料的特性和应用场景介绍:
{review}
材料科普帖子:
"""
prompt_template = PromptTemplate(
    input_variables=["introduction", "review"],
    template=template
)
social_post_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="social_post_text"
)

# 总的链：按顺序运行三个链
overall_chain = SequentialChain(
    chains=[introduction_chain, review_chain, social_post_chain],
    input_variables=["name", "color"],
    output_variables=["introduction", "review", "social_post_text"],
    verbose=True
)

# 运行链并打印结果
result = overall_chain({
    "name": "特种陶瓷",
    "color": "特种定向凝固"
})
print(result)

'''
{'name': '特种陶瓷', 'color': '特种定向凝固', 'introduction': '\n\n特种陶瓷是一种经过特殊工艺加工的陶瓷材料。它是通过特种定向凝固技术将碳素陶瓷材料转化成特种陶瓷材料的。这种材料具有较高的热稳定性和耐磨性，可在极端条件下仍保持其结构不变。特种陶瓷也具有较高的耐腐蚀性，可以抵抗腐蚀环境的侵蚀', 'review': '\n\n特种陶瓷是一种高性能材料，具有高热稳定性、耐磨性、耐腐蚀性等特性，广泛应用于高温、高压、腐蚀性环境等极端条件下的环境。它可以用来制造各种抗摩擦摩擦件，用于汽车、航空和航天等行业，这些件具有高热稳定性、高耐磨性、高耐腐蚀性等优异', 'social_post_text': '\n特种陶瓷是一种高性能材料，它能够经受极端的环境条件，具有较高的热稳定性、耐磨性和耐腐蚀性，是制造各种抗摩擦摩擦件的理想材料。特种陶瓷在汽车、航空和航天等行业中应用广泛，因为它具有高热稳定性、高耐磨性和高耐腐蚀性等优异特性。\n\n特'}
'''