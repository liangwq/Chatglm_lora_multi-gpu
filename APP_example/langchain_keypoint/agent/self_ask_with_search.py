from langchain import OpenAI, SerpAPIWrapper 
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer", 
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "把定向凝固理论用到特种陶瓷生产的科学家在哪国的实验室就职?"  
)
'''
> Entering new AgentExecutor chain...
 Yes.
Follow up: 谁是把定向凝固理论用到特种陶瓷生产的科学家？
Intermediate answer: ['“ 定向凝固技术在航空涡轮叶片制备上的成功应用及以成分过冷理论为代表的定量凝固科学的出现，使定向凝固工艺的实验研究逐步进入精确定量阶段并与先进的航空航天材料相 ...', '1993 年乌克兰科学家Shapovalov在美. 国申请的一个专利[1]，则使我们对金属中的气孔需要重新认识。专利中提出了一种制备多孔金属的. 新方法⎯⎯金属/气体共晶定向凝固法， ...', '作为材料科学与工程的基本组成，凝固科学技术正在现代科学理论的基础上针对传统材料的改性提高和新材料的发展需求，以控形、控构、控性为目标开展优质铸件的定向、晶体生长 ...', '他开辟了单晶及定向组织超细化研究的新领域，建立了枝胞转换及亚快速定向凝固的理论框架，研制了超高梯度定向凝固装备，高温合金定向凝固界面温度梯度达 ...', '... 将现有的凝固理论用于处. 理新材料的制备成形。而新材料的凝固加工多数涉及定向凝固技术，所以先进材料的定向. 凝固受到广泛关注。 西北工业大学和哈尔滨工业大学长期 ...', '新材料的发展需求,以控形、控构、控性为目标开展优质铸件的定向、晶体生长、快凝、深过冷及各种新型和. 超常领域凝固过程的研究,并介绍了其中某些方面和展望了可能的 ...', '不同于传统烧结陶瓷，LSZM技术制备ZrB2-SiC陶瓷无需添加烧结助剂，通过熔体生长可获得界面结合良好、无非晶相的超细化凝固组织，有利于陶瓷性能的提升。', '特种陶瓷是具有高强、耐温、耐腐蚀特性或具有各种敏感特性的陶瓷材料，由于其制作工艺、化学组成、显微结构及特性不同于传统陶瓷，所以有特种陶瓷之称，又叫先进陶瓷、新型 ...', '特别是日本的陶瓷建筑材料和工业化学品生产商——神岛化学工业株式会社的科学家们正一直在为他们的透明陶瓷寻找市场。 美国利弗莫尔国家实验室的研究人员意识到，这些陶瓷 ...', '后,将合金熔体浇入水冷铸型中定向凝固。由于氢气在. 固相和液相中的溶解度差,在适当的 ... 尽管如此,由于研究时. 间到现在不过10年,无论是理论研究还是试验研究都. 还很少 ...']
Intermediate answer: 特种陶瓷生产的科学家是美国利弗莫尔国家实验室的研究人员。
Follow up: 美国利弗莫尔国家实验室在哪国？
Intermediate answer: LLNL位于美国加利福尼亚州利弗莫尔，是受到美国政府机构资助的研发中心。 去年12月13日，美国能源部召开新闻发布会宣布，LLNL的科学家在12月5日首次成功在核聚变实验中实现“净能量增益(Net Energy Gain)”，即受控核聚变反应产生的能量超过驱动反应发生的激光能量。
So the final answer is: 美国加利福尼亚州利弗莫尔

> Finished chain.
'美国加利福尼亚州利弗莫尔'
'''