from langchain.agents import AgentExecutor
from custom_search import DeepSearch
from tool_set import *
from intent_agent imprt IntentAgent
from llm_model import  ChatGLM

#model_path换成自己的模型地址
llm = ChatGLM(model_path="/root/autodl-tmp/ChatGLM2-6B/llm_model/models--THUDM--chatglm2-6b/snapshots/8eb45c842594b8473f291d0f94e7bbe86ffc67d8")
llm.load_model()
agent = IntentAgent(tools=tools, llm=llm)

tools = [Search_www_Tool(llm = llm), Actor_knowledge_Tool(llm=llm),Search_www_Tool(llm = llm),Product_knowledge_Tool(llm=llm)]

agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("生成10条描述金融产品卖点的营销文案")
agent_exec.run("四川高考分数线？")