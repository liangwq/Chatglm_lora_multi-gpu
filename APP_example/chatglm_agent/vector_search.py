from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool

#文档导入
loader = TextLoader("/root/autodl-tmp/ChatGLM2-6B/read.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#embbeing       
model_name = "nghuyong/ernie-3.0-xbase-zh"
#model_name = "nghuyong/ernie-3.0-nano-zh"
#model_name = "shibing624/text2vec-base-chinese"
#model_name = "GanymedeNil/text2vec-large-chinese"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
   model_name=model_name,
   model_kwargs=model_kwargs,
   encode_kwargs=encode_kwargs,
   cache_folder = "/root/autodl-tmp/ChatGLM2-6B/llm_model"
)
#把数据存入向量数据库，这部分代码可以分开单独做一个类
db = FAISS.from_documents(docs, hf)


'''测试向量相似度检索
query = "chatglm是什么？"
docs = db.similarity_search(query)
print(docs[0].page_content)


docs_and_scores = db.similarity_search_with_score(query)
docs_and_scores[0]

embedding_vector = hf.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
docs_and_scores[0]
'''

#问题改写，下面的检索可以基于改写后的问题来做
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)
len(retriever_from_llm.get_relevant_documents(query=query))

'''
以下是三个不同版本的用户问题，用于从向量数据库中检索相关文档：

1. 什么是chatglm？
2. chatglm是什么？它有什么用处？
3. chatglm是一个基于什么技术构建的？
'''

#下面是3种检索方式
from langchain.chains import RetrievalQA
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

ruff.run(query)

from langchain.chains import RetrievalQA
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="refine", retriever=db.as_retriever()
)
ruff.run(query)

from langchain.chains import RetrievalQA
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="map_reduce", retriever=db.as_retriever()
)
ruff.run(query)


#把基于向量库的检索生成封装成Tool
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)


tools = [
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]

'''挂到Agent上测试效果
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)
'''