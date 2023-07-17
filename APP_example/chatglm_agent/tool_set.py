from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_func(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_func(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")
        


        
class Product_knowledge_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "生成文案,产品卖点查询"
    description = "用户输入的是生成内容问题，并且没有描述产品具体卖点无法给出答案，可以用互联网检索查询来获取相关信息"
    
    # QA params
    qa_template = """
    请根据下面信息```{text}```，回答问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None
    '''
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    '''
    
    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            
            
class Actor_knowledge_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "生成文案,人群画像查询"
    description = "用户输入的是内容生成问题，并且没有描述人群画像和人群的喜好无法给出答案，可以用互联网检索查询来获取相关信息"
    
    # QA params
    qa_template = """
    请根据下面信息```{text}```，回答问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None
    '''
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    '''
    
    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            
class Search_www_Tool(functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "互联网检索查询"
    description = "用户输入的问题是一些常识，实事问题，无法直接给出答案，可以用互联网检索查询来获取相关信息"
    
    # QA params
    qa_template = """
    请根据下面信息```{text}```，回答问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None
    '''
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    '''
    
    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = DeepSearch.search(query = query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp
    

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)