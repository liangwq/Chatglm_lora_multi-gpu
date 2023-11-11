from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 定义一个动作类
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# 创建一个解析器
parser = PydanticOutputParser(pydantic_object=Action)

# 创建一个带有错误的响应 {"action": "search", "action_input": "some input"}
bad_response = ' {"action": "search", }'

#parser.parse(bad_response) # 这行代码会出错

# 使用RetryWithErrorOutputParser
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))
retry_parser.parse_with_prompt(bad_response, prompt_value)