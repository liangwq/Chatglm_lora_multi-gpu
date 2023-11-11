from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 使用Pydantic创建一个数据格式，表示材料
class Flower(BaseModel):
    name: str = Field(description="name of a Material")
    colors: List[str] = Field(description="the colors of this Material")
# 定义一个用于获取某种材料的颜色列表的查询
flower_query = "Generate the charaters for a random Material."

# 定义一个格式不正确的输出
misformatted = "{'name': '汽车漆', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

# 创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
parser = PydanticOutputParser(pydantic_object=Flower)
# 使用Pydantic解析器解析不正确的输出
#parser.parse(misformatted) # 这行代码会出错
#OutputParserException: Failed to parse Flower from completion {'name': '汽车漆', 'colors': ['粉红色','白色','红色','紫色','黄色']}. Got: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

# 从langchain库导入所需的模块
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser


# 使用OutputFixingParser创建一个新的解析器，该解析器能够纠正格式不正确的输出
new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
#print(new_parser)
# 使用新的解析器解析不正确的输出
result = new_parser.parse(misformatted) # 错误被自动修正
print(result) # 打印解析后的输出结果
#name='汽车漆' colors=['粉红色', '白色', '红色', '紫色', '黄色']