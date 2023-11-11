# 创建模型实例
from langchain import OpenAI
model = OpenAI(model_name='text-davinci-003')

# ------Part 2
# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# 数据准备
flowers = ["合金金属", "特种高分子", "水性溶胶"]
prices = ["500", "3000", "20000"]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="材料的种类")
    price: int = Field(description="材料的价格")
    description: str = Field(description="材料功能的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

# ------Part 3
# 创建输出解析器
from langchain.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：",format_instructions)


# ------Part 4
# 创建提示模板
from langchain import PromptTemplate
prompt_template = """您是一位专业的材料技术专家。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
       partial_variables={"format_instructions": format_instructions}) 

# 打印提示
print("提示：", prompt)

# ------Part 5
for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower, price=price)
    # 打印提示
    print("提示：", input)

    # 获取模型的输出
    output = model(input)

    # 解析模型的输出
    parsed_output = output_parser.parse(output)
    parsed_output_dict = parsed_output.dict()  # 将Pydantic格式转换为字典

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output.dict()

# 打印字典
print("输出的数据：", df.to_dict(orient='records'))

'''
输出格式： The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"flower_type": {"description": "\u6750\u6599\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u6750\u6599\u7684\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u6750\u6599\u529f\u80fd\u7684\u63cf\u8ff0\u6587\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}

提示： input_variables=['flower', 'price'] partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"flower_type": {"description": "\\u6750\\u6599\\u7684\\u79cd\\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\\u6750\\u6599\\u7684\\u4ef7\\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\\u6750\\u6599\\u529f\\u80fd\\u7684\\u63cf\\u8ff0\\u6587\\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\\u4e3a\\u4ec0\\u4e48\\u8981\\u8fd9\\u6837\\u5199\\u8fd9\\u4e2a\\u6587\\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}\n```'} template='您是一位专业的材料技术专家。\n对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？\n{format_instructions}'
提示： 您是一位专业的材料技术专家。
对于售价为 500 元的 合金金属 ，您能提供一个吸引人的简短中文描述吗？
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"flower_type": {"description": "\u6750\u6599\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u6750\u6599\u7684\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u6750\u6599\u529f\u80fd\u7684\u63cf\u8ff0\u6587\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}

提示： 您是一位专业的材料技术专家。
对于售价为 3000 元的 特种高分子 ，您能提供一个吸引人的简短中文描述吗？
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"flower_type": {"description": "\u6750\u6599\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u6750\u6599\u7684\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u6750\u6599\u529f\u80fd\u7684\u63cf\u8ff0\u6587\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}

提示： 您是一位专业的材料技术专家。
对于售价为 20000 元的 水性溶胶 ，您能提供一个吸引人的简短中文描述吗？
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"flower_type": {"description": "\u6750\u6599\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u6750\u6599\u7684\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u6750\u6599\u529f\u80fd\u7684\u63cf\u8ff0\u6587\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}

输出的数据： [{'flower_type': 'Alloy Metal', 'price': 500, 'description': '这款合金金属具有优质的耐腐蚀性、抗高温性、强度高，是安全可靠的选择。', 'reason': '为了介绍优质的合金金属，以及该产品的价格。'}, {'flower_type': 'Special High Polymer', 'price': 3000, 'description': '这款特种高分子具有优质的导电性，耐热性和耐腐蚀性，具有超长寿命，是极佳的替代材料', 'reason': '质量优良，性能出色，价格实惠'}, {'flower_type': '水性溶胶', 'price': 20000, 'description': '这种水性溶胶具有良好的耐腐蚀、耐磨损性能，可以满足高精度的工艺要求，具有良好的耐切削性能，可以满足切削加工的要求', 'reason': '这款水性溶胶以其优良的性能，具有很高的性价比，价格实惠，值得购买。'}]
'''
