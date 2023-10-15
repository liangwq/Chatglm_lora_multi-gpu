# -*- coding: utf-8 -*-
# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

loader = PyPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")

## Other options for loaders 
#loader = UnstructuredPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")
data = loader.load()
# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=500)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

'''
把pdf转txt后切分成块后做翻译，保存文件
'''
import openai

import os
import logging


# 指定文件夹路径
folder_path = " article_en"

# 检查文件夹是否存在，如果不存在则创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
# 指定文件夹路径
folder_path_ch = " article_keyword"

# 检查文件夹是否存在，如果不存在则创建
if not os.path.exists(folder_path_ch):
    os.makedirs(folder_path_ch)
    
# 设置日志记录
logging.basicConfig(filename='retry.log', level=logging.ERROR)

def translate_article(folder_path,folder_path_ch,content):
    # 在新建的文件夹中创建文件并写入英文数据
    file_path = os.path.join(folder_path, "example_"+str(i)+".txt")
    # 判断文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，删除文件
        os.remove(file_path)
        print(f"文件 {file_path} 存在并已删除。")
    
    # 打开文件并写入英文数据
    with open(file_path, "w") as file:
        file.write(content.page_content)

    # 关闭文件
    file.close()
    
    # 在新建的文件夹中创建文件并写入中文数据
    file_path_ch = os.path.join(folder_path_ch, "example_"+str(i)+".txt")
    
    # 判断文件是否存在
    if os.path.exists(file_path_ch):
        # 如果文件存在，删除文件
        os.remove(file_path_ch)
        print(f"文件 {file_path_ch} 存在并已删除。")
    
    
    #请求openapi把英文翻译成中文
    try:
        import openai

        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "none"
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "你是一个专业学术论点抽取机器人，可以精准抽取论文中关键信息、精准抓取出论文片段中关键词、关键观点、生成精准摘要，并把英文翻译成中文输出，严格执行人类指令;\n"},
            {"role": "user", "content": content.page_content+'\n对上面论文片段抽取关信息、关键词、关键观点、生成摘要；并以{"关键信息":,"关键词":,"关键观点":,"生成摘要":}json格式输出,英文翻译成中文'},
            ],
            stream=False,
            stop=[] # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
        )
        # 打开文件并写入中文数据
        with open(file_path_ch, "w") as file:
            file.write(response["choices"][0]["message"]["content"])
    except Exception as e:
        pass


    # 关闭文件
    file.close()
    
for i in range(len(texts)):
    # 最大重试次数
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            #print(data[i])
            translate_article(folder_path,folder_path_ch,texts[i])

            # 如果操作成功，退出循环
            break
        except Exception as e:
            # 操作失败，记录异常到日志
            logging.error(f"操作失败: {e}")
            # 增加重试次数
            retry_count += 1
            if retry_count < max_retries:
                print(f"操作失败，重试中 ({retry_count}/{max_retries})...")
                print(data[i])
                # 等待一段时间后重试
                #time.sleep(1)
            else:
                # 达到最大重试次数，抛出异常
                raise
# 提示操作完成
print("文件夹和文件创建完成。")

