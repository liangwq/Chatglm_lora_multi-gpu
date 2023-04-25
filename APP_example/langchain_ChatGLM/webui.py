import gradio as gr
import os
import shutil
from .chains.local_doc_qa import LocalDocQA
from .configs.model_config import *
import nltk

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3


def get_file_list():
    if not os.path.exists("content"):
        return []
    return [f for f in os.listdir("content")]


file_list = get_file_list()

embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()


def upload_file(file):
    if not os.path.exists("content"):
        os.mkdir("content")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "content/" + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def get_answer(query, vs_path, history):
    if vs_path:
        resp, history = local_doc_qa.get_knowledge_based_answer(
            query=query, vs_path=vs_path, chat_history=history)
    else:
        history = history + [[None, "请先加载文件后，再进行提问。"]]
    return history, ""


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


def init_model():
    try:
        local_doc_qa.init_cfg()
        return """模型已成功加载，请选择文件后点击"加载文件"按钮"""
    except Exception as e:
        print(e)
        return """模型未成功加载，请重新选择后点击"加载模型"按钮"""


def reinit_model(llm_model, embedding_model, llm_history_len, top_k, history):
    try:
        local_doc_qa.init_cfg(llm_model=llm_model,
                              embedding_model=embedding_model,
                              llm_history_len=llm_history_len,
                              top_k=top_k)
        model_status = """模型已成功重新加载，请选择文件后点击"加载文件"按钮"""
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]



def get_vector_store(filepath, history):
    if local_doc_qa.llm and local_doc_qa.embeddings:
        vs_path = local_doc_qa.init_knowledge_vector_store(["content/" + filepath])
        if vs_path:
            file_status = "文件已成功加载，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    return vs_path, history + [[None, file_status]]


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}

.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉langchain-ChatGLM WebUI🎉

"""

init_message = """欢迎使用 langchain-ChatGLM Web UI，开始提问前，请依次如下 3 个步骤：
1. 选择语言模型、Embedding 模型及相关参数后点击"重新加载模型"，并等待加载完成提示
2. 上传或选择已有文件作为本地知识文档输入后点击"重新加载文档"，并等待加载完成提示
3. 输入要提交的问题后，点击回车提交 """

def clear_session():
    return '', None

def create_demo():
    model_status = init_model()
    with gr.Blocks(css=block_css) as demo:
        vs_path, file_status, model_status = gr.State(""), gr.State(""), gr.State(model_status)
        gr.Markdown(webui_title)
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",
                                   ).style(container=False)

            with gr.Column(scale=1):
                llm_model = gr.Radio(llm_model_dict_list,
                                     label="LLM 模型",
                                     value=LLM_MODEL,
                                     interactive=True)
                llm_history_len = gr.Slider(0,
                                            10,
                                            value=LLM_HISTORY_LEN,
                                            step=1,
                                            label="LLM history len",
                                            interactive=True)
                embedding_model = gr.Radio(embedding_model_dict_list,
                                           label="Embedding 模型",
                                           value=EMBEDDING_MODEL,
                                           interactive=True)
                top_k = gr.Slider(1,
                                  20,
                                  value=VECTOR_SEARCH_TOP_K,
                                  step=1,
                                  label="向量匹配 top k",
                                  interactive=True)
                load_model_button = gr.Button("重新加载模型")

                # with gr.Column():
                with gr.Tab("select"):
                    selectFile = gr.Dropdown(file_list,
                                             label="content file",
                                             interactive=True,
                                             value=file_list[0] if len(file_list) > 0 else None)
                with gr.Tab("upload"):
                    file = gr.File(label="content file",
                                   file_types=['.txt', '.md', '.docx', '.pdf']
                                   )  # .style(height=100)
                load_file_button = gr.Button("加载文件")
                
                with gr.Column(scale=2):    
                    with gr.Row():
                        clear_history = gr.Button("🧹 清除历史对话")
                        send = gr.Button("🚀 发送")
                        
                # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)
        load_model_button.click(reinit_model,
                                show_progress=True,
                                inputs=[llm_model, embedding_model, llm_history_len, top_k, chatbot],
                                outputs=chatbot
                                )

        # 将上传的文件保存到content文件夹下,并更新下拉框
        file.upload(upload_file,
                    inputs=file,
                    outputs=selectFile)
        load_file_button.click(get_vector_store,
                               show_progress=True,
                               inputs=[selectFile, chatbot],
                               outputs=[vs_path, chatbot],
                               )

        query.submit(get_answer,
                     [query, vs_path, chatbot],
                     [chatbot, query],
                     )
    return demo

#demo.queue(concurrency_count=3).launch(
#   server_name='0.0.0.0', share=False, inbrowser=False,server_port=6006)
