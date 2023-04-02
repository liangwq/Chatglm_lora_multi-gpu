import os
import re
import time
import json
import datetime
import pandas as pd

import torch
import streamlit as st

from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration

#torch.set_default_tensor_type(torch.cuda.HalfTensor)


st.set_page_config(
    page_title="LLM Playground",
    layout="wide"
)


device = 'cpu'
max_new_tokens = 300
peft_model_path = "./"

LOG_PATH = 'log'
DATASET_PATH = 'data'

LOG_FILE = 'web_log.log'
FEEDBACK_FILE = 'human_feedback.log'
DATASET_FILE = 'dataset.jsonl'


if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

if not os.path.exists(os.path.join(DATASET_PATH, DATASET_FILE)):
    with open(os.path.join(DATASET_PATH, DATASET_FILE), 'w', encoding='utf8') as f:
        print('标注数据集已创建。')


if 'model_out' not in st.session_state:
    st.session_state['model_out'] = ''
    st.session_state['used_time'] = 0.0


if 'model' not in st.session_state:
    with st.spinner('Loading Model...'):
        #config = PeftConfig.from_pretrained(peft_model_path)
        peft_path = "chatglm-lora-new.pt"

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True,
            r=8,
            lora_alpha=32, 
            lora_dropout=0.1)

        
        model = ChatGLMForConditionalGeneration.from_pretrained(
            "THUDM/chatglm-6b", 
            cache_dir ='./',
            trust_remote_code=True
        ).float().to(device)
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(peft_path), strict=False)
        '''model = PeftModel.from_pretrained(
            model, 
            peft_model_path
        )'''
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", 
            cache_dir ='./',
            trust_remote_code=True
        )
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer


def start_evaluate_page():
    """
    模型测试页面。
    """
    model = st.session_state['model']
    c1, c2 = st.columns([5, 5])
    with c1:
        with st.expander('⚙️ Instruct 设定（Instruct Setting）', expanded=True):
                instruct = st.text_area(
                    f'Instruct',
                    value='你现在是一个很厉害的写作器，严格按照人类指令进行回答。',
                    height=250
                )
    
    with c2:
        with st.expander('💬 对话输入框', expanded=True):
            current_input = st.text_area(
                '当前用户输入',
                value='帮我用以下关键词生成诗歌，并输出为json，不要做多余的回复：\n\n星空、满月、时光、流浪',
                height=200
            )
            bt = st.button('Generate')
        if bt:
            start = time.time()
            with st.spinner('生成中...'):
                with torch.no_grad():
                    input_text = f"Instruction: {instruct}\n"
                    input_text += f"Input: {current_input}\n"
                    input_text += f"Answer:"
                    batch = st.session_state['tokenizer'](input_text, return_tensors="pt")
                    out = model.generate(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=torch.ones_like(batch["input_ids"]).bool().to(device),
                        max_new_tokens=max_new_tokens,
                        temperature=0
                    )
                    out_text = st.session_state['tokenizer'].decode(out[0])
                    answer = out_text.split('Answer: ')[-1]
                    used_time = round(time.time() - start, 2)

                st.session_state['model_out'] = answer
                st.session_state['used_time'] = used_time

                with open(os.path.join(LOG_PATH, LOG_FILE), 'a', encoding='utf8') as f:
                    log_dict = {
                        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'used_seconds': used_time,
                        'instruct': instruct,
                        'input': current_input,
                        'model_output': st.session_state['model_out']
                    }
                    f.write(f'{json.dumps(log_dict, ensure_ascii=False)}\n')

    if st.session_state['model_out']:

        c1, c2 = st.columns([5, 5])

        with c2:
            with st.expander(f"🤖 当前模型输出（{st.session_state['used_time']}s）", expanded=True):
                answer = st.session_state['model_out']

                if len(answer) < 200:
                    height = 100
                elif len(answer) < 500:
                    height = 200
                else:
                    height = 300

                st.text_area(
                    '', 
                    value=f'{answer}',
                    height=height
                )

                human_feedback = st.radio(
                    "模型生成结果是否正确 👇",
                    ["🤓 不反馈", "😊 正确", "😡 错误"],
                    key="visibility",
                    horizontal=True
                )

                if human_feedback in ["😡 错误", "😊 正确"]:
                    if human_feedback == "😡 错误":
                        error_feedback = st.text_area(
                            '[error feedback]',
                            placeholder='感谢您的反馈，请填写问题的正确答案以帮助模型改进 🥺'
                        )
                        current_feedback = {
                            'instruct': instruct,
                            'input': current_input,
                            'model_output': st.session_state['model_out'],
                            'human_feedback': error_feedback,
                            'is_error': True
                        }
                    else:
                        advice = st.text_area(
                            '[advice feedback]',
                            placeholder='请输入您的反馈...'
                        )
                        current_feedback = {
                            'instruct': instruct,
                            'input': current_input,
                            'model_output': st.session_state['model_out'],
                            'human_feedback': advice,
                            'is_error': False
                        }
                    submit_button = st.button('提交')
                    if submit_button:
                        with open(os.path.join(LOG_PATH, FEEDBACK_FILE), 'a', encoding='utf8') as f:
                            f.write(f'{json.dumps(current_feedback, ensure_ascii=False)}\n')
                        st.success('感谢您的反馈~', icon="✅")
    
        with c1:
            with st.expander(f'💻 json 解析结果', expanded=True):
                st.markdown(answer)
                json_res = re.findall(r'```json(.*)```', answer.replace('\n', ''))
                if len(json_res):
                    json_res = json_res[0]
                    try:
                        json_res = json.loads(json_res)
                        st.write(json_res)
                    except:
                        pass


def read_dataset_file():
    """
    读取本地标注的数据集。
    """
    temp_dict = {}
    with open(os.path.join(DATASET_PATH, DATASET_FILE), 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                if key not in temp_dict:
                    temp_dict[key] = []
                temp_dict[key].append(value)
    df = pd.DataFrame.from_dict(temp_dict)
    return df


def start_label_page():
    """
    数据集标注页面。
    """
    c1, c2 = st.columns([4, 6])

    with c1:
        with st.expander(f'💻 标注界面', expanded=True):
            instruct = st.text_area(
                    'Human Instruct',
                    value='你现在是一个很厉害的写作理解器，严格按照人类指令进行回答。'
                )
            inputs = st.text_area(
                    'Human Input',
                    placeholder='输入人工构造问题，例如: 帮我生成2句诗词，用json格式返回...'
                )
            answer = st.text_area(
                    'Human Output',
                    placeholder='输入人工构造答案，例如:\n 好的，以下是生成的诗词的json信息：\n```json\n{\n"subject1": "惊雷一声梨花开，寒潭生暖大地春", \n"subject2": "那些光亮的就让它光亮，哪些晦暗的就让它再明", \n}\n```',
                    height=500
                )
            save_button = st.button('Save')
            if save_button:
                with open(os.path.join(DATASET_PATH, DATASET_FILE), 'a', encoding='utf8') as f:
                    context = f'Instruction: {instruct}\nInput: {inputs}\nAnswer: '
                    current_sample = {
                        'context': context,
                        'target': answer
                    }
                    f.write(json.dumps(current_sample, ensure_ascii=False) + '\n')
                st.success('数据已保存！', icon="✅")
                st.session_state['dataset_df'] = read_dataset_file()
    
    with c2:
        if 'dataset_df' not in st.session_state:
            st.session_state['dataset_df'] = read_dataset_file()
        
        with st.expander(f"📚 本地数据集（共 {len(st.session_state['dataset_df'])} 条）", expanded=True):
            st.dataframe(st.session_state['dataset_df'], height=820)

            refresh_button = st.button('刷新数据集')
            if refresh_button:
                st.session_state['dataset_df'] = read_dataset_file()


def main():
    """
    主函数流程。
    """


    evaluate_page, label_page = st.tabs(['evaluate_page', 'label_page'])
    
    with evaluate_page:
        start_evaluate_page()

    with label_page:
        start_label_page()


if __name__ == '__main__':
    main()
