#from openai import OpenAI
import streamlit as st
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


history_mssg = []#st.session_state.messages
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print(st.session_state.messages)
    
        
    messages =[]
    messages.append({"role": "user", "content": prompt})
    history_mssg.append({"role": "user", "content":str(st.session_state.messages)+ prompt})
    #print(history_mssg)
    response = openai.ChatCompletion.create(model="Qwen", messages=history_mssg,#st.session_state.messages,
    stream=False,
    stop=[])
    msg = response.choices[0].message.content
    assistant_mssg = {"role": "assistant", "content": msg}
    st.session_state.messages.append({"role": "assistant", "content": msg})
    history_mssg.append(assistant_mssg)
    st.chat_message("assistant").write(msg)

