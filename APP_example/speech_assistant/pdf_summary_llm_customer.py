# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

loader = PyPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")

## Other options for loaders 
#loader = UnstructuredPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")
data = loader.load()
# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=250)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",openai_api_key='',openai_api_base='')

summary_output_options = {
    'one_sentence' : """
     - Only one sentence
    """,
    
    'bullet_points': """
     - Bullet point format
     - Separate each bullet point with a new line
     - Each bullet point should be concise
    """,
    
    'short' : """
     - A few short sentences
     - Do not go longer than 4-5 sentences
    """,
    
    'long' : """
     - A verbose summary
     - You may do a few paragraphs to describe the transcript if needed
    """
}

template="""

You are a helpful assistant, assisting {rep_name}, a professional researcher, in extracting important information from this {rep_company} academic paper. 
Your goal is to write a summary from an academic perspective, highlighting key points relevant to this academic paper.
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

human_template="{text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
    
template="""

You are a helpful assistant, assisting {rep_name}, a professional researcher, in extracting important information from this {rep_company} academic paper. 
Your goal is to write a summary from an academic perspective, highlighting key points relevant to this academic paper.
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"

Respond with the following format
{output_format}

"""
system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template)

human_template="{text}" # Simply just pass the text as a human message
human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_combine, human_message_prompt_combine])


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",openai_api_key='none',openai_api_base='http://localhost:8000/v1')

# verbose=True will output the prompts being sent to the 
#chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
#output = chain.run(texts)

chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True
                            )

user_selection = 'one_sentence'

output = chain.run({
                    "input_documents": texts,
                    "rep_company": "Quantum computing latest", \
                    "rep_name" : "Quantum computing discipline",
                    "output_format" : summary_output_options[user_selection]
                   })


print(output)
