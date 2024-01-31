import streamlit as st
import os
from PIL import Image
import pickle
import faiss
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

import torch
st.set_page_config(layout="wide")
print("Available models:", available_models())  
@st.cache_resource
def load_data(faiss_index_path, embeddings_path, device=0):
    # load faiss index
    index = faiss.read_index(faiss_index_path)
    # load embeddings
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)
    embedding_path_list = results['img_path']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
    model.eval()
    #model, preprocess = clip.load('ViT-B/32', device)
    return index, embedding_path_list, model, preprocess

def display(features,num_search):
    features /= features.norm(dim=-1, keepdim=True)
    embedding_query = features.detach().cpu().numpy().astype(np.float32)
    D,I = index.search(embedding_query, num_search)
    match_path_list = [embedding_path_list[i] for i in I[0]]

    # calculate number of rows
    num_rows = -(-num_search // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

    # display
    for i in range(num_rows):
        cols = st.columns(images_per_row)
        for j in range(images_per_row):
            idx = i*images_per_row + j
            if idx < num_search:
                path = match_path_list[idx]
                distance = D[0][idx]
                img = Image.open(path).convert('RGB')
                cols[j].image(img, caption=f'Distance: {distance:.2f} path {path}', use_column_width=True)

# preprocess
device = 0
faiss_index_path = 'results/index.faiss'
embeddings_path = 'results/embeddings.pkl'
index, embedding_path_list, model, preprocess = load_data(faiss_index_path, embeddings_path, device)

# select box
search_mode = st.sidebar.selectbox('Search mode', ('Text', 'Upload Image', 'Image'))

# sliders
if search_mode == 'Image':
    img_idx = st.slider('Image index', 0, len(embedding_path_list)-1, 0)
    img_path = embedding_path_list[img_idx]
num_search = st.sidebar.slider('Number of search results', 1, 10, 5)
images_per_row = st.sidebar.slider('Images per row', 1, num_search, min(5, num_search))

if search_mode == 'Image':
    # search by image
    img = Image.open(img_path).convert('RGB')
    st.image(img, caption=f'Query Image: {img_path}')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor.to(device))
        #display(features,num_search)
elif search_mode == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    #img = Image.open("../image/train/left/00000.jpg").convert('RGB')
    img = Image.open("examples/pokemon.jpeg").convert('RGB')
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor.to(device))
            #display(features,num_search)
else:
    # search by text
    query_text = st.text_input('Enter a search term:')
    with torch.no_grad():
        text = clip.tokenize([query_text]).to(device)
        features = model.encode_text(text)
        #display(features,num_search)

    
features /= features.norm(dim=-1, keepdim=True)
embedding_query = features.detach().cpu().numpy().astype(np.float32)
D,I = index.search(embedding_query, num_search)
match_path_list = [embedding_path_list[i] for i in I[0]]

# calculate number of rows
num_rows = -(-num_search // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

# display
for i in range(num_rows):
    cols = st.columns(images_per_row)
    for j in range(images_per_row):
        idx = i*images_per_row + j
        if idx < num_search:
            path = match_path_list[idx]
            distance = D[0][idx]
            img = Image.open(path).convert('RGB')
            cols[j].image(img, caption=f'Distance: {distance:.2f} path {path}', use_column_width=True)
