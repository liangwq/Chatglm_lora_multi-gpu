import gradio as gr
import os
import random

import os
from PIL import Image
import pickle
import faiss
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

import torch

 
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
index, embedding_path_list, clip_model, preprocess = load_data(faiss_index_path, embeddings_path, device)


def text_search_image(query_text, num_search):
    with torch.no_grad():
        text = clip.tokenize([query_text]).to(device)
        features = clip_model.encode_text(text)
    features /= features.norm(dim=-1, keepdim=True)
    embedding_query = features.detach().cpu().numpy().astype(np.float32)
    D,I = index.search(embedding_query, num_search)
    match_path_list = [embedding_path_list[i] for i in I[0]]
    images = []
    for idx in range(num_search):
        path = match_path_list[idx]
        images.append(path)
    return images

def image_search_image(query_image, num_search):
    img_tensor = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor.to(device))
    features /= features.norm(dim=-1, keepdim=True)
    embedding_query = features.detach().cpu().numpy().astype(np.float32)
    D,I = index.search(embedding_query, num_search)
    match_path_list = [embedding_path_list[i] for i in I[0]]
    images = []
    for idx in range(num_search):
        path = match_path_list[idx]
        images.append(path)
    return images


with gr.Blocks().queue() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                , columns=[1], rows=[5], object_fit="contain", height="auto")
            slider = gr.Slider(0, 10, step=1)
        with gr.Column(scale=4):
            input_image = gr.Image( type="pil")
            text_prompt = gr.Textbox(label="Prompts")
            
            with gr.Row():
                text_button = gr.Button(value="Text Search")
                image_button = gr.Button(value="Image Search")
             
    image_button.click(image_search_image, inputs = [input_image,slider], outputs =[gallery])      
    text_button.click(text_search_image, inputs = [text_prompt,slider], outputs =[gallery])
    


if __name__ == "__main__":
    demo.launch(server_port=6006,share=True)


