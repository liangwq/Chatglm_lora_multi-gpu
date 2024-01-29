import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel

#import clip
from search_dataset import ClipSearchDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import click

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models





#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

def extract_features_dino(image,processor_dino,model_dino,device):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)



@click.command()
@click.option('--img_dir', default='../image/train/left', help='Directory of images.')
@click.option('--save_path', default='results/embeddings_diniv2.pkl', help='Path to save the embeddings.')
@click.option('--batch_size', default=256, help='Batch size for DataLoader.')
@click.option('--num_workers', default=40, help='Number of workers for DataLoader.')
def compute_embeddings(img_dir, save_path, batch_size, num_workers):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


    #Load DINOv2 model and processor
    processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    #Retrieve all filenames
    images = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith('jpg'):
                images.append(root + '/'+ file)

    index_dino = faiss.IndexFlatL2(768)

    #Iterate over the dataset to extract features X2 and store features in indexes
    for image_path in images:
        img = Image.open(image_path).convert('RGB')
        dino_features = extract_features_dino(img,processor_dino,model_dino,device)
        add_vector_to_index(dino_features,index_dino)


    faiss.write_index(index_dino,save_path)


if __name__ == "__main__":
    compute_embeddings()