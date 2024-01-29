import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
from PIL import Image
import os

#Input image
source='examples/pokemon.jpeg'
image = Image.open(source)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

 #Extract features for DINOv2
with torch.no_grad():
    inputs_dino = processor_dino(images=image, return_tensors="pt").to(device)
    outputs_dino = model_dino(**inputs_dino)
    image_features_dino = outputs_dino.last_hidden_state
    image_features_dino = image_features_dino.mean(dim=1)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

image_features_dino = normalizeL2(image_features_dino)


#Search the top 5 images
index_dino = faiss.read_index("dino.index")

#Get distance and indexes of images associated
d_dino,i_dino = index_dino.search(image_features_dino,5)