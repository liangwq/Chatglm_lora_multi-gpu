import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

#import clip
from search_dataset import ClipSearchDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import click

@click.command()
@click.option('--img_dir', default='../image/train', help='Directory of images.')
@click.option('--save_path', default='results/embeddings.pkl', help='Path to save the embeddings.')
@click.option('--batch_size', default=256, help='Batch size for DataLoader.')
@click.option('--num_workers', default=40, help='Number of workers for DataLoader.')
def compute_embeddings(img_dir, save_path, batch_size, num_workers):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='./')
    model.eval()
    dataset = ClipSearchDataset(img_dir = img_dir, preprocess = preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    img_path_list, embedding_list = [], []
    for img, img_path in tqdm(dataloader):
        with torch.no_grad():
            features = model.encode_image(img.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            embedding_list.extend(features.detach().cpu().numpy())
            img_path_list.extend(img_path)

    result = {'img_path': img_path_list, 'embedding': embedding_list}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)

if __name__ == "__main__":
    compute_embeddings()
