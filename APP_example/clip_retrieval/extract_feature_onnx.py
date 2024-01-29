import os
import glob

def find_image_files(folder_path, extensions=['jpg', 'jpeg', 'png', 'gif']):
    image_files = []
    for extension in extensions:
        pattern = os.path.join(folder_path, f'*.{extension}')
        image_files.extend(glob.glob(pattern))
    return image_files


# 完成必要的import（下文省略）
import onnxruntime
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from cn_clip.clip  import load_from_name, available_models
from cn_clip.clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# 载入ONNX图像侧模型（**请替换${DATAPATH}为实际的路径**）
img_sess_options = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path="./deploy/vit-h-14.img.fp16.onnx"
img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 预处理图片
model_arch = "ViT-H-14" # 这里我们使用的是ViT-B-16规模，其他规模请对应修改
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])


def compute_embeddings(img_dir, save_path, batch_size, num_workers,model,preprocess):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 指定文件夹路径
    folder_path = '../image/train/left'

    # 查找图片文件
    image_files = find_image_files(folder_path)

    img_path_list, embedding_list = [], []
    for image in tqdm(image_files):
        with torch.no_grad():
            '''
            features = model.encode_image(img.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            '''
            # 用ONNX模型计算图像侧特征 
            image = preprocess(Image.open(image)).unsqueeze(0)

            # 用ONNX模型计算图像侧特征 .cpu().numpy()
            image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] # 未归一化的图像特征
            image_features = torch.tensor(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化后的Chinese-CLIP图像特征，用于下游任务
            print(image_features.shape) # Torch Tensor shape: [1, 特征向量维度]
            embedding_list.extend(image_features)#.detach().cpu().numpy())
            img_path_list.extend(image)
            

    result = {'img_path': img_path_list, 'embedding': embedding_list}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)

img_dir = "../image/train/left"
save_path = "results/embeddings.pkl"
batch_size = 256
num_workers = 40

compute_embeddings(img_dir , save_path, batch_size, num_workers,img_session,preprocess)
