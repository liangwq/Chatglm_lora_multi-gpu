部署文本生成图片模型pixart-sigma：

1.安装环境和下载源码
conda create -n pixart python==3.9.0
conda activate pixart
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/PixArt-alpha/PixArt-sigma.git
cd PixArt-sigma
pip install -r requirements.txt
2.下载模型
# SDXL-VAE, T5 checkpoints
git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers

# PixArt-Sigma checkpoints
python tools/download.py # environment eg. HF_ENDPOINT=https://hf-mirror.com can use for HuggingFace mirror
3.启动模型后台
python scripts/interface.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-2k-MS.pth --image_size 2048 --port 6006
