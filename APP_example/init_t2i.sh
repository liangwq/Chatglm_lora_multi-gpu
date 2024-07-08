#创建指定python版本虚拟环境
conda create -n lumina python=3.11
conda activate lumina

#下载lumina源码
git clone https://github.com/Alpha-VLLM/Lumina-T2X.git
conda activate lumina
#安装需要工具包
pip install -q flash_attn==2.5.9.post1 --no-build-isolation
pip install git+https://github.com/Alpha-VLLM/Lumina-T2X

#下载模型参数
apt -y install -qq aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/ckpt/Lumina-Next-SFT/resolve/main/consolidated_ema.00-of-01.safetensors -d ./Lumina-T2X/models -o consolidated_ema.00-of-01.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/ckpt/Lumina-Next-SFT/resolve/main/model_args.pth -d ./Lumina-T2X/models -o model_args.pth

#启动lumina做图软件做图
cd ./Lumina-T2X/lumina_next_t2i
python demo.py --ckpt ../models --ema
