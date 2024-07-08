#下载项目文件
git clone https://github.com/TMElyralab/MuseTalk.git
cd ./MuseTalk

#初始化环境
pip install -q diffusers==0.27.2 accelerate==0.28.0 omegaconf ffmpeg-python mmpose mmdet gradio
pip install -q https://github.com/camenduru/wheels/releases/download/colab2/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl

#下载模型
apt -y install -qq aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco.pth -d ./MuseTalk/models/dwpose -o dw-ll_ucoco.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.onnx -d ./MuseTalk/models/dwpose -o dw-ll_ucoco_384.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.pth -d ./MuseTalk/models/dwpose -o dw-ll_ucoco_384.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-mm_ucoco.pth -d ./MuseTalk/models/dwpose -o dw-mm_ucoco.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ss_ucoco.pth -d ./MuseTalk/models/dwpose -o dw-ss_ucoco.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-tt_ucoco.pth -d ./MuseTalk/models/dwpose -o dw-tt_ucoco.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-l_ucoco_256-95bb32f5_20230822.pth -d ./MuseTalk/models/dwpose -o rtm-l_ucoco_256-95bb32f5_20230822.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-x_ucoco_256-05f5bcb7_20230822.pth -d ./MuseTalk/models/dwpose -o rtm-x_ucoco_256-05f5bcb7_20230822.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-x_ucoco_384-f5b50679_20230822.pth -d ./MuseTalk/models/dwpose -o rtm-x_ucoco_384-f5b50679_20230822.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/yolox_l.onnx -d ./MuseTalk/models/dwpose -o yolox_l.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth -d ./MuseTalk/models/face-parse-bisent -o 79999_iter.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/resnet18-5c106cde.pth -d ./MuseTalk/models/face-parse-bisent -o resnet18-5c106cde.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/raw/main/musetalk/musetalk.json -d ./MuseTalk/models/musetalk -o musetalk.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/musetalk/pytorch_model.bin -d ./MuseTalk/models/musetalk -o pytorch_model.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/raw/main/sd-vae-ft-mse/config.json -d ./MuseTalk/models/sd-vae-ft-mse -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.bin -d ./MuseTalk/models/sd-vae-ft-mse -o diffusion_pytorch_model.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.safetensors -d ./MuseTalk/models/sd-vae-ft-mse -o diffusion_pytorch_model.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/whisper/tiny.pt -d ./MuseTalk/models/whisper -o tiny.pt

#启动可视化界面
python app.py
