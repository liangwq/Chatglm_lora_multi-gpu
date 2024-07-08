#下载项目源码
git clone --recursive https://github.com/TMElyralab/MuseV.git
cd MuseV

#初始化环境
pip install -r requirements.txt
pip install --no-cache-dir -U openmim 
pip install mmengine 
pip install mmcv>=2.0.1
pip install mmdet>=3.1.0
pip install mmpose>=1.1.0

cd ..
current_dir=$(pwd)
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/MMCM
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/diffusers/src
export PYTHONPATH=${PYTHONPATH}:${current_dir}/MuseV/controlnet_aux/src
cd MuseV

#下载模型
git clone https://huggingface.co/TMElyralab/MuseV ./checkpoints

#输入文本、图像的视频生成
python scripts/inference/text2video.py   --sd_model_name majicmixRealv6Fp16   --unet_model_name musev   -test_data_path ./configs/tasks/example.yaml  --output_dir ./output  --n_batch 1  --target_datas yongen  --time_size 12 --fps 12 

#输入视频的视频生成
python scripts/inference/video2video.py --sd_model_name majicmixRealv6Fp16  --unet_model_name musev    -test_data_path ./configs/tasks/example.yaml --output_dir ./output  --n_batch 1 --controlnet_name dwpose_body_hand  --which2video "video_middle"  --target_datas  dance1   --fps 12 --time_size 12

#Gradio 演示
cd scripts/gradio
python app.py
