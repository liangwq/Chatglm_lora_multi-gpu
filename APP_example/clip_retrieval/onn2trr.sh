export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
# 如前文，${DATAPATH}请根据实际情况替换
python cn_clip/deploy/onnx_to_tensorrt.py \
       --model-arch ViT-H-14 \
       --convert-text \
       --text-onnx-path ./deploy/vit-h-14.txt.fp16.onnx \
       --convert-vision \
       --vision-onnx-path ./deploy/vit-h-14.img.fp16.onnx \
       --save-tensorrt-path ./deploy/vit-h-14 \
       --fp16