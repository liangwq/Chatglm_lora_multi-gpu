export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=valid # 指定计算valid或test集特征
resume=./clip_cn_vit-h-14.pt

python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --image-data="/root/auto-tmp/image/train.lmdb" \
    --img-batch-size=32 \
    --resume=${resume} \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-base-chinese