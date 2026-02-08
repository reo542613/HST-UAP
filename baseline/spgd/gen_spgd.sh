#CUDA_VISIBLE_DEVICES='2' python imagenet_attack.py --data_dir /mnt/imagenet/train/ --uaps_save /mnt/SPGDandSGA/spgd/VITL/ --batch_size 125 --alpha 10 --epoch 20 --spgd 1 --num_images 10000 --model_name ViT-L --Momentum 0 --cross_loss 0


#!/bin/bash
# 顺序训练 AlexNet / GoogLeNet / VGG16 / VGG19 / ResNet152
models=("AlexNet" "GoogLeNet" "VGG16" "VGG19" "ResNet152")
models=( "VGG19" "ResNet152")
base_dir="/mnt/SPGDandSGA/spgd"

for m in "${models[@]}"; do
    echo "========== Training $m =========="
    save_path="${base_dir}/${m}/"
    mkdir -p "$save_path"

    CUDA_VISIBLE_DEVICES=1 python imagenet_attack.py \
        --data_dir /mnt/imagenet/train/ \
        --uaps_save "$save_path" \
        --batch_size 125 \
        --alpha 10 \
        --epoch 20 \
        --spgd 1 \
        --num_images 10000 \
        --model_name "$m" \
        --Momentum 0 \
        --cross_loss 0
done
echo "All models finished!"
