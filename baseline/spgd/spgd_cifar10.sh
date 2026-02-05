
models=( "ResNet56")
base_dir="/mnt/igps_622/la/SPGDandSGA/spgd_cifar"

for m in "${models[@]}"; do
    echo "========== Training $m =========="
    save_path="${base_dir}/${m}/"
    mkdir -p "$save_path"

    CUDA_VISIBLE_DEVICES=0 python cifar_attack.py \
        --data_dir /mnt/igps_622/la/imagenet/train/ \
        --uaps_save "$save_path" \
        --batch_size 125 \
        --alpha 10 \
        --epoch 20 \
        --spgd 1 \
        --num_images 50000 \
        --model_name "$m" \
        --Momentum 0 \
        --cross_loss 0
done
echo "All models finished!"