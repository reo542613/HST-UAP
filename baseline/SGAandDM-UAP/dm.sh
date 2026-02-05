#!/bin/bash

# ========== DM-UAP 多模型顺序攻击+评估脚本 ==========

# 定义模型列表
models=(  "ResNet56")

# 基础配置
data_dir_train="/mnt/igps_622/la/imagenet/train/"
data_dir_val="/mnt/igps_622/la/imagenet/val/"
base_checkpoint_dir="/mnt/igps_622/la/DM-UAP-main/SGA_cifar10"

# 攻击参数
batch_size=125
alpha=10
epoch=20
dm=0
num_images=50000
Momentum=0
cross_loss=1
rho=4
steps=10
aa=25
cc=10
smooth_rate=0.2

# 评估参数
eval_batch_size=125
eval_number=10000

# 循环遍历每个模型
for model in "${models[@]}"; do
    echo "========================================"
    echo "Starting attack on $model ..."
    echo "========================================"

    # 创建保存目录
    uaps_save_path="${base_checkpoint_dir}/${model}/"
    mkdir -p "$uaps_save_path"
#
#    # 训练（生成 UAP）
#    CUDA_VISIBLE_DEVICES=1 python cifar10_attack.py \
#        --data_dir "$data_dir_train" \
#        --uaps_save "$uaps_save_path" \
#        --batch_size $batch_size \
#        --alpha $alpha \
#        --epoch $epoch \
#        --dm $dm \
#        --num_images $num_images \
#        --model_name "$model" \
#        --Momentum $Momentum \
#        --cross_loss $cross_loss \
#        --rho $rho \
#        --steps $steps \
#        --aa $aa \
#        --cc $cc \
#        --smooth_rate $smooth_rate
#
    echo "$model attack finished!"
    echo ""

    # 构建 delta 文件名（根据你的命名规则）
    # 格式: dm_[data_num]_[epoch]epoch_[batchsize]batch.pth
    delta_file_name="sga_${num_images}_${epoch}epoch_${batch_size}batch.pth"
    uaps_save_path_eval="${uaps_save_path}${delta_file_name}"
    uaps_save_path2="${uaps_save_path}result.log"

    echo "Starting evaluation on $model ..."
    echo "Delta file: $uaps_save_path_eval"

    # 评估
    python cifar10_eval.py \
        --data_dir "$data_dir_val" \
        --uaps_save "$uaps_save_path_eval" \
        --batch_size $eval_batch_size \
        --number $eval_number \
        --model_name all 2>&1 | tee -a "$uaps_save_path2"

    echo "$model evaluation finished!"
    echo ""
    echo ""
done

echo "All models attack and evaluation finished!"