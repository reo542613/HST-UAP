# Notice that the form of "delta_file_name" is [dm/sga]_[data_num]_[epoch]epoch_[batchsize]batch.pth

#uaps_save_path="/mnt/igps_622/la/DM-UAP-main/DM/SWINB/SWINBdm_10000_3epoch_125batch.pth"
#uaps_save_path2="/mnt/igps_622/la/DM-UAP-main/DM/SWINB/result.log"
#
#python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ \
#  --uaps_save "$uaps_save_path" \
#  --batch_size 125  --number 50000 \
#  --model_name all 2>&1|tee -a "$uaps_save_path2


# === 1. AlexNet ===
CUDA_VISIBLE_DEVICES=1 python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ --uaps_save /mnt/igps_622/la/DM-UAP-main/SGA/AlexNet/sga_10000_20epoch_125batch.pth --batch_size 125 --number 50000 --model_name all 2>&1 | tee -a /mnt/igps_622/la/DM-UAP-main/SGA/AlexNet/result.log

# === 2. GoogLeNet ===
CUDA_VISIBLE_DEVICES=1 python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ --uaps_save /mnt/igps_622/la/DM-UAP-main/SGA/GoogLeNet/sga_10000_20epoch_125batch.pth --batch_size 125 --number 50000 --model_name all 2>&1 | tee -a /mnt/igps_622/la/DM-UAP-main/SGA/GoogLeNet/result.log

# === 3. VGG16 ===
CUDA_VISIBLE_DEVICES=1 python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ --uaps_save /mnt/igps_622/la/DM-UAP-main/SGA/VGG16/sga_10000_20epoch_125batch.pth --batch_size 125 --number 50000 --model_name all 2>&1 | tee -a /mnt/igps_622/la/DM-UAP-main/SGA/VGG16/result.log

# === 4. VGG19 ===
CUDA_VISIBLE_DEVICES=1 python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ --uaps_save /mnt/igps_622/la/DM-UAP-main/SGA/VGG19/sga_10000_20epoch_125batch.pth --batch_size 125 --number 50000 --model_name all 2>&1 | tee -a /mnt/igps_622/la/DM-UAP-main/SGA/VGG19/result.log

# === 5. ResNet152 ===
CUDA_VISIBLE_DEVICES=1 python imagenet_eval.py --data_dir /mnt/igps_622/la/imagenet/val/ --uaps_save /mnt/igps_622/la/DM-UAP-main/SGA/ResNet152/sga_10000_20epoch_125batch.pth --batch_size 125 --number 50000 --model_name all 2>&1 | tee -a /mnt/igps_622/la/DM-UAP-main/SGA/ResNet152/result.log
