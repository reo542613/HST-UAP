# HST-UAP: Hierarchical Spatial Transformation Universal Adversarial Perturbation

HST-UAP is a novel universal adversarial attack framework that integrates hierarchical spatial transformation (flow field) and additive perturbation to generate highly transferable universal adversarial perturbations (UAPs). It uses two structurally independent generators — FlowGEN for spatial flow and NoiseGEN for additive noise — achieving explicit disentanglement from the generation stage, resulting in stronger attack performance, better generalization, and superior cross-model transferability across CNNs and Vision Transformers.

## Features

- Dual-generator architecture for independent control of flow field and additive noise
- Hierarchical suppression and joint optimization for imperceptible yet powerful UAPs
- Strong transferability to CNNs (AlexNet, VGG16/19, ResNet152, GoogLeNet) and Transformers (ViT, DeiT, Swin, ConvNeXt, BEiT)
- Support for ImageNet (large-scale) and CIFAR-10 (subset) datasets
- Comprehensive evaluation: attack success rate (ASR), L2 distance, hyperparameter ablation, visualization grids, and cross-model transfer

## Installation

### Dependencies

```bash
pip install torch torchvision timm tqdm matplotlib numpy opencv-python pillow pyyaml
Clone the Repository
Bashgit clone https://github.com/reo542613/HST-UAP.git
cd HST-UAP
Dataset Preparation
Place ImageNet-1K at the following path:
text/mnt/igps_622/la/imagenet/
├── train/
└── val/
CIFAR-10 is automatically downloaded via torchvision if needed.
Training
Basic training command:
Bashpython main.py \
  --model VGG19 \
  --dataset ImageNet \
  --batch_size 128 \
  --epochs 60 \
  --lr 0.001 \
  --gamma 0.2 \
  --alpha1 0.05 \
  --alpha2 0.01 \
  --beta1 0.0085 \
  --beta2 0.05 \
  --train_num 10000 \
  --tau 0.1 \
  --allow 10./255
Key Arguments

--model: Target model for perturbation generation (VGG19, ResNet152, VGG16, GoogLeNet, etc.)
--train_num: Training samples per class (e.g., 500 → 500 classes × 1 image; 10000 → 1000 classes × 10 images)
--gamma, --alpha1, --alpha2, --beta1, --beta2: Loss trade-off weights
--tau: Max flow magnitude
--allow: Linf bound for additive noise

Checkpoints are saved every 10 epochs to:
text/mnt/igps_622/la/GUAP/revise_checkpoint_L_2loss/{model}/...
Evaluation
After training, the best perturbation (highest ASR + lowest loss) is automatically used for evaluation every 10 epochs on multiple models.
Supported test models:

CNNs: AlexNet, VGG16, VGG19, ResNet152, GoogLeNet
Transformers: ViT-B/L, DeiT-S/B, Swin-T/S/B

Logs are saved to:
text/mnt/igps_622/la/GUAP/revise_checkpoint_L_2loss/{model}/{train_num}_{model}_ImageNetGUAP_checkpoint.log
Visualization
Adversarial examples, flow fields, and hyperparameter grids are saved to:
text/mnt/igps_622/la/GUAP/revise_checkpoint_L_2loss/{model}/savefig/
To generate hyperparameter ablation grids (after training):
Bashpython vis_hyper_grid.py  # Run in the directory with .pth files
Citation
If you use HST-UAP in your research, please cite:
bibtex@article{hst-uap,
  title={HST-UAP: Hierarchical Spatial Transformation Universal Adversarial Perturbation},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2025}
}
Acknowledgments

Built upon foundational UAP works and modern architectures (ViT, DeiT, Swin, ConvNeXt, BEiT)
Uses timm library for pre-trained Transformer models

License
MIT License
Contact

GitHub: https://github.com/reo542613/HST-UAP
Issues: https://github.com/reo542613/HST-UAP/issues

For any questions or collaborations, feel free to open an issue.
text
