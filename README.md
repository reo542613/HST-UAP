# HST-UAP: Hierarchical Spatial Transformation Universal Adversarial Perturbation

HST-UAP is a novel universal adversarial attack framework that integrates **hierarchical spatial transformation** (flow field) and **additive perturbation** to generate highly transferable universal adversarial perturbations (UAPs). By using two structurally independent generators — FlowGEN for spatial flow and NoiseGEN for additive noise — HST-UAP achieves explicit disentanglement from the generation stage, overcoming the structural entanglement and limited diversity of prior methods.

This approach leads to stronger attack success rates, better generalization, and superior cross-model transferability across both CNNs and modern Vision Transformers.

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
