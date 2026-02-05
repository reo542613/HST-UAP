
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import time
import timm
from tqdm import tqdm
import logging
import sys
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import structural_similarity_index_measure



def flow_st(images, flows, batch_size):
    # print(images.shape)
    H, W = images.size()[2:]

    # basic grid: tensor with shape (2, H, W) with value indicating the
    # pixel shift in the x-axis or y-axis dimension with respect to the
    # original images for the pixel (2, H, W) in the output images,
    # before applying the flow transforms
    grid_single = torch.stack(
        torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    ).float()

    grid = grid_single.repeat(batch_size, 1, 1, 1)  # 100,2,28,28

    images = images.permute(0, 2, 3, 1)  # 100, 28,28,1

    grid = grid.cuda()
    grid_new = grid + flows
    # assert 0

    sampling_grid_x = torch.clamp(
        grid_new[:, 1], 0., (W - 1.)
    )
    sampling_grid_y = torch.clamp(
        grid_new[:, 0], 0., (H - 1.)
    )

    # now we need to interpolate

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a square around the point of interest
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate image boundaries
    # - 2 for x0 and y0 helps avoiding black borders
    # (forces to interpolate between different points)
    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    b = torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, H, W).cuda()
    # assert 0
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    perturbed_image = wa * Ia + wb * Ib + wc * Ic + wd * Id

    perturbed_image = perturbed_image.permute(0, 3, 1, 2)

    return perturbed_image


model_name_map = {
    
    'EfficientNetV2-S': 'tf_efficientnetv2_s',
    'EfficientNetV2-M': 'tf_efficientnetv2_m',

    
    'ConvNeXt-T': 'convnext_tiny',
    'ConvNeXt-S': 'convnext_small',
    'ConvNeXt-B': 'convnext_base',
    'ConvNeXt-B-22K': 'convnext_base.fb_in22k_ft_in1k', 

    
    'BEiT-B': 'beit_base_patch16_224',
    'BEiTv2-B': 'beitv2_base_patch16_224',

    
    'DINOv2-S': 'vit_small_patch14_dinov2.lvd142m',
    'DINOv2-B': 'vit_base_patch14_dinov2.lvd142m',

    
    'ViT-B': 'vit_base_patch16_224',
    'DeiT-S': 'deit_small_patch16_224',
    'DeiT-B': 'deit_base_patch16_224',
    'Swin-T': 'swin_tiny_patch4_window7_224',
    'Swin-S': 'swin_small_patch4_window7_224',
    'Swin-B': 'swin_base_patch4_window7_224',
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Imagenet', help='CIFAR10, CIFAR100, or Imagenet')
    parser.add_argument('--model', type=str, default='1', help='')
    parser.add_argument('--flow_path', type=str, default='/mnt/DeiT-S/flow.pth', help='Path to saved flow_field (.pth)')
    parser.add_argument('--noise_path', type=str, default='/mnt//DeiT-S/.pth', help='Path to saved perb_noise (.pth)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--data_path', type=str, default='/imagenet/val/', help='Path to dataset')
    parser.add_argument('--log_dir', type=str, default='/DeiT-S/',
                        help='Directory to save logs')
    return parser.parse_args()



def setup_logger(log_dir):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    
    log_file_path = os.path.join(log_dir, f"Test_Result_{get_args().model}.log")

   
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)

    
    if not logger.handlers:
       
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
        logger.addHandler(fh)

       
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
        logger.addHandler(sh)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    return logger, log_file_path


def main():
    args = get_args()

   
    logger, log_file = setup_logger(args.log_dir)
    logger.info("=" * 50)
    logger.info(f"Starting Test Script")
    logger.info(f"Log saved to: {log_file}")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 50)

    
    logger.info("Initializing LPIPS model (AlexNet backbone)...")
    try:
        loss_fn_lpips = lpips.LPIPS(net='alex').cuda()
    except Exception as e:
        logger.error(f"Failed to load LPIPS: {e}")
        return

    
    logger.info(f"Loading dataset: {args.dataset}...")
    if args.dataset == 'CIFAR10':
        dataset_mean = [0.4914, 0.4822, 0.4465]
        dataset_std = [0.2023, 0.1994, 0.2010]
        testset = datasets.CIFAR10(root='./data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(dataset_mean, dataset_std)
                                   ]))
    elif args.dataset == 'CIFAR100':
        dataset_mean = [0.5071, 0.4867, 0.4408]
        dataset_std = [0.2675, 0.2565, 0.2761]
        testset = datasets.CIFAR100(root='./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(dataset_mean, dataset_std)
                                    ]))
    elif args.dataset == 'Imagenet':
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        transforms_normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        transform_data = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms_normalize,
        ])
        testset = datasets.ImageFolder(args.data_path, transform=transform_data)
    else:
        raise ValueError("Unsupported dataset")

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    sample_img, _ = testset[0]
    nc, H, W = sample_img.shape

    
    logger.info(f"Loading perturbations:\n  Flow: {args.flow_path}\n  Noise: {args.noise_path}")

    try:
        flow_field = torch.load(args.flow_path).cuda()
        perb_noise = torch.load(args.noise_path).cuda()
    except Exception as e:
        logger.error(f"Error loading perturbation files: {e}")
        return

    mu = torch.tensor(dataset_mean).view(3, 1, 1).cuda()
    std = torch.tensor(dataset_std).view(3, 1, 1).cuda()
    unnormalize = lambda x: x * std + mu
    normalize = lambda x: (x - mu) / std

    test_model_names = [
        #'AlexNet', 'VGG16', 'VGG19', 'ResNet152', 'GoogLeNet',
         'DeiT-B', 'DeiT-S' ,  'Swin-B', 'Swin-S', 'Swin-T','ViT-B'

    ]
    # test_model_names = [args.model]

    
    quality_metrics = {
        'L2': 0.0, 'SSIM': 0.0, 'FID': 0.0, 'LPIPS': 0.0,
        'calculated': False
    }

    # ================= 5. 开始测试循环 =================
    logger.info("\nStarting evaluation loop...")
    header = f"{'Model':<12} | {'Clean Acc':<10} | {'Adv Acc':<10} | {'ASR':<10} | {'FID':<8} | {'LPIPS':<8} | {'SSIM':<8} | {'L2':<8} | {'Time':<8}"
    logger.info("-" * 100)
    logger.info(header)
    logger.info("-" * 100)

    for test_model_name in test_model_names:
        try:
           
            if test_model_name == 'VGG19':
                test_model = models.vgg19(pretrained=True)
            elif test_model_name == 'ResNet152':
                test_model = models.resnet152(pretrained=True)
            elif test_model_name == 'VGG16':
                test_model = models.vgg16(pretrained=True)
            elif test_model_name == 'GoogLeNet':
                test_model = models.googlenet(pretrained=True)
            elif test_model_name == 'AlexNet':
                test_model = models.alexnet(pretrained=True)
            # Timm Models
            elif test_model_name == 'ViT-B':
                test_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            elif test_model_name == 'ViT-L':
                test_model = timm.create_model('vit_large_patch16_224', pretrained=True)
            elif test_model_name == 'DeiT-S':
                test_model = timm.create_model('deit_small_patch16_224', pretrained=True)
            elif test_model_name == 'DeiT-B':
                test_model = timm.create_model('deit_base_patch16_224', pretrained=True)
            elif test_model_name == 'Swin-T':
                test_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            elif test_model_name == 'Swin-S':

                test_model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
            elif test_model_name == 'Swin-B':

                test_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            
            elif args.model == 'EfficientNetV2-S':
            
                test_model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
            elif args.model == 'EfficientNetV2-M':
                test_model = timm.create_model('tf_efficientnetv2_m', pretrained=True)

           
            elif args.model == 'ConvNeXt-T':
                test_model = timm.create_model('convnext_tiny', pretrained=True)
            elif args.model == 'ConvNeXt-S':
                test_model = timm.create_model('convnext_small', pretrained=True)
            elif args.model == 'ConvNeXt-B':
                test_model = timm.create_model('convnext_base', pretrained=True)
            elif args.model == 'ConvNeXt-B-22K':
               
                test_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)

            elif args.model == 'BEiT-B':
                
                test_model = timm.create_model('beit_base_patch16_224', pretrained=True)
            elif args.model == 'BEiTv2-B':
                test_model = timm.create_model('beitv2_base_patch16_224', pretrained=True)
            else:
                logger.warning(f"Skipping unsupported model: {test_model_name}")
                continue
        except Exception as e:
            logger.error(f"Error loading {test_model_name}: {e}")
            continue

        test_model = test_model.cuda()
        test_model.eval()
        for p in test_model.parameters():
            p.requires_grad = False

        correct_clean = 0
        correct_adv = 0
        total_samples = 0
        start_time = time.time()

        # FID 
        if not quality_metrics['calculated']:
            
            fid_metric = FrechetInceptionDistance(feature=2048).cuda()
            fid_metric.inception.cuda()  
            fid_metric.inception.eval()
            lpips_scores = []
            l2_scores = []
            ssim_scores = []

       
        pbar = tqdm(test_loader, desc=f"Testing {test_model_name}", leave=False)
        for X, y in pbar:
            X, y = X.cuda(), y.cuda()
            bs = X.size(0)

            with torch.no_grad():
                X_raw = unnormalize(X)  # [0, 1]
                X_st_raw = flow_st(X_raw, flow_field, bs)
                X_adv_raw = torch.clamp(X_st_raw + perb_noise, 0, 1)
                X_adv_norm = normalize(X_adv_raw)

                # if not quality_metrics['calculated']:
                #     # FID
                #     X_clean_uint8 = (X_raw * 255).clamp(0, 255).to(torch.uint8)
                #     X_adv_uint8 = (X_adv_raw * 255).clamp(0, 255).to(torch.uint8)
                #     fid_metric.update(X_clean_uint8, real=True)
                #     fid_metric.update(X_adv_uint8, real=False)
                #
                #     # LPIPS
                #     X_clean_lpips = X_raw * 2 - 1
                #     X_adv_lpips = X_adv_raw * 2 - 1
                #     batch_lpips = loss_fn_lpips(X_clean_lpips, X_adv_lpips)
                #     lpips_scores.append(batch_lpips.mean().item())
                #
                #     # SSIM & L2
                #     # for j in range(bs):
                #     #     ssim_val = calculate_ssim(X_raw[j], X_adv_raw[j])
                #     #     ssim_scores.append(ssim_val)
                #     batch_ssim_score = structural_similarity_index_measure(X_raw, X_adv_raw, data_range=1.0)
                #     ssim_scores.append(batch_ssim_score.item())
                #
                #     l2_val = cal_l2dist(X_raw, X_adv_raw)
                #     l2_scores.append(l2_val)

                # --- 推理 ---
                out_clean = test_model(X)
                pred_clean = out_clean.argmax(dim=1)
                correct_clean += (pred_clean == y).sum().item()

                out_adv = test_model(X_adv_norm)
                pred_adv = out_adv.argmax(dim=1)
                correct_adv += (pred_adv == pred_clean).sum().item()

                total_samples += bs

        # --- 后处理与记录 ---
        if not quality_metrics['calculated']:
            logger.info("Computing FID score (this may take a moment)...")
            try:
                quality_metrics['FID'] = fid_metric.compute().item()
                quality_metrics['LPIPS'] = np.mean(lpips_scores)
                quality_metrics['SSIM'] = np.mean(ssim_scores)
                quality_metrics['L2'] = np.mean(l2_scores)
                quality_metrics['calculated'] = True
                del fid_metric 
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")

        acc_clean = correct_clean / total_samples
        acc_adv = correct_adv / total_samples
        asr = (1 - acc_adv) * 100
        elapsed = time.time() - start_time

        result_str = (f"{test_model_name:<12} | {acc_clean:.4f}     | {acc_adv:.4f}     | {asr:.2f}%     | "
                      f"{quality_metrics['FID']:.4f}   | {quality_metrics['LPIPS']:.4f}   | "
                      f"{quality_metrics['SSIM']:.4f}   | {quality_metrics['L2']:.4f}   | {elapsed:.1f}s")

        logger.info(result_str)

        torch.cuda.empty_cache()

    logger.info("-" * 100)
    logger.info("Testing completed.")


if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    main()
