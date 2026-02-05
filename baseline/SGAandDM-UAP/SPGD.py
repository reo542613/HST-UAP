import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import numpy as np
import argparse
import os
import time
import sys
from tqdm import tqdm

# ==========================================
# 1. 基础配置与 Normalizer
# ==========================================

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# ==========================================
# 2. 智能模型加载器
# ==========================================
def load_model_smart(model_name):
    print(f"Loading model: {model_name}...")

    timm_map = {
        'ViT-B': 'vit_base_patch16_224',
        'ViT-L': 'vit_large_patch16_224',
        'DeiT-S': 'deit_small_patch16_224',
        'DeiT-B': 'deit_base_patch16_224',
        'Swin-T': 'swin_tiny_patch4_window7_224',
        'Swin-S': 'swin_small_patch4_window7_224',
        'Swin-B': 'swin_base_patch4_window7_224'
    }

    base_model = None

    if model_name in timm_map:
        try:
            base_model = timm.create_model(timm_map[model_name], pretrained=True)
        except Exception as e:
            print(f"Timm load failed: {e}")
            sys.exit(1)
    else:
        tv_name = model_name.lower()
        if hasattr(torchvision.models, tv_name):
            model_fn = getattr(torchvision.models, tv_name)
            try:
                base_model = model_fn(weights='DEFAULT')
            except:
                base_model = model_fn(pretrained=True)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    base_model = nn.DataParallel(base_model).cuda()
    normalize_layer = Normalizer(mean=IMGNET_MEAN, std=IMGNET_STD).cuda()
    model = nn.Sequential(normalize_layer, base_model).cuda()

    model.eval()
    return model





# ==========================================
# 4. Loss 计算辅助函数
# ==========================================
def cal_loss(loader, model, delta, beta, loss_function):
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()

    max_eval_batches = 20

    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            if i >= max_eval_batches: break

            x_val = data.cuda()
            outputs_ori = model(x_val)
            _, target_label = torch.max(outputs_ori, 1)

            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)

            if loss_function:
                loss = torch.mean(torch.min(loss_fn(outputs, target_label), beta))
            else:
                loss = torch.mean(outputs.gather(1, target_label.unsqueeze(1)).squeeze(1))
            loss_total += loss.item()

    return loss_total / min((i + 1), max_eval_batches)


# ==========================================
# 5. UAP 训练算法 (SGA)
# ==========================================
def uap_sga(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=True,
            batch_size=32, minibatch=10, loader_eval=None, dir_uap=None,
            center_crop=224, iter=10, target_att=-1):
    model.eval()
    delta = torch.zeros(1, 3, center_crop, center_crop).cuda()

    criterion = nn.CrossEntropyLoss(reduction='none')
    beta_t = torch.tensor([beta]).cuda()

    def clamped_loss(output, target):
        return torch.mean(torch.min(criterion(output, target), beta_t))

    print(f"Start Training UAP (Eps={eps:.4f}, Epochs={nb_epoch})...")

    for epoch in range(nb_epoch):
        start_t = time.time()
        eps_step = eps * step_decay

        train_fooling_cnt = 0
        train_total = 0
        train_loss_sum = 0

        # === [修改 1] 使用 tqdm 包装 loader ===
        # desc: 进度条左边的描述文字
        # leave=True: 跑完后保留进度条痕迹
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{nb_epoch}", leave=True)

        # 注意：这里遍历的是 pbar，而不是原来的 loader
        for i, (data, labels) in enumerate(pbar):
            x_val = data.cuda()

            # 1. 获取原始标签 (Target)
            with torch.no_grad():
                outputs_ori = model(x_val)
                _, target_label = torch.max(outputs_ori, 1)

            current_bs = x_val.shape[0]

            # 计算更新次数
            num_updates = max(1, (iter * current_bs) // minibatch)

            noise_inner_all = []
            delta_inner = delta.detach().clone()

            # --- SGA 内层循环 (计算梯度) ---
            for j in range(num_updates):
                indices = np.random.choice(current_bs, min(minibatch, current_bs), replace=False)
                pert = torch.zeros_like(delta).requires_grad_()

                adv_inputs = torch.clamp(x_val[indices] + delta_inner + pert, 0, 1)
                outputs = model(adv_inputs)

                if loss_function:
                    loss = clamped_loss(outputs, target_label[indices])
                else:
                    loss = -torch.mean(outputs.gather(1, target_label[indices].unsqueeze(1)).squeeze(1))

                loss.backward()

                grad = pert.grad.data
                noise_inner_all.append(grad)

                delta_inner = delta_inner + grad.sign() * eps_step
                delta_inner = torch.clamp(delta_inner, -eps, eps)

            # --- 外层更新 (Batch Update) ---
            if len(noise_inner_all) > 0:
                stacked_grads = torch.cat(noise_inner_all, dim=0)
                avg_grad = torch.mean(stacked_grads, dim=0, keepdim=True)
                grad_sign = avg_grad.sign()

                if target_att == -1:
                    delta = delta + grad_sign * eps_step
                else:
                    delta = delta - grad_sign * eps_step

                delta = torch.clamp(delta, -eps, eps)

            # === 计算统计数据 ===
            with torch.no_grad():
                adv_inputs_check = torch.clamp(x_val + delta, 0, 1)
                outputs_adv = model(adv_inputs_check)
                pred_adv = outputs_adv.argmax(dim=1)

                fooling_batch = (pred_adv != target_label).sum().item()
                train_fooling_cnt += fooling_batch
                train_total += current_bs

                if loss_function:
                    loss_batch = clamped_loss(outputs_adv, target_label)
                    train_loss_sum += loss_batch.item()

            # === [修改 2] 实时更新进度条右侧的指标 ===
            current_fr = train_fooling_cnt / train_total
            current_loss = train_loss_sum / (i + 1)

            # 这会在进度条后面显示: Loss=xxx, FR=xxx
            pbar.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "FR": f"{current_fr:.2%}"
            })

        # Epoch 结束后的汇总计算
        train_fr = train_fooling_cnt / train_total if train_total > 0 else 0
        train_avg_loss = train_loss_sum / (i + 1)

        # 构造最终消息 (可以用来记录到日志文件)
        # 注意：由于使用了 tqdm，print 会在进度条下方输出
        msg = (f"Summary Ep {epoch + 1}: "
               f"Time: {time.time() - start_t:.1f}s | "
               f"Final FR: {train_fr:.2%} | "
               f"Final Loss: {train_avg_loss:.4f}")

        # 如果有验证集
        # if loader_eval:
        #     # 注意：cal_loss 函数如果耗时较久，建议也可以在 cal_loss 内部加个 tqdm
        #     val_loss = cal_loss(loader_eval, model, delta, beta_t, loss_function)
        #     msg += f" | Val Loss: {val_loss:.4f}"

        print(msg)

        if dir_uap:
            torch.save(delta.cpu(), os.path.join(dir_uap, f'uap_epoch_{epoch + 1}.pth'))

    return delta.detach()


# ==========================================
# 6. 测试评估函数
# ==========================================
def evaluate_uap(model, loader, delta):
    print("\n" + "=" * 40)
    print("Evaluating Performance...")
    delta = delta.cuda()

    total = 0
    top1_clean = 0
    top1_adv = 0
    fooling_cnt = 0
    max_test = 2000

    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            if total >= max_test: break

            data = data.cuda()
            labels = labels.cuda()

            out_clean = model(data)
            pred_clean = out_clean.argmax(dim=1)

            data_adv = torch.clamp(data + delta, 0, 1)
            out_adv = model(data_adv)
            pred_adv = out_adv.argmax(dim=1)

            total += data.size(0)
            top1_clean += (pred_clean == labels).sum().item()
            top1_adv += (pred_adv == labels).sum().item()
            fooling_cnt += (pred_clean != pred_adv).sum().item()

    print(f"Evaluated {total} images")
    print(f"Clean Acc:    {top1_clean / total:.2%}")
    print(f"Adv Acc:      {top1_adv / total:.2%}")
    print(f"Fooling Rate: {fooling_cnt / total:.2%}")
    print("=" * 40)


# ==========================================
# 7. 主程序
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/igps_622/la/imagenet/', help='Dataset path')
    parser.add_argument('--save_dir', default='/mnt/igps_622/la/DM-UAP-main/SPGD/', help='Save path')
    parser.add_argument('--model_name', default='ViT-B', help='Target model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epoch', type=int, default=5)
    parser.add_argument('--eps', type=float, default=10.0 / 255.0)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Config: {args}")

    model = load_model_smart(args.model_name)

    print("Loading Data...")
    from torch.utils.data import Subset

    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]
    transforms_normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    transform_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Original datasets
    trainset = datasets.ImageFolder('/mnt/igps_622/la/imagenet/train/', transform=transform_data)
    testset = datasets.ImageFolder('/mnt/igps_622/la/imagenet/val/', transform=transform_data)

    # trainset = datasets.ImageFolder('D:/imagedata/train/', transform=transform_data)
    # testset = datasets.ImageFolder('D:/imagedata/val/', transform=transform_data)

    #  Function to select first N images per class 每个种类取十张
    def select_first_n_per_class(dataset, n=10):
        # Get class to indices mapping
        class_to_idx = dataset.class_to_idx

        # Store indices for selected samples
        selected_indices = []

        # Group indices by class
        for class_idx in range(len(class_to_idx)):
            # Get all indices for this class
            class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            # Select first n indices (or all if less than n)
            selected = class_indices[:min(n, len(class_indices))]
            selected_indices.extend(selected)

        # Create a subset of the dataset
        return Subset(dataset, selected_indices)

    # Create new datasets with first 10 images per class
    trainset = select_first_n_per_class(trainset, n=10)
    testset = torchvision.datasets.ImageFolder('/mnt/igps_622/la/imagenet/val/', transform=transform_data)
    # 10000

    # #每个种类取一张，前500个种类
    # def select_first_n_per_class(dataset, n=1, num_classes=500):
    #     # 获取类别到索引的映射
    #     class_to_idx = dataset.class_to_idx
    #
    #     # 存储选择的样本索引
    #     selected_indices = []
    #
    #     # 遍历前 num_classes 个类别
    #     for class_idx in range(min(num_classes, len(class_to_idx))):
    #         # 获取该类别所有样本的索引
    #         class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    #         # 选择前 n 个索引（如果少于 n，则取所有）
    #         selected = class_indices[:min(n, len(class_indices))]
    #         selected_indices.extend(selected)
    #
    #     # 创建数据集子集
    #     return Subset(dataset, selected_indices)
    #
    # trainset = select_first_n_per_class(trainset, n=1, num_classes=500)
    # testset = torchvision.datasets.ImageFolder('/mnt/igps_622/la/imagenet/val/', transform=transform_data)
    # # 500

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=16)

    best_delta = uap_sga(
        model=model,
        loader=train_loader,
        nb_epoch=args.nb_epoch,
        eps=args.eps,
        loader_eval=test_loader,
        dir_uap=args.save_dir,
        batch_size=args.batch_size
    )

    save_path = os.path.join(args.save_dir, f'{args.model_name}_final_uap.pth')
    torch.save(best_delta.cpu(), save_path)

    print(f"UAP saved to {save_path}")
    print(f"\nTraining model: {args.model}")
    # print(f"\nTraining model: {model_name}")
    import timm
    import torchvision.models as models
    test_model_names = ['AlexNet', 'VGG16', 'VGG19', 'ResNet152', 'GoogLeNet', 'ViT-B', 'ViT-L', 'DeiT-S', 'DeiT-B',
                        'Swin-T', 'Swin-S', 'Swin-B']
    for test_model_name in test_model_names:
        print(f"\nTesting model: {test_model_name}")
        # print(f"\n==> Start testing {test_model_name} ..")

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
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        test_model = test_model.cuda()
        for params in test_model.parameters():
            params.requires_grad = False
        test_model.eval()
        evaluate_uap(test_model, test_loader, best_delta)


if __name__ == '__main__':
    main()