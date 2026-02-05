import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import random
import torchvision
from torchvision import datasets, transforms

sys.path.append(os.path.realpath('..'))
import argparse
import datetime
from attack_min_x_theta_adam import uap_dm
from attacks_sga_ori import uap_sga
from torchvision import datasets, transforms

from utils import model_cifar10
from prepare_imagenet_data import create_imagenet_npy


def seed_torch(seed=0):  # random seed
    torch.random.fork_rng()
    rng_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)
    torch.random.set_rng_state(rng_state)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    print(args)
    if not os.path.exists(args.uaps_save):
        try:
            # exist_ok=True 防止在并发情况下报错，makedirs 可以创建多级目录
            os.makedirs(args.uaps_save, exist_ok=True)
            print(f"Directory did not exist, created: {args.uaps_save}")
        except OSError as e:
            print(f"Error creating directory {args.uaps_save}: {e}")
    seed_torch(args.seed)
    time1 = datetime.datetime.now()
    dir_uap = args.uaps_save
    batch_size = args.batch_size
    DEVICE = torch.device("cuda:0")


    center_crop = 32
    transforms_normalize = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/torch_datasets', train=True,
                                            download=True,
                                            transform=transforms_normalize)
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    loader_eval = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=24)
    model = model_cifar10(args.model_name)
    if not os.path.exists(dir_uap):
        os.makedirs(dir_uap)

    nb_epoch = args.epoch
    eps = args.alpha / 255
    beta = args.beta
    step_decay = args.step_decay
    losses_min = []
    if args.dm:
        uap, losses, losses_min = uap_dm(model, loader, nb_epoch, eps, beta, step_decay, loss_function=args.cross_loss,
                                         batch_size=batch_size, loader_eval=loader_eval, dir_uap=dir_uap,
                                         center_crop=center_crop, Momentum=args.Momentum, img_num=args.num_images,
                                         rho=args.rho, aa=args.aa, cc=args.cc, steps=args.steps,
                                         smooth_rate=args.smooth_rate)
    else:
        uap, losses = uap_sga(model, loader, nb_epoch, eps, beta, step_decay, loss_function=args.cross_loss,
                              batch_size=batch_size, minibatch=args.minibatch, loader_eval=loader_eval, dir_uap=dir_uap,
                              center_crop=center_crop, iter=args.iter, Momentum=args.Momentum, img_num=args.num_images)

    if args.dm:
        save_name = 'dm_' + args.model_name
    else:
        save_name = 'sga_' + args.model_name

    plt.plot(losses)
    if len(losses_min) > 0:
        np.save(dir_uap + "losses_min.npy", losses_min)
    np.save(dir_uap + "losses.npy", losses)
    plt.savefig(dir_uap + save_name + '_loss_epoch.png')

    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/igps_622/la/imagenet/train/',
                        help='training set directory')
    parser.add_argument('--uaps_save', default='/mnt/igps_622/la/DM-UAP-main/checkpoint/',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='batch size', default=125)

    parser.add_argument('--alpha', type=float, default=10, help='aximum perturbation value (L-infinity) norm')
    parser.add_argument('--beta', type=float, default=9, help='clamping value')

    parser.add_argument('--epoch', type=int, default=20, help='epoch num')
    parser.add_argument('--dm', type=int, default=1, help='choose to run DM-UAP(1) or SGA(0)')
    parser.add_argument('--num_images', type=int, default=10000, help='num of training images')
    parser.add_argument('--model_name', default='ViT-B', help='proxy model')
    parser.add_argument('--cross_loss', type=int, default=1, help='loss type,default is 1,cross_entropy loss')
    parser.add_argument('--step_decay', type=float, default=0.1, help='delta step size')

    # Parameters only for SGA
    parser.add_argument('--minibatch', type=int, help='inner batch size for SGA', default=10)
    parser.add_argument('--iter', type=int, default=4, help='inner iteration num')
    parser.add_argument('--Momentum', type=int, default=0, help='Momentum item')

    # Parameters only for DM-UAP
    parser.add_argument('--rho', type=float, default=4, help='rho of min-theta')
    parser.add_argument('--steps', type=int, default=10, help='min_theta_steps')
    parser.add_argument('--aa', type=float, default=25, help='eps of min-x, diffrent from eps of max-delta')
    parser.add_argument('--cc', type=int, default=10, help='min_x_steps')
    parser.add_argument('--smooth_rate', type=float, default=-0.2, help='label smoothing rate')
    parser.add_argument('--seed', type=int, default=0, help='seed used')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
