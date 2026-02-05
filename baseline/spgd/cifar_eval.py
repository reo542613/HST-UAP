import os
import sys
import torch
import argparse
import datetime
import torchvision
from torchvision import datasets, transforms
sys.path.append(os.path.realpath('..'))

from utils import loader_imgnet, model_cifar10, evaluate

def main(args):
    print(args)
    DEVICE = torch.device("cuda:0")
    time1 = datetime.datetime.now()
    dir_data = args.data_dir
    dir_uap = args.uaps_save
    batch_size = args.batch_size
    model_dimension = 299 if args.model_name == 'inception_v3' else 256
    center_crop = 299 if args.model_name == 'inception_v3' else 224
    center_crop = 32
    transforms_normalize = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data/torch_datasets', train=False,
                                           download=True,
                                           transform=transforms_normalize)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    if args.model_name.lower() == 'all':
        model_list = [
            'VGG19', 'ResNet56', 'DenseNet121'
        ]
    else:
        model_list = [args.model_name]
    for model_name in model_list:
        model = model_cifar10(model_name)
        uap = torch.load(dir_uap)
        _, _, _, _, outputs, labels, y_outputs = evaluate(model, loader, uap=uap, batch_size=batch_size, DEVICE=DEVICE)
        print(f"[{model_name}] Results:")
        print('true image Accuracy:', sum(y_outputs == labels) / len(labels))
        print('adversarial image Accuracy:', sum(outputs == labels) / len(labels))
        print('fooling rate:', 1 - sum(outputs == labels) / len(labels))
        print('fooling ratio:', 1 - sum(y_outputs == outputs) / len(labels))

        time2 = datetime.datetime.now()
        print("time consumed: ", time2 - time1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/igps_622/la/imagenet/val/',
                        help='training set directory')
    parser.add_argument('--uaps_save', default='./uaps_save/spgd/spgd_10000_20epoch_250batch.pth',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='', default=250)
    parser.add_argument('--model_name', default='all', help='loss type')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))