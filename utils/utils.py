# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
from skimage.metrics import structural_similarity as ssim
import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz
import torch

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF  # 导入高斯模糊支持
import torchvision.utils as vutils
@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        grid (Tensor): the tensor containing grid of images.
    Example:
        See this notebook
        `here <https://github.com/pytorch/vision/blob/master/examples/python/visualization_utils.ipynb>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)


def gaussian_kernel(size: int, sigma: float):
    """ 生成 2D 高斯核 """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def calculate_ssim(img1, img2):
    """计算两张图像的 SSIM"""
    img1 = img1.cpu().numpy().transpose(1, 2, 0)  # 转换为 (H, W, C)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    return ssim(img1, img2, channel_axis=2, win_size=7, data_range=1.0)

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



# 优化后的 flow_st 函数，去除 device 依赖

class Loss_flow(nn.Module):
    def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super(Loss_flow, self).__init__()

    def forward(self, flows):
        paddings = (1, 1, 1, 1, 0, 0, 0, 0)
        padded_flows = F.pad(flows, paddings, "constant", 0)

        # #rook
        shifted_flowsr = torch.stack([
            padded_flows[:, :, 2:, 1:-1],  # bottom mid
            padded_flows[:, :, 1:-1, :-2],  # mid left
            padded_flows[:, :, :-2, 1:-1],  # top mid
            padded_flows[:, :, 1:-1, 2:],  # mid right
        ], -1)

        flowsr = flows.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        _, h, w, _ = flowsr[:, 0].shape

        loss0 = torch.norm((flowsr[:, 0] - shifted_flowsr[:, 0]).view(-1, 4), p=2, dim=(0), keepdim=True) ** 2
        loss1 = torch.norm((flowsr[:, 1] - shifted_flowsr[:, 1]).view(-1, 4), p=2, dim=(0), keepdim=True) ** 2

        return torch.max(torch.sqrt((loss0 + loss1) /(h * w)))

class Loss_flow_grok(nn.Module):
    def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super(Loss_flow_grok, self).__init__()
        self.neighbours = torch.tensor(neighbours, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda() / neighbours.sum()
        self.padding = (1, 1, 1, 1)

    def forward(self, flows, epoch):
        # 平滑损失
        padded_flows = F.pad(flows, self.padding, "constant", 0)
        neighbor_mean = F.conv2d(padded_flows, self.neighbours.repeat(2, 1, 1, 1), groups=2)
        smooth_loss = (0.1 * (1 - epoch / 50) + 0.01) * torch.mean(torch.norm(flows - neighbor_mean, p=2, dim=(1, 2, 3)) ** 2)

        # 幅度损失
        mag_loss = (0.01 * (epoch / 50) + 0.001) * torch.mean(torch.norm(flows, p=2, dim=(1, 2, 3)) ** 2)

        # 多样性鼓励（惩罚过小幅度）
        var_loss = -0.01 * torch.mean(torch.var(flows.view(flows.size(0), 2, -1), dim=2) ** 2)

        total_loss = smooth_loss + mag_loss + var_loss
        return total_loss

def cal_l2dist(X1, X2):
    list_bhat = []
    list_hdist = []
    list_ssim = []
    list_l2 = []
    batch, nc, _, _ = X1.shape

    for i in range(batch):
        img1 = X1[i].unsqueeze(0)
        img2 = X2[i].unsqueeze(0)
        x1 = img1.mul(255).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        x2 = img2.mul(255).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        list_l2.append(np.sqrt(np.sum((x1 - x2) ** 2)))
    return np.mean(list_l2)


def norm_ip(img):
    min = float(img.min())
    max = float(img.max())
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img

