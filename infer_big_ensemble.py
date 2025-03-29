import pickle
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2 as transforms
import time
import argparse
import os
from torch.autograd import Function
from collections import OrderedDict
from typing import List, Tuple
from cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
import pandas as pd

class ShakeFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_function = ShakeFunction.apply


def get_alpha_beta(batch_size, shake_config, device):
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = torch.rand(1)
    elif forward_shake and shake_image:
        alpha = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        alpha = torch.FloatTensor([0.5])

    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])

    alpha = alpha.to(device)
    beta = beta.to(device)

    return alpha, beta

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        mid_channels = in_channels // reduction

        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        n_batches, n_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, output_size=1).view(n_batches, n_channels)
        y = F.relu(self.fc1(y), inplace=True)
        y = F.sigmoid(self.fc2(y)).view(n_batches, n_channels, 1, 1)
        return x * y

class ResidualPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        se_reduction = 4
        self.se = SELayer(out_channels, se_reduction)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=self.stride, padding=0)
        y1 = self.conv1(y1)

        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=self.stride, padding=0)
        y2 = self.conv2(y2)

        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)

        return z


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shake_config):
        super().__init__()

        self.shake_config = shake_config

        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'skip', SkipConnection(in_channels, out_channels, stride))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)

        return self.shortcut(x) + y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        depth = config['depth']
        self.shake_config = (config['shake_forward'], config['shake_backward'],
                             config['shake_image'])

        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(input_shape[1],
                              16,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16,
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(in_channels,
                          out_channels,
                          stride=stride,
                          shake_config=self.shake_config))
            else:
                stage.add_module(
                    block_name,
                    block(out_channels,
                          out_channels,
                          stride=1,
                          shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = self.bn(self.conv(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

_act_inplace = True			# use inplace activation or not
_drop_inplace = True		# use inplace drop layer or not

def get_drop(drop_type, dropout_rate, inplace=True,
                block_size=7, gamma_scale=1.0):
    """
    get the drop layer.
    Args:
        drop_type:  the name of the drop layer, ['dropout', 'dropblock', 'droppath']
    """
    if drop_type == 'dropout':
        return nn.Dropout(dropout_rate, inplace=inplace)

    elif drop_type == 'dropblock':
        return DropBlock2d(drop_prob=dropout_rate,
                    block_size=block_size,
                    gamma_scale=gamma_scale,
                    inplace=inplace,
                    fast=True)

    elif drop_type == 'droppath':
        return DropPath(dropout_rate)

    else:
        raise NotImplementedError


def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7, gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_act(act, inplace=False, memory_efficient=False):
	"""get the activation functions"""
	if act == 'relu':
		return nn.ReLU(inplace=inplace)

	elif act == 'leakyrelu':
		return nn.LeakyReLU(0.01, inplace=inplace)
	
	elif act == 'swish':
		if memory_efficient:
			return MemoryEfficientSwish()
		else:
			return Swish(inplace=inplace)
	
	elif act == 'hardswish':
		return HardSwish(inplace=inplace)
	
	else:
		raise NotImplementedError


class Swish(nn.Module):
	"""
	Swish: Swish Activation Function
	Described in: https://arxiv.org/abs/1710.05941
	"""
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class SwishImplementation(torch.autograd.Function):
	"""
	A memory-efficient implementation of Swish function from
	https://github.com/lukemelas/EfficientNet-PyTorch
	"""

	@staticmethod
	def forward(ctx, i):
		result = i * torch.sigmoid(i)
		ctx.save_for_backward(i)
		return result

	@staticmethod
	def backward(ctx, grad_output):
		i = ctx.saved_tensors[0]
		sigmoid_i = torch.sigmoid(i)
		return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
	def forward(self, x):
		return SwishImplementation.apply(x)


class HardSwish(nn.Module):
	"""
	PyTorch has this, but not with a consistent inplace argmument interface.

	Searching for MobileNetV3`:
		https://arxiv.org/abs/1905.02244

	"""
	def __init__(self, inplace: bool = False):
		super(HardSwish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return F.hardswish(x, inplace=self.inplace)

class SEModule(nn.Module):

	def __init__(self, channels, reduction, act='relu'):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
								padding=0)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)
		
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
								padding=0)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x

class Bottleneck(nn.Module):
	"""
	Base class for bottlenecks that implements `forward()` method.
	"""
	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = self.se_module(out)

		# ResNeSt use dropblock afer each BN (&downsample)
		# https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/gluon/resnet.py
		if self.drop is not None and self.drop_type != 'dropblock':
			out = self.drop(out)

		out = out + residual

		out = self.relu(out)

		return out

class SEResNetBottleneck(Bottleneck):
	"""
	ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
	implementation and uses `stride=stride` in `conv1` and not in `conv2`
	(the latter is used in the torchvision implementation of ResNet).
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
					downsample=None, drop_p=None, act='relu',
					drop_type='dropout', dropblock_size=0, gamma_scale=1.0):
		super(SEResNetBottleneck, self).__init__()
		self.drop_p = drop_p
		self.drop_type = drop_type

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
								stride=1)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
								groups=groups, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)

		self.se_module = SEModule(planes * 4, reduction=reduction, act=act)
		self.downsample = downsample
		self.stride = stride
		self.drop = None
		if self.drop_p:
			# self.drop = nn.Dropout(drop_p, inplace=_drop_inplace)
			self.drop = get_drop(drop_type, self.drop_p, inplace=_drop_inplace,
									block_size=dropblock_size, gamma_scale=gamma_scale)

class SENet(nn.Module):

	def __init__(self, arch, block, layers, groups, reduction, dropout_p=0.2,
					inplanes=128, input_3x3=True, downsample_kernel_size=3,
					downsample_padding=1, num_classes=10, zero_init_residual=True,
					dataset='cifar10', split_factor=1, output_stride=8,
					act='relu', mix_act=False, mix_act_block=2,
					block_drop=False, block_drop_p=0.5,
					drop_type='dropout', crop_size=32
					):
		"""
		Parameters
		----------
		block (nn.Module): Bottleneck class.
			- For SENet154: SEBottleneck
			- For SE-ResNet models: SEResNetBottleneck
			- For SE-ResNeXt models:  SEResNeXtBottleneck
		layers (list of ints): Number of residual blocks for 4 layers of the
			network (layer1...layer4).
		groups (int): Number of groups for the 3x3 convolution in each
			bottleneck block.
			- For SENet154: 64
			- For SE-ResNet models: 1
			- For SE-ResNeXt models:  32
		reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
			- For all models: 16
		dropout_p (float or None): Drop probability for the Dropout layer.
			If `None` the Dropout layer is not used.
			- For SENet154: 0.2
			- For SE-ResNet models: None
			- For SE-ResNeXt models: None
		inplanes (int):  Number of input channels for layer1.
			- For SENet154: 128
			- For SE-ResNet models: 64
			- For SE-ResNeXt models: 64
		input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
			a single 7x7 convolution in layer0.
			- For SENet154: True
			- For SE-ResNet models: False
			- For SE-ResNeXt models: False
		downsample_kernel_size (int): Kernel size for downsampling convolutions
			in layer2, layer3 and layer4.
			- For SENet154: 3
			- For SE-ResNet models: 1
			- For SE-ResNeXt models: 1
		downsample_padding (int): Padding for downsampling convolutions in
			layer2, layer3 and layer4.
			- For SENet154: 1
			- For SE-ResNet models: 0
			- For SE-ResNeXt models: 0
		num_classes (int): Number of outputs in `last_linear` layer.
			- For all models: 1000
		
		dataset (str): 'imagenet', 'cifar10', 'cifar100'

		split_factor (int): divide the network into {split_factor} small networks
		
		mix_act (bool): whether use mixed activations, {ReLU and HardSwish}
		
		mix_act_block (int): the last (4 - {mix_act_block}) blocks use HardSwish act function
		
		block_drop (bool): whether use block drop layer or not
		
		drop_type (int): 'dropout', 'dropblock', 'droppath'
		
		block_drop_p (folat): drop probablity in drop layers
			- For dropout, 0.2 or 0.3
			- For dropblock, 0.1
			- For droppath, 0.1 or 0.2
		"""
		super(SENet, self).__init__()
		self.dataset = dataset

		# modification of activations
		self.act = act
		self.mix_act = mix_act
		self.mix_act_block = mix_act_block if self.mix_act else len(layers) + 1
		if self.mix_act_block < 4:
			print('INFO:PyTorch: last {} block(s) use'
					' hardswish activation function'.format(4 - self.mix_act_block))

		# modification of drop blocks
		self.crop_size = crop_size
		self.block_drop = block_drop
		dropblock_size = [0, 0, 3, 3]
		self.gamma_scales = [0, 0, 1.0, 1.0]
		self.dropblock_size = [int(x * crop_size / 224) for x in dropblock_size]
		self.drop_type = drop_type
		if self.block_drop:
			# add dropout or other drop layers within each block
			print('INFO:PyTorch: Using {} within blocks'.format(self.drop_type))
			block_drop_p = block_drop_p / (split_factor ** 0.5)
			n = sum(layers)
			if self.drop_type in ['dropout', 'droppath']:
				self.block_drop_ps = [block_drop_p * (i + 1) / (n + 1) for i in range(n)]
			else:
				block_drop_flag = [False, False, True, True]
				self.block_drop_ps = [block_drop_p] * n
				# a mixed drop manner
				j = 0
				for i in range(len(block_drop_flag)):
					if not block_drop_flag[i]:
						for k in range(j, j + layers[i]):
							self.block_drop_ps[k] = 0
						j += layers[i]

		# inplanes and base width of the bottleneck
		if groups == 1:
			self.groups = groups
			inplanes_dict = {'imagenet': {1: 64, 2: 44, 4: 32, 8: 24},
								'cifar10': {1: 16, 2: 12, 4: 8},
								'cifar100': {1: 16, 2: 12, 4: 8},
								'svhn': {1: 16, 2: 12, 4: 8},
							}

			self.inplanes = inplanes_dict[dataset][split_factor]

			if 'cifar' in dataset or 'svhn' in dataset:
				reduction = 4

		elif groups in [8, 16, 32, 64, 128]:
			# For resnext, just divide groups
			self.groups = groups
			if split_factor > 1:
				self.groups = int(groups / split_factor)
				print("INFO:PyTorch: Dividing {}, change groups from {} "
						"to {}.".format(arch, groups, self.groups))
			self.inplanes = 64
		
		else:
			raise NotImplementedError

		self.layer0_inplanes = self.inplanes

		if inplanes == 128:
			self.inplanes = self.inplanes * 2
		print('INFO:PyTorch: The initial inplanes of SENet is {}'.format(self.inplanes))
		print('INFO:PyTorch: The reduction of SENet is {}'.format(reduction))

		if 'imagenet' in dataset:
			layer0_modules = [
					('conv1', nn.Conv2d(3, self.layer0_inplanes, 3, stride=2, padding=1,
										bias=False)),
					('bn1', nn.BatchNorm2d(self.layer0_inplanes)),
					('relu1', get_act(act, inplace=_act_inplace)),
					('conv2', nn.Conv2d(self.layer0_inplanes, self.layer0_inplanes, 3, stride=1, padding=1,
										bias=False)),
					('bn2', nn.BatchNorm2d(self.layer0_inplanes)),
					('relu2', get_act(act, inplace=_act_inplace)),
					('conv3', nn.Conv2d(self.layer0_inplanes, self.inplanes, 3, stride=1, padding=1,
										bias=False)),
					('bn3', nn.BatchNorm2d(self.inplanes)),
					('relu3', get_act(act, inplace=_act_inplace)),
					('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
						]
			strides = [1, 2, 2, 2]

		elif 'cifar' in dataset or 'svhn' in dataset:
			layer0_modules = [
					('conv1', nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=False)),
					('bn1', nn.BatchNorm2d(self.inplanes)),
					('relu1', get_act(act, inplace=_act_inplace)),
						]
			strides = [1, 2, 2, 1]
		else:
			raise NotImplementedError
		
		self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
		
		# For CIFAR/SVHN, layer1 - layer3, channels - [16, 32, 64]
		# For ImageNet, layer1 - layer4, channels - [64, 128, 256, 512]
		inplanes_origin = self.layer0_inplanes

		self.layer1 = self._make_layer(
			block,
			planes=inplanes_origin,
			blocks=layers[0],
			stride=strides[0],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=1,
			downsample_padding=0,
			act=act,
			dropblock_size=self.dropblock_size[0],
			gamma_scale=self.gamma_scales[0]
		)
		self.layer2 = self._make_layer(
			block,
			planes=inplanes_origin * 2,
			blocks=layers[1],
			stride=strides[1],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding,
			act='hardswish' if self.mix_act_block < 2 else act,
			dropblock_size=self.dropblock_size[1],
			gamma_scale=self.gamma_scales[1]
		)
		self.layer3 = self._make_layer(
			block,
			planes=inplanes_origin * 4,
			blocks=layers[2],
			stride=strides[2],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding,
			act='hardswish' if self.mix_act_block < 3 else act,
			dropblock_size=self.dropblock_size[2],
			gamma_scale=self.gamma_scales[2]
		)
		inplanes_now = inplanes_origin * 4
		
		self.layer4 = None
		if 'imagenet' in dataset:
			print('INFO:PyTorch: Using layer4 for ImageNet Training')
			self.layer4 = self._make_layer(
				block,
				planes=inplanes_origin * 8,
				blocks=layers[3],
				stride=strides[3],
				groups=self.groups,
				reduction=reduction,
				downsample_kernel_size=downsample_kernel_size,
				downsample_padding=downsample_padding,
				act='hardswish' if self.mix_act_block < 4 else act,
				dropblock_size=self.dropblock_size[3],
				gamma_scale=self.gamma_scales[3]
			)
			inplanes_now = inplanes_origin * 8
		
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		
		self.dropout = None
		if dropout_p is not None:
			dropout_p = dropout_p / (split_factor ** 0.5)
			# You can also use the below code.
			# dropout_p = dropout_p / split_factor
			print('INFO:PyTorch: Using dropout before last fc layer with ratio {}'.format(dropout_p))
			self.dropout = nn.Dropout(dropout_p, inplace=True)

		self.last_linear = nn.Linear(inplanes_now * block.expansion, num_classes)

		# initialize the parameters
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=1e-3)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
					downsample_kernel_size=1, downsample_padding=0,
					act='relu', dropblock_size=0, gamma_scale=1.0):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			if downsample_kernel_size != 1:
				# using conv3x3 to reserve information in SENet154
				downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
									kernel_size=downsample_kernel_size, stride=stride,
									padding=downsample_padding, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
				)
			else:
				# otherwise, using 2x2 average pooling to reserve information
				if stride == 1:
					downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
									kernel_size=1, stride=stride, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
					)
				else:
					downsample = nn.Sequential(
						# Ref:
						# Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
						# https://arxiv.org/abs/1812.01187
						# https://github.com/rwightman/pytorch-image-models/blob
						# /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
						# nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
						nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True,
										padding=0, count_include_pad=False),
						nn.Conv2d(self.inplanes, planes * block.expansion,
										kernel_size=1, stride=1, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
					)

		layers = []
		layers.append(block(self.inplanes, planes, groups, reduction, stride,
							downsample, act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale))
		self.inplanes = planes * block.expansion

		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups, reduction,
							act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale))

		return nn.Sequential(*layers)

	def features(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		if self.layer4 is not None:
			x = self.layer4(x)
		return x

	def logits(self, x):
		x = self.avg_pool(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.last_linear(x)
		return x

	def forward(self, x):
		x = self.features(x)
		x = self.logits(x)
		return x

def se_resnet164():
    model = SENet('se_resnet164', SEResNetBottleneck, [18, 18, 18, 18], groups=1, reduction=16,
					dropout_p=None, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=10, dataset="cifar10")
    return model

val_transform = transforms.Compose([
    #transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
    					(0.2023, 0.1994, 0.2010)),
])

model_config = OrderedDict([
    ('arch', 'shake_shake'),
    ('depth', 26),
    ('base_channels', 64),
    ('shake_forward', True),
    ('shake_backward', True),
    ('shake_image', True),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 10),
])

class EnsembleModel(torch.nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for path in model_paths:
            model = Network(model_config)
            model.load_state_dict(torch.load(path))
            self.models.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = [f"./shake_shake_64base_channels_noval_SE_red4d_ricap/fold_{i}_best.pth" for i in range(1)]
ensemble1 = EnsembleModel(model_paths).to(device)

class EnsembleModel2(torch.nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for path in model_paths:
            model = se_resnet164()
            model.load_state_dict(torch.load(path))
            self.models.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

model_paths = [f"./seresnet_COR/fold_{i}_best.pth" for i in range(4)]
ensemble2 = EnsembleModel2(model_paths).to(device)
model_paths = [f"./seresnet_noval/fold_{i}_best.pth" for i in range(1)]
ensemble3 = EnsembleModel2(model_paths).to(device)

class EnsembleModel3(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

ensemble4 = EnsembleModel3([ensemble1, ensemble2, ensemble3]).to(device)
ensemble4.eval()

class CIFAR10TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.data = []
        self.ids = []
        
        # Load test data (modify based on actual file structure)
        test_files = sorted(glob.glob(f"{test_dir}/*.pkl"))
        for f in test_files:
            with open(f, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            self.data.append(batch[b'data'].transpose(0, 3, 1, 2))
            # Assuming IDs are in order if not provided in files
            self.ids.extend(list(batch[b'ids']))
            
        self.data = np.vstack(self.data)  # Shape: (N, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # (3, 32, 32) uint8
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.ids[idx]

test_dir = "./deep-learning-spring-2025-project-1"
test_dataset = CIFAR10TestDataset(test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

all_ids = []
all_preds = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = ensemble4(images)
        _, preds = torch.max(outputs, 1)
        
        all_ids.extend(ids.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': all_ids,
    'Labels': all_preds
})

# Save to CSV
submission_path = "distillation_labels.csv"
submission_df.to_csv(submission_path, index=False)