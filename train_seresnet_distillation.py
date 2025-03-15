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
from sklearn.model_selection import KFold
import time
import argparse
import os
from typing import List, Tuple
from cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from collections import OrderedDict
import math
import pandas as pd

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True,
                   help='Name of the experiment directory to create')
args = parser.parse_args()

experiment_dir = args.experiment_name
os.makedirs(experiment_dir, exist_ok=True)

def get_distil_label(ID, distillation_labels):
    label = distillation_labels.loc[distillation_labels["ID"] == ID, "Labels"].tolist()
    assert len(label) == 1
    return label[0]

class CIFAR10CombinedDataset(Dataset):
    def __init__(self, root_dir, test_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load all available data
        all_files = glob.glob(f"{root_dir}/*_batch*")  # Matches data_batch* and test_batch
        for f in all_files:
            with open(f, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            self.data.append(batch[b'data'].reshape(-1, 3, 32, 32))
            self.labels.extend(batch[b'labels'])

        test_files = sorted(glob.glob(f"{test_dir}/*.pkl"))
        distillation_labels = pd.read_csv("./distillation_labels.csv")
        for f in test_files:
            with open(f, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            self.data.append(batch[b'data'].transpose(0, 3, 1, 2))
            labels = [get_distil_label(ID, distillation_labels) for ID in list(batch[b'ids'])]
            self.labels.extend(labels)
            
        self.data = np.vstack(self.data)  # (N, 3, 32, 32)
        print(self.data.shape)
        print(len(self.labels))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]  # (3, 32, 32) uint8
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        #image = torch.tensor(image, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

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

class cos_lr_scheduler(object):
	def __init__(self,  init_lr=0.1,
						num_epochs=1800,
						iters_per_epoch=300,
						slow_start_epochs=60,
						slow_start_lr=5e-3,
						end_lr=1e-4,
						multiplier=1.0,
						):

		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.end_lr = end_lr

		self.num_epochs = num_epochs
		self.iters_per_epoch = iters_per_epoch

		self.slow_start_iters = slow_start_epochs * iters_per_epoch
		self.slow_start_lr = slow_start_lr
		self.total_iters = (num_epochs - slow_start_epochs) * iters_per_epoch

		self.multiplier = multiplier

		# log info
		print("INFO:PyTorch: Using cos learning rate scheduler with"
				" warm-up epochs of {}!".format(slow_start_epochs))

	def __call__(self, optimizer, i, epoch):
		"""call method"""
		T = epoch * self.iters_per_epoch + i

		if self.slow_start_iters > 0 and T <= self.slow_start_iters:
			# slow start strategy -- warm up
			# see 	https://arxiv.org/pdf/1812.01187.pdf
			# 	Bag of Tricks for Image Classification with Convolutional Neural Networks
			# for details.
			lr = (1.0 * T / self.slow_start_iters) * (self.init_lr - self.slow_start_lr)
			lr = min(lr + self.slow_start_lr, self.init_lr)
		
		else:
			T = T - self.slow_start_iters
			lr = 0.5 * self.init_lr * (1.0 + math.cos(1.0 * T / self.total_iters * math.pi))

		lr = max(lr, self.end_lr)
		self.now_lr = lr

		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier

def se_resnet164():
    model = SENet('se_resnet164', SEResNetBottleneck, [18, 18, 18, 18], groups=1, reduction=16,
					dropout_p=None, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=10, dataset="cifar10")
    return model

class Cutout:
    def __init__(self):
        self.p = 1.
        self.mask_size = 16
        self.cutout_inside = False
        self.mask_color = 0

        self.mask_size_half = self.mask_size // 2
        self.offset = 1 if self.mask_size % 2 == 0 else 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if self.cutout_inside:
            cxmin = self.mask_size_half
            cxmax = w + self.offset - self.mask_size_half
            cymin = self.mask_size_half
            cymax = h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image

def ricap(
    batch: Tuple[torch.Tensor, torch.Tensor], beta: float
) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
    data, targets = batch
    image_h, image_w = data.shape[2:]
    ratio = np.random.beta(beta, beta, size=2)
    w0, h0 = np.round(np.array([image_w, image_h]) * ratio).astype(np.int32)
    w1, h1 = image_w - w0, image_h - h0
    ws = [w0, w1, w0, w1]
    hs = [h0, h0, h1, h1]

    patches = []
    labels = []
    label_weights = []
    for w, h in zip(ws, hs):
        indices = torch.randperm(data.size(0))
        x0 = np.random.randint(0, image_w - w + 1)
        y0 = np.random.randint(0, image_h - h + 1)
        patches.append(data[indices, :, y0:y0 + h, x0:x0 + w])
        labels.append(targets[indices])
        label_weights.append(h * w / (image_h * image_w))

    data = torch.cat(
        [torch.cat(patches[:2], dim=3),
         torch.cat(patches[2:], dim=3)], dim=2)
    targets = (labels, label_weights)

    return data, targets


class RICAPCollator:
    def __init__(self):
        self.beta = 0.3

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = ricap(batch, self.beta)
        return batch

class RICAPLoss:
    def __init__(self, reduction: str):
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[List[torch.Tensor], List[float]]) -> torch.Tensor:
        target_list, weights = targets
        return sum([
            weight * self.loss_func(predictions, targets)
            for targets, weight in zip(target_list, weights)
        ])

"""train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])"""
train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
    					(0.2023, 0.1994, 0.2010)),
	transforms.RandomErasing(p=0.5,
							scale=(0.125, 0.2),
							ratio=(0.99, 1.0),
							value=0, inplace=False),
    #Cutout()
])

val_transform = transforms.Compose([
    #transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
    					(0.2023, 0.1994, 0.2010)),
])

root_dir = "./deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py"
test_dir = "./deep-learning-spring-2025-project-1"
full_dataset = CIFAR10CombinedDataset(root_dir, test_dir)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Training loop for each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    if fold >= 2:
        break
    print(f"\n=== Training Fold {fold+1}/5 ===")
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_subset.dataset = CIFAR10CombinedDataset(root_dir, test_dir)
    val_subset.dataset = CIFAR10CombinedDataset(root_dir, test_dir)

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_collator = RICAPCollator() #ricap#

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_collator,) #ricap#
    #train_loader = DataLoader(CutMix(train_subset, 10,
    #           beta=1.0, prob=0.5, num_mix=2), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = se_resnet164().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    criterion = RICAPLoss(reduction='mean') #nn.CrossEntropyLoss()#ricap#
    #criterion = CutMixCrossEntropyLoss(True).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    max_epochs = 1800
    scheduler = cos_lr_scheduler(num_epochs=max_epochs, iters_per_epoch=len(train_loader))

    best_acc = 0.0
    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_start = time.time()
        train_samples = 0
        i = 0
        #for inputs, labels in train_loader:
        for inputs, targets in train_loader: #ricap#
            scheduler(optimizer, i, epoch)
            inputs = inputs.to(device)
            #labels = labels.to(device)
            labels, weights = targets #ricap#
            labels = [label.to(device) for label in labels] #ricap#
            labels = (labels, weights) #ricap#

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_samples += inputs.size(0)
            i += 1

        train_time = time.time() - train_start
        train_fps = train_samples / train_time

        # Validation
        model.eval()
        total, correct = 0, 0
        val_start = time.time()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_time = time.time() - val_start
        val_fps = len(val_loader.dataset) / val_time
        acc = 100. * correct / total

        epoch_time = time.time() - epoch_start
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) 
                      else model.state_dict(), os.path.join(experiment_dir, f"fold_{fold}_best.pth"))

        print(f"Epoch {epoch+1:04d} | "
              f"lr {optimizer.param_groups[0]['lr']:.4f} |"
              f"Train FPS: {train_fps:.1f} | "
              f"Val FPS: {val_fps:.1f} | "
              f"Val Acc: {acc:.2f}% | "
              f"Epoch Time: {epoch_time:.1f}s")
    
    fold_results.append(best_acc)
    print(f"Fold {fold+1} Best Acc: {best_acc:.2f}%")

print(f"\nAverage Validation Accuracy: {np.mean(fold_results):.2f}%")

class EnsembleModel(torch.nn.Module):
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

model_paths = [os.path.join(experiment_dir, f"fold_{i}_best.pth") for i in range(2)]
ensemble = EnsembleModel(model_paths).to(device)
ensemble.eval()

import pandas as pd
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
            self.ids.extend(list(batch[b'ids']))
            
        self.data = np.vstack(self.data)  # Shape: (N, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # (3, 32, 32) uint8
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        #image = torch.tensor(image, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)
            
        return image, self.ids[idx]

# Initialize test dataset and loader
test_dir = "./deep-learning-spring-2025-project-1"
test_dataset = CIFAR10TestDataset(test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Inference
all_ids = []
all_preds = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = ensemble(images)
        _, preds = torch.max(outputs, 1)
        
        all_ids.extend(ids.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': all_ids,
    'Labels': all_preds
})

# Save to CSV
submission_path = os.path.join(experiment_dir, 'submission.csv')
submission_df.to_csv(submission_path, index=False)
