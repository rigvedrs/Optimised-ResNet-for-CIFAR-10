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

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True,
                   help='Name of the experiment directory to create')
args = parser.parse_args()

experiment_dir = args.experiment_name
os.makedirs(experiment_dir, exist_ok=True)

class CIFAR10CombinedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
            
        self.data = np.vstack(self.data)  # (N, 3, 32, 32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]  # (3, 32, 32) uint8
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

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

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
def get_cosine_annealing_scheduler(optimizer, optim_config):
    total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            optim_config['lr_min'] / optim_config['base_lr']))

    return scheduler

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
full_dataset = CIFAR10CombinedDataset(root_dir, transform=train_transform)

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

optim_config = OrderedDict([
    ('epochs', 1800),
    ('batch_size', 128),
    ('base_lr', 0.2),
    ('weight_decay', 1e-4),
    ('momentum', 0.9),
    ('nesterov', True),
    ('lr_min', 0.),
])

train_collator = RICAPCollator() #ricap#
train_loader = DataLoader(full_dataset, batch_size=optim_config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_collator) #ricap#
#train_loader = DataLoader(CutMix(full_dataset, 10,
#               beta=1.0, prob=0.5, num_mix=2), batch_size=optim_config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

optim_config['steps_per_epoch'] = len(train_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(model_config).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)

criterion = RICAPLoss(reduction='mean') #ricap#
#criterion = CutMixCrossEntropyLoss(True).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = get_cosine_annealing_scheduler(optimizer, optim_config)

for epoch in range(optim_config['epochs']):
    epoch_start = time.time()

    # Training
    model.train()
    train_start = time.time()
    train_samples = 0
    #for inputs, labels in train_loader:
    for inputs, targets in train_loader:#ricap#
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
        scheduler.step()

    train_time = time.time() - train_start
    train_fps = train_samples / train_time

    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:04d} | "
            f"Train FPS: {train_fps:.1f} | "
            f"Epoch Time: {epoch_time:.1f}s")

torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) 
    else model.state_dict(), os.path.join(experiment_dir, f"fold_0_best.pth"))

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

model_paths = [os.path.join(experiment_dir, f"fold_{i}_best.pth") for i in range(1)]
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
