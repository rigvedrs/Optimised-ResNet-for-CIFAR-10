# ResNet Optimization for CIFAR-10 Classification 🔬

![CIFAR-10 Classification](https://img.shields.io/badge/Task-Image%20Classification-blue)
![Deep Learning](https://img.shields.io/badge/Field-Deep%20Learning-brightgreen)
![ResNet](https://img.shields.io/badge/Architecture-ResNet-orange)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-90.757%25-success)

## 📋 Overview

This project focuses on optimizing ResNet architectures for CIFAR-10 image classification while adhering to a strict parameter budget of 5 million. We systematically explore various architectural modifications, training strategies, and regularization techniques to maximize classification accuracy without using external data such as pre-training on ImageNet.

Our best-performing model achieves **90.757%** test accuracy on the final leaderboard, demonstrating significant improvements over baseline approaches through the application of advanced techniques.

## .👨‍🔬 Authors

- Joseph Amigo (ja5009@nyu.edu)
- Rigved Shirvalkar (rss9347@nyu.edu)
- Om Thakur (ot2131@nyu.edu)

## 🌟 Key Features

- ⚙️ Parameter-efficient ResNet architectures under 5M parameters
- 🔄 Advanced data augmentation strategies
- 🛡️ State-of-the-art regularization techniques
- 🧠 Teacher-student knowledge distillation
- 📊 Systematic experimental methodology

### 📌 Key Constraints:

- **No external data usage** (e.g., no ImageNet pre-training, no fine-tuning outside CIFAR-10)
- **Total parameter count ≤ 5M** to maintain model efficiency

We conducted extensive experiments on **training strategies, data augmentation techniques, and regularization methods** to enhance the performance of the model under these constraints. These optimizations led to significant improvements in accuracy and generalization capability.



## 🛠️ Methodology

Our approach involved iterative improvements through the following stages:

### 1. 🏁 Baseline Models
- Pruned ResNet-18 (removed fourth stage of layer blocks to stay under parameter constraints)
- Initial accuracy: 82.88%
- Implementation details:
  - Standard SGD optimizer with momentum 0.9
  - Initial learning rate: 0.1
  - Batch size: 128
  - Basic preprocessing only

### 2. ⚡ Optimization Techniques
- 📉 Cosine annealing learning rate schedule: `lr = η_min + 0.5(η_max - η_min)(1 + cos(epoch / total_epochs * π))`
- 🚀 Nesterov momentum (acceleration factor: 0.9)
- ⚖️ Weight decay (5e-4) for implicit regularization
- 📈 Improved accuracy: 86.25%
- Implementation details:
  - Warm-up period of 5 epochs
  - Higher initial learning rate (0.2)
  - Improved initialization techniques

### 3. 🔄 Data Augmentation
- 📌 Standard techniques:
  - Random cropping (32x32 with 4px padding)
  - Horizontal flipping (p=0.5)
- 🔍 Advanced techniques:
  - AutoAugment (CIFAR-10 policy)
  - Random erasing (p=0.25, area ratio=0.02-0.4)
  - Cutout (16x16 patches)
- 🧪 Experimental techniques:
  - TrivialAugment (unexpectedly didn't improve results)
  - AugMix (mixing coefficient=0.4, severity=3)
- 📈 Improved accuracy: 87.23%
- Implementation details:
  - Normalized using CIFAR-10 mean (0.4914, 0.4822, 0.4465) and std (0.247, 0.243, 0.261)
  - Sequential application of augmentations

### 4. 🏗️ Architectural Enhancements
- 🔀 Shake-Shake regularization:
  - Two parallel residual branches
  - Different scaling factors for forward/backward passes
  - Introduces beneficial noise during training
- 🔌 Increased base channels from 32 to 40
- 🔍 Squeeze-and-Excitation blocks:
  - Channel attention mechanism
  - Reduction ratio: 16
  - Applied after each residual block
- 🧩 RICAP (Random Image Cropping and Patching):
  - Beta distribution parameters: α=β=0.3
  - Multi-target training with weighted loss
- 📈 Improved accuracy: 89.36%
- Implementation details:
  - SE block architecture: Global Average Pooling → FC(channels/r) → ReLU → FC(channels) → Sigmoid

### 5. 📏 Scaling Depth
- Shifted to ResNet-164 architecture (inspired by the original ResNet paper)
- Bottleneck blocks with 3 layers each
- 54 layers per stage (3 stages total)
- 2.49M parameters (well under budget)
- Implementation details:
  - Bottleneck structure: 1x1 conv (reduce) → 3x3 conv → 1x1 conv (expand)
  - Expansion factor: 4
  - Added SE modules to each bottleneck
- 📈 Improved accuracy: 89.68%

### 6. 🤝 Ensembling & Distillation
- 👥 Ensemble of models:
  - Shake-Shake 64 with SE trained with RICAP + AutoAugment
  - 5x SEResNet-164 trained on different data folds
  - SEResNet-164 trained on all data
- 🎓 Teacher-student knowledge distillation:
  - Temperature parameter: T=4
  - α=0.9 balancing factor between soft and hard targets
  - KL divergence for soft targets
- 📊 Training details:
  - 300 epochs for teacher models
  - 200 epochs for student models
  - SGD with Nesterov momentum and cosine annealing
- 📈 Final accuracy: 90.76%

## 📊 Models

| Model | Techniques | Parameters | Accuracy |
|-------|------------|------------|----------|
| ResNet-18 (pruned) | Basic training | <5M | 82.88% |
| ResNet-18 (pruned) | Cosine LR + Basic Aug | <5M | 86.25% |
| ResNet-18 (pruned) | Cosine LR + Advanced Aug | <5M | 87.23% |
| Shake-Shake (32) | Cosine LR + ARH + Cutout | <5M | 87.73% |
| Shake-Shake (40) | Cosine LR + ARH + Cutout | <5M | 88.39% |
| Shake-Shake (40) | Cosine LR + SE + RICAP | <5M | 89.36% |
| SEResNet-164 | Cosine LR + Advanced Aug + RICAP | <5M | 89.68% |
| SEResNet-164 (Ensemble) | All techniques | <5M | 89.76% |
| SEResNet-164 (Distilled) | Teacher-Student Pipeline | <5M | 90.76% |

*ARH = AutoAugment + RandomCrop + HorizontalFlip*


## 💡 Key Findings

1. **Parameter Efficiency**: ResNet-164 provides excellent performance while staying well under parameter constraints
2. **Regularization Impact**: Shake-Shake and RICAP significantly improve generalization
3. **Augmentation Strategy**: Carefully chosen augmentation combinations outperform generic approaches
4. **Knowledge Distillation**: Ensemble teachers provide valuable guidance for single student models
5. **Training Duration**: Extended training periods with appropriate learning rate scheduling yield significant gains

## Future Work

As noted in our analysis, further improvements could be achieved by:
- Implementing PyramidNet+ShakeDrop trained with AutoAugment (shown to generalize best according to Cubuk et al. 2019)
- Exploring more advanced distillation techniques
- Investigating recently published architectural innovations that maintain parameter efficiency



## 📁 Project Structure

```
./
├── cutmix/                  # Implementation of CutMix augmentation technique
├── deep-learning-spring-2025-project-1/  # Core project files
├── weights/                 # Saved model weights
├── distillation_labels.csv  # Pre-computed soft labels for distillation
├── train_seresnet.py        # SEResNet-164 training script with validation
├── train_seresnet_noval.py  # SEResNet-164 training script without validation
├── train_seresnet_distillation.py        # Distillation training with validation
├── train_seresnet_distillation_noval.py  # Distillation training without validation
└── train_shake_shake_unfolded_noval.py   # Shake-Shake training script
```
