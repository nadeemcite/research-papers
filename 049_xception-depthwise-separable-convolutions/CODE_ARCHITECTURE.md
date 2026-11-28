---
title: "Xception: Code architecture"
parent: "049 Xception"
nav_order: 1
---
# Code Architecture — Xception Notebook

## Overview

The notebook implements the Xception module (depthwise separable convolutions with a residual connection) from scratch in PyTorch. It builds a small Xception-style network for CIFAR-10 and an Inception-style network with the same layout and channel counts. The Inception-style network has about 3.7 times as many parameters (363,720 vs 97,422), so the two are not matched in size. Both are trained the same way for 3 epochs on CIFAR-10. It runs on a GPU if there is one (the saved run used a Tesla T4).

## Section-by-Section Breakdown

### 1. Imports & Setup
- PyTorch, torchvision (CIFAR-10), NumPy, matplotlib; nothing is installed
- Seed 42; `CUBLAS_WORKSPACE_CONFIG` set before importing torch, cuDNN deterministic, `torch.use_deterministic_algorithms(True, warn_only=True)`
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Separable Convolution Block (from scratch)
- **`SeparableConv(in_channels, out_channels)`**:
  - `nn.Conv2d(in, in, 3, padding=1, groups=in, bias=False)`: depthwise, one 3×3 filter per channel
  - `nn.BatchNorm2d(in)`, then **no activation** (the paper's Section 4.7 finding)
  - `nn.Conv2d(in, out, 1, bias=False)`: pointwise, mixes the channels
  - `nn.BatchNorm2d(out)` + `nn.ReLU`

### 3. Xception Module (with residual connection)
- **`XceptionModule(in_channels, out_channels, stride=1)`**: `SeparableConv(in→out)`, then 2×2 max pooling if `stride=2`, then `SeparableConv(out→out)`
- Residual: identity if the shape is unchanged, otherwise a strided 1×1 convolution + BatchNorm; added to the output
- Every module has a residual connection, including the last (the paper leaves it out of the first and last modules)

### 4. Inception-Style Block (for comparison)
- **`InceptionBlock(in_channels, out_channels, stride=1)`**: three parallel branches over all input channels, a 1×1, a 3×3 and a 5×5 convolution, each producing about a third of the output channels
- Concatenate → BatchNorm → ReLU → 2×2 max pooling if `stride=2`, plus the same kind of residual connection
- The 3×3 and 5×5 branches are full convolutions, which is why this network is larger

### 5. Small Xception Architecture
- **`SmallXception(num_classes=10)`**:
  - Entry flow: 3×3 conv (3→32) + BN + ReLU, then `XceptionModule(32, 48, stride=2)` and `XceptionModule(48, 96, stride=2)`
  - Middle flow: one `XceptionModule(96, 96)` (the paper repeats its middle flow 8 times)
  - Exit flow: `XceptionModule(96, 128, stride=2)`, then `XceptionModule(128, 10)`
  - Global average pooling of the 10 channels gives the class scores; there is no fully connected layer

### 6. Small Inception-Style Architecture
- **`SmallInceptionNet(num_classes=10)`**: the same layout with `InceptionBlock` in place of every `XceptionModule`

### 7. Parameter Counting
- **`count_parameters(model)`**: trainable parameters. SmallXception 97,422, SmallInceptionNet 363,720 (3.73×)
- **`param_breakdown(model, name)`**: convolution, BatchNorm and other parameters by name

### 8. Data Loading (CIFAR-10)
- Random crop (padding 4) and horizontal flip for training, normalisation for both; the dataset is downloaded quietly
- All 50,000 training and 10,000 test images; batch size 128, seeded shuffling

### 9. Training
- **`train_model(model, trainloader, testloader, epochs=3, lr=0.01, name)`**: SGD (momentum 0.9, weight decay 5e-4), cosine annealing, cross-entropy; prints loss, training accuracy and test accuracy after each epoch and returns the history
- Trains SmallXception, then SmallInceptionNet, both for **3 epochs**

### 10. Results Comparison
- One figure with three panels: parameter count, test accuracy per epoch, training loss per epoch
- A table with parameters and final training and test accuracy, the test-accuracy difference (SmallXception minus SmallInceptionNet) and the parameter difference

### 11. What the Paper Found
- The paper's findings with section, figure and table references, labelled as not reproduced here

### 12. Summary
- `### What the run above shows` with the saved run's numbers

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `SeparableConv` | Depthwise 3×3 → BN → pointwise 1×1 → BN → ReLU; no activation in between |
| `XceptionModule` | Two `SeparableConv` blocks, optional max pooling, residual (identity or 1×1 projection) |
| `InceptionBlock` | Parallel 1×1 / 3×3 / 5×5 convolutions → concat → BN → ReLU, plus residual |
| `SmallXception` | Entry / middle / exit flow of `XceptionModule`s, global average pooling |
| `SmallInceptionNet` | The same layout with `InceptionBlock`s |
| `count_parameters()` | Counts trainable parameters |
| `param_breakdown()` | Splits parameters into convolution, BatchNorm and other |
| `train_model()` | Training loop with loss and accuracy per epoch |

## Data Flow / Shapes (CIFAR-10, batch B)

```
Input:                          [B, 3, 32, 32]
→ Conv 3×3 + BN + ReLU:         [B, 32, 32, 32]
→ XceptionModule 32→48, s=2:    [B, 48, 16, 16]
→ XceptionModule 48→96, s=2:    [B, 96, 8, 8]
→ XceptionModule 96→96:         [B, 96, 8, 8]
→ XceptionModule 96→128, s=2:   [B, 128, 4, 4]
→ XceptionModule 128→10:        [B, 10, 4, 4]
→ Global avg pool + flatten:    [B, 10]
```

## Deliberate Simplifications vs Full Paper

1. **Scale:** a first convolution and 5 modules with up to 128 channels on 32×32 images, instead of 14 modules and 36 convolutional layers on ImageNet-size images (Section 3, Figure 5).
2. **Baseline:** a small Inception-style network with the same layout, not Inception V3, and about 3.7 times larger than the small Xception; the paper compares two networks of nearly the same size (Table 3).
3. **Data and training:** CIFAR-10 for 3 epochs with SGD, learning rate 0.01, cosine annealing and weight decay 5e-4, instead of ImageNet and JFT with the paper's settings (Sections 4.2 and 4.3); no dropout and no Polyak averaging. Both networks are far from fully trained.
4. **Residuals:** every module has a residual connection; the paper's first and last modules do not.
5. **No ablations:** no runs without residual connections (Figure 9) or with an intermediate ReLU or ELU (Figure 10); the notebook only builds the paper's best setting, with no intermediate activation.
