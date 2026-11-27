---
title: "MobileNets: Code architecture"
parent: "038 MobileNets"
nav_order: 1
---
# Code Architecture — MobileNets Notebook

## Overview

The notebook implements depthwise separable convolutions from scratch in PyTorch. It builds a small MobileNet-style CNN and a standard-convolution network with the same layout, both at width multiplier α = 1.0. It counts their parameters and multiply-adds, times a forward pass, and trains both for 4 epochs on 10,000 CIFAR-10 training images. It runs on a GPU if there is one (the saved run used a Tesla T4).

## Section-by-Section Breakdown

### 1. Imports & Setup
- PyTorch, torchvision (CIFAR-10), matplotlib; nothing is installed
- Seed 42; `CUBLAS_WORKSPACE_CONFIG` set before importing torch, cuDNN deterministic, `torch.use_deterministic_algorithms(True, warn_only=True)`
- `ALPHA = 1.0`: one width multiplier for every comparison
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Depthwise Separable Convolution (from scratch)
- **`DepthwiseSeparableConv`**:
  - `nn.Conv2d(in_ch, in_ch, 3, groups=in_ch)`: depthwise, one filter per channel
  - `nn.BatchNorm2d(in_ch)` + `nn.ReLU`
  - `nn.Conv2d(in_ch, out_ch, 1)`: pointwise, combines the channels
  - `nn.BatchNorm2d(out_ch)` + `nn.ReLU`
- `groups=in_ch` is what makes the convolution depthwise in PyTorch

### 3. Standard Convolution Block (for comparison)
- **`StandardConv`**: one `nn.Conv2d(in_ch, out_ch, 3)` + BatchNorm + ReLU, with the same input and output channels

### 4. Small MobileNet Architecture
- **`SmallMobileNet(alpha)`**: a standard 3×3 conv (3→32 channels, stride 1, as the paper's first layer is a full convolution), then 4 depthwise separable blocks (32→64→128→256→256, stride 2 in blocks 2 and 4), global average pooling and a linear layer to 10 classes
- Every channel count is multiplied by α (at least 8 channels)

### 5. Standard Convolution Equivalent
- **`SmallStandardNet(alpha)`**: the same layout with every depthwise separable block replaced by `StandardConv`; the small-scale version of the paper's Table 4 comparison

### 6. Parameter & FLOP Comparison
- **`count_parameters(model)`**: trainable parameters
- **`estimate_macs(model)`**: forward hooks on every `nn.Conv2d` add up kernel_h·kernel_w·(C_in/groups)·C_out·H_out·W_out for one 32×32 image (convolutions only)
- Prints both networks at α = 1.0 and the MobileNet at α = 1, 0.75, 0.5 and 0.25

### 7. Inference Speed Benchmark
- **`benchmark_inference(model, input_shape=(1, 3, 32, 32), n_warmup=10, n_runs=100)`**: warm-up, then the mean time of 100 forward passes of one image, with CUDA synchronisation
- Prints both times and StandardNet time / MobileNet time (above 1 means the MobileNet is faster)

### 8. Training on CIFAR-10
- Random crop (padding 4) and horizontal flip for training, normalisation for both; the dataset is downloaded quietly
- Only the first 10,000 of the 50,000 training images; all 10,000 test images
- **`train_model`**: Adam (learning rate 1e-3) with cosine decay, batch size 128, **4 epochs**; test accuracy after each epoch; returns the history and the best test accuracy
- 8.1 trains `SmallMobileNet(alpha=ALPHA)` and 8.2 trains `SmallStandardNet(alpha=ALPHA)`, each from seed 42

### 9. Results Comparison
- One figure with three panels for the two trained models: parameter count, the inference times measured in Section 7 (lower is faster), and test accuracy per epoch

### 10. Final Numbers
- A table for both networks at α = 1.0: parameters, estimated MACs, best test accuracy, inference time and the ratios, plus the accuracy difference

### 11. Summary
- `### What the run above shows` with the saved run's numbers

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `DepthwiseSeparableConv` | From-scratch depthwise + pointwise block with BN and ReLU |
| `StandardConv` | Standard 3×3 conv block for comparison |
| `SmallMobileNet` | Small MobileNet for CIFAR-10, with width multiplier |
| `SmallStandardNet` | Standard-conv network with the same layout |
| `count_parameters()` | Counts trainable parameters |
| `estimate_macs()` | Estimates multiply-adds of the conv layers with forward hooks |
| `benchmark_inference()` | Times forward passes |
| `train_model()` | Training loop with loss and test accuracy per epoch |

## Data Flow / Shapes (CIFAR-10, α = 1.0)

```
Input:  [B, 3, 32, 32]
→ Conv 3×3, stride 1:   [B, 32, 32, 32]
→ DW-Sep block 1 (s=1): [B, 64, 32, 32]
→ DW-Sep block 2 (s=2): [B, 128, 16, 16]
→ DW-Sep block 3 (s=1): [B, 256, 16, 16]
→ DW-Sep block 4 (s=2): [B, 256, 8, 8]
→ Global avg pool:      [B, 256, 1, 1]
→ Flatten + linear:     [B, 10]
```

## Deliberate Simplifications vs Full Paper

1. **Scale:** a first convolution plus 4 depthwise separable blocks (up to 256 channels) on 32×32 images, instead of 13 blocks (up to 1024 channels) on 224×224 ImageNet images.
2. **Training:** Adam for 4 epochs on 10,000 CIFAR-10 images, instead of RMSprop with asynchronous gradient descent on ImageNet; no special weight-decay setting for the depthwise filters. Both networks end at about 50–55% test accuracy, far from fully trained.
3. **Multipliers:** the width multiplier is shown with parameter and MAC counts only; there is no resolution multiplier and no sweep over the paper's 16 α×ρ combinations.
4. **Speed:** timed on a GPU with one image per batch, which is not the mobile hardware the paper targets.
5. **No applications:** no fine-grained recognition, geolocalization, face attributes, detection or face embeddings.
