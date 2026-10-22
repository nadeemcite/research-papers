---
title: "VGGNet: Code architecture"
parent: "004 VGGNet"
nav_order: 1
---
# CODE_ARCHITECTURE.md — VGGNet From Scratch

## What the notebook builds

A self-contained PyTorch notebook that builds a small VGG-style network, trains it on CIFAR-10, and produces accuracy/loss plots plus visualizations of learned filters. The network is much smaller than the original VGG-16/VGG-19 because CIFAR-10 has 32×32 images instead of 224×224, but it keeps the same architectural principles: stacked 3×3 convolutions, doubling channels after each max-pool, and a small fully connected classifier.

## Notebook sections (in order)

### 1. Setup and reproducibility
- Imports: `torch`, `torchvision`, `torch.nn`, `torch.optim`, `matplotlib`, `numpy`.
- `seed_everything(42)` sets the random seeds and makes cuDNN deterministic.
- Detect GPU/CUDA and set `device`.

### 2. VGG block and network builder
- `make_vgg_block(in_channels, out_channels, num_convs=2, use_1x1=False)` stacks `num_convs` 3×3 convs (padding 1), each followed by `BatchNorm2d` and `ReLU`, then `MaxPool2d(2, 2)`. `use_1x1` can add a 1×1 conv (like configuration C); it is not used here.
- `class VGG(nn.Module)` takes the conv `features` and adds the classifier: FC-512 → ReLU → Dropout(0.5) → FC-512 → ReLU → Dropout(0.5) → FC-10. `_initialize_weights()` uses Kaiming-normal init for the conv layers and sets BatchNorm to weight 1, bias 0. The `Linear` layers keep PyTorch's default init: Kaiming `fan_out` init on the final `Linear(512, 10)` produced huge logits and left the network stuck at chance.
- `cfg_cifar_d` follows VGG-D: 5 blocks with 2, 2, 3, 3, 3 convs and 64, 128, 256, 512, 512 channels (13 conv + 3 FC = 16 weight layers). `build_vgg_from_cfg(cfg)` builds the conv part. Total: about 15.25M parameters.
- A dummy `(2, 3, 32, 32)` batch checks the output shape `(2, 10)`.

### 3. CIFAR-10 data loaders
- `get_cifar10_loaders(batch_size=128, num_workers=2)` uses `torchvision.datasets.CIFAR10` (read from the attached Kaggle dataset `pankrzysiu/cifar10-python` when it is present, otherwise downloaded) with normalization `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`.
- Training augmentation: random crop (padding 4) + horizontal flip. There is no validation split: only `train_loader` (391 batches) and `test_loader` (79 batches).

### 4. Training loop
- Loss: `CrossEntropyLoss`.
- Optimizer: SGD, learning rate 0.01, momentum 0.9, weight decay 5e-4, with `CosineAnnealingLR` over the same number of epochs (`T_max` = 20 on a GPU, 3 on a CPU).
- `train_epoch` and `evaluate` return average loss and accuracy. The loop runs 20 epochs on a GPU (3 on a CPU, where an epoch is much slower), records train/test loss and accuracy, and prints the time of each epoch.

### 5. Training curves
- Matplotlib plots of train vs. test loss and train vs. test accuracy per epoch.

### 6. Filter visualization
- `plot_filters(weights)` shows the 64 filters of the first conv layer (3×3, averaged over the RGB channels and shown in grayscale).

### 7. Parameter count: VGG-style vs. plain CNN
- `class Plain5x5CNN` has five 5×5 conv layers (one per stage) and a single FC-10 layer. The cell prints both parameter counts and their ratio.

### 8. Receptive-field / parameter-count demo
- Compares three stacked 3×3 convs with one 7×7 conv for C = 64 channels and prints the parameter savings and the number of ReLUs.

### 9. Summary
- A markdown summary of what the notebook built and what the saved run shows.

## Key functions / classes

- `make_vgg_block(in_channels, out_channels, num_convs=2, use_1x1=False)` — returns a `nn.Sequential` of `Conv2d → BatchNorm2d → ReLU` repeated `num_convs` times, ending with `MaxPool2d`.
- `class VGG(nn.Module)` — conv features plus the 3-layer FC classifier; defines `forward(x)`.
- `build_vgg_from_cfg(cfg)` — builds the conv features from a list of `(in_c, out_c, n_convs, use1x1)` tuples.
- `get_cifar10_loaders(batch_size, num_workers)` — returns the train and test loaders.
- `train_epoch(model, loader, criterion, optimizer, device)` — one training epoch; returns average loss and accuracy.
- `evaluate(model, loader, criterion, device)` — test evaluation; returns average loss and accuracy.
- `plot_filters(weights, title, ncols=8)` — plots first-layer filters.
- `class Plain5x5CNN(nn.Module)` — the comparison network with 5×5 filters.

## Data flow and shapes

- Input batch: `(B, 3, 32, 32)` for CIFAR-10.
- Block 1: `(B, 64, 32, 32)` → pool → `(B, 64, 16, 16)`.
- Block 2: `(B, 128, 16, 16)` → pool → `(B, 128, 8, 8)`.
- Block 3: `(B, 256, 8, 8)` → pool → `(B, 256, 4, 4)`.
- Block 4: `(B, 512, 4, 4)` → pool → `(B, 512, 2, 2)`.
- Block 5: `(B, 512, 2, 2)` → pool → `(B, 512, 1, 1)`.
- Flatten: `(B, 512)` → FC-512 → FC-512 → `(B, 10)`.

## Deliberate simplifications vs. the full paper

- **CIFAR-10 instead of ImageNet**: 32×32 colour images, 10 classes. The original paper used 224×224 ImageNet crops with 1,000 classes.
- **Smaller classifier head**: FC-512 instead of the original FC-4096.
- **Single model, not an ensemble**: The paper's ILSVRC submission was a 7-net ensemble, and its best result used 2 nets with multi-crop and dense evaluation. The notebook trains one network and tests on the plain 32×32 test images.
- **Short schedule**: 20 epochs with cosine annealing instead of 74 epochs with step decay (3 epochs on a CPU).
- **BatchNorm added**: The original paper predates batch normalization (Ioffe & Szegedy, 2015).

The notebook therefore teaches the VGG design recipe — deep stacks of 3×3 convs, channel doubling, small classifier — rather than reproducing the full ILSVRC 2014 submission. In the saved GPU run the network reaches 88.57% test accuracy; see the README.
