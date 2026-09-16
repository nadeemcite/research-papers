# CODE_ARCHITECTURE.md — VGGNet From Scratch

## What the notebook builds

A self-contained PyTorch notebook that builds a small VGG-style network, trains it on CIFAR-10, and produces accuracy/loss plots plus visualizations of learned filters. The network is much smaller than the original VGG-16/VGG-19 because CIFAR-10 has 32×32 images instead of 224×224, but it keeps the same architectural principles: stacked 3×3 convolutions, doubling channels after each max-pool, and a small fully connected classifier.

## Notebook sections (in order)

### 1. Setup and reproducibility
- Imports: `torch`, `torchvision`, `torch.nn`, `torch.optim`, `matplotlib`, `numpy`.
- Set random seed for reproducibility.
- Detect GPU/CUDA and set `device`.

### 2. CIFAR-10 data loaders
- `torchvision.datasets.CIFAR10` with standard normalization (`mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`).
- Train/validation split and a `test_loader`.
- Data augmentation: random crop + horizontal flip for training.

### 3. VGG block builder
Helper `make_vgg_block(in_channels, out_channels, num_convs, use_batchnorm=True)`:
- Stacks `num_convs` 3×3 convolutions with same padding.
- Optional `BatchNorm2d` after each conv for training stability.
- `ReLU` after each conv.
- Final `MaxPool2d(2, 2)`.

### 4. Full VGG-style network
Class `VGGNetSmall`:
- 4 blocks:
  - Block 1: 64 channels, 2 convs
  - Block 2: 128 channels, 2 convs
  - Block 3: 256 channels, 3 convs
  - Block 4: 512 channels, 3 convs
- Global average pooling or flatten, then classifier:
  - FC-512 → ReLU → Dropout(0.5) → FC-10
- Reason: on 32×32 images the original 4096-unit fully connected layers would over-parameterize and overfit.

### 5. Training loop
- Loss: `CrossEntropyLoss`.
- Optimizer: SGD with momentum `0.9`, weight decay `5e-4`, cosine or step LR schedule.
- Loop records train/validation loss and accuracy per epoch.
- Early stopping or fixed epochs (e.g., 30–50 for a notebook demo).

### 6. Evaluation and plots
- Test-set accuracy.
- Matplotlib line plots:
  - Train vs. validation loss over epochs.
  - Train vs. validation accuracy over epochs.

### 7. Filter visualization
- Extract weights from the first convolutional layer (`conv1_1`).
- Plot the first 64 filters as 3×3 RGB grids.

### 8. Receptive-field / parameter-count demo
- A short cell that compares:
  - one 7×7 convolution vs. three stacked 3×3 convolutions for the same receptive field.
- Prints parameters for each and the parameter savings.

### 9. (Optional) Kaggle output cell
- Auto-injected by `kaggle_validator.py`: copies `__notebook__.ipynb` to `/kaggle/working/` so the executed notebook can be downloaded with all outputs.

## Key functions / classes

- `make_vgg_block(in_channels, out_channels, num_convs, use_batchnorm)`  
  Returns a `nn.Sequential` block of `Conv2d → [BN] → ReLU` repeated `num_convs` times, ending with `MaxPool2d`.

- `class VGGNetSmall(nn.Module)`  
  Assembles blocks, classifier head, and defines `forward(x)`.

- `train_epoch(model, loader, criterion, optimizer, device)`  
  Single training epoch; returns average loss and accuracy.

- `evaluate(model, loader, criterion, device)`  
  Validation/test evaluation; returns average loss and accuracy.

- `plot_history(history)`  
  Plots loss and accuracy curves.

- `visualize_filters(model)`  
  Plots first-layer filters.

- `receptive_field_comparison()`  
  Prints parameter counts for 7×7 vs. 3×3 stack.

## Data flow and shapes

- Input batch: `(B, 3, 32, 32)` for CIFAR-10.
- Block 1: `(B, 64, 32, 32)` → pool → `(B, 64, 16, 16)`.
- Block 2: `(B, 128, 16, 16)` → pool → `(B, 128, 8, 8)`.
- Block 3: `(B, 256, 8, 8)` → pool → `(B, 256, 4, 4)`.
- Block 4: `(B, 512, 4, 4)` → pool → `(B, 512, 2, 2)`.
- Flatten / global pool: `(B, 512)` or `(B, 512*2*2)`.
- Classifier output: `(B, 10)`.

## Deliberate simplifications vs. the full paper

- **CIFAR-10 instead of ImageNet**: 32×32 colour images, 10 classes. The original paper used 224×224 ImageNet with 1,000 classes.
- **Smaller classifier head**: FC-512 instead of the original FC-4096. This keeps the notebook feasible to run in minutes while still demonstrating the architecture.
- **Fewer total layers**: 11–13 weight layers instead of 16–19. The design principle (3×3 stacking, doubling channels) is preserved.
- **Single model, not an ensemble**: The paper reports best results with 7-model ensembles and multi-crop evaluation. The notebook trains a single network and evaluates on center crops.
- **BatchNorm added**: The original paper predates batch normalization (introduced by Ioffe & Szegedy in 2015). Adding it makes the small-scale demo train reliably without changing the core VGG idea.

The notebook therefore teaches the VGG design recipe — deep stacks of 3×3 convs, channel doubling, small classifier — rather than reproducing the full ILSVRC 2014 submission.
