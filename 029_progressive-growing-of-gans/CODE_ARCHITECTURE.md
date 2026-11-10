---
title: "ProGAN: Code architecture"
parent: "029 ProGAN"
nav_order: 1
---
# Code Architecture — Progressive Growing of GANs

## Overview

The notebook implements a small Progressive Growing GAN (ProGAN) in PyTorch, following Karras et al. (2017). Both networks grow from 4×4 to 64×64 with fade-in transitions, and use equalized learning rate, pixelwise normalization and minibatch standard deviation. It trains on MNIST digits for 500 iterations per stage, so it shows the mechanics rather than the paper's image quality.

## Section-by-Section Breakdown

### 1. Imports and Setup
- PyTorch, torchvision, NumPy, matplotlib; seeds 42; deterministic cuDNN settings
- Figures shown as JPEG at 80 dpi to keep the notebook small
- On Kaggle, an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Configuration
- `LATENT_DIM = 128`
- `CHANNELS = [512, 512, 256, 128, 64]` for resolutions `[4, 8, 16, 32, 64]`
- `FADE_ITERATIONS = 200`, `STABLE_ITERATIONS = 300` (500 per stage; the 4×4 stage has no fade-in)
- `LR = 0.001`, `BETA1 = 0.0`, `BETA2 = 0.99` (the paper's Adam settings), `LRELU_SLOPE = 0.2`
- `BATCH_SIZES = {4: 64, 8: 64, 16: 32, 32: 16, 64: 8}`

### 3. Equalized Learning Rate (`EqualizedConv2d`, `EqualizedLinear`)
- Weights start from N(0, 1) and are multiplied at every forward pass by `√(2 / fan_in)`, He's constant
- This follows Section 4.1 of the paper

### 4. Normalization Layers
- `PixelNorm`: `x / sqrt(mean(x², over channels) + 1e-8)`, used after each convolution in the generator (Section 4.2)
- `MinibatchStdDev`: the standard deviation of each feature at each location over the batch, averaged to one value and appended as an extra channel (Section 3)

### 5. Generator (`GeneratorBlock`, `Generator`)
- Initial block: `EqualizedLinear(128 → 512×4×4)` → reshape → 3×3 conv → PixelNorm → LeakyReLU
- Each growth block: nearest-neighbor upsampling ×2 → (3×3 conv → PixelNorm → LeakyReLU) twice
- `to_rgb`: one 1×1 conv per resolution
- Fade-in: `(1 - α) · upsample(toRGB(old features)) + α · toRGB(new features)`

### 6. Discriminator (`DiscriminatorBlock`, `Discriminator`)
- `from_rgb`: one 1×1 conv per resolution
- Each block: (3×3 conv → LeakyReLU) twice → average pooling ×2
- Final block at 4×4: MinibatchStdDev → 3×3 conv → 4×4 conv → linear → one logit
- Fade-in: blends `fromRGB(avgpool(image))` on the old path with `block(fromRGB(image))` on the new path

### 7. Data Loading (`MultiResolutionMNIST`)
- MNIST resized to the current resolution, scaled to [-1, 1] and repeated to 3 channels
- Downloaded quietly to a temp folder (needs Kaggle internet on)

### 8. Training Functions (`get_dataloader`, `train_step`)
- One discriminator update, then one generator update, per batch
- Loss: non-saturating GAN loss, i.e. binary cross-entropy on the logits. The paper uses WGAN-GP with a drift term instead

### 9. Progressive Training
- For each stage: grow both networks, recreate the Adam optimizers, run the fade-in (α from 0 to 1), then the stable phase
- Prints the losses every 50 fade-in and 100 stable iterations, and keeps 16 samples per resolution

### 10–11. Visualization
- A grid of 16 samples for each resolution, then raw and smoothed loss curves with stage boundaries

### 12. Summary
- What the saved run shows, with its real numbers, plus the simplifications

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `EqualizedConv2d`, `EqualizedLinear` | Layers with runtime weight scaling |
| `PixelNorm` | Per-pixel normalization across channels |
| `MinibatchStdDev` | Appends the batch's average standard deviation as a feature map |
| `GeneratorBlock` | Upsample + two 3×3 convs |
| `DiscriminatorBlock` | Two 3×3 convs + downsample |
| `Generator`, `Discriminator` | Progressive networks with `grow()` and `forward(x, alpha)` |
| `MultiResolutionMNIST` | MNIST at a chosen resolution |
| `get_dataloader`, `train_step` | Data loader per stage; one D + G update |

## Data Flow / Shapes

```
Latent z (B, 128)
  → Linear → (B, 512, 4, 4)
  → [Upsample ×2 → Conv → Conv] per stage: 8×8 → 16×16 → 32×32 → 64×64
  → toRGB (1×1 conv) → (B, 3, H, W)

Image (B, 3, H, W)
  → fromRGB (1×1 conv)
  → [Conv → Conv → AvgPool ×2] per stage: … → 4×4
  → MinibatchStdDev → Conv3×3 → Conv4×4 → Linear → (B, 1)
```

## Deliberate Simplifications vs Full Paper

1. **Dataset:** MNIST digits instead of CelebA-HQ faces (the paper also uses CelebA, LSUN and CIFAR-10).
2. **Resolution:** 64×64 instead of 1024×1024.
3. **Training length:** 500 iterations per stage, against the paper's 800k real images per fade-in or stable phase.
4. **Loss:** non-saturating GAN loss instead of WGAN-GP with a drift term.
5. **Latent size:** 128 instead of 512, and the latent vectors are not normalized onto a hypersphere.
6. **No running average of generator weights:** the paper uses one, with decay 0.999, for the images it shows.
