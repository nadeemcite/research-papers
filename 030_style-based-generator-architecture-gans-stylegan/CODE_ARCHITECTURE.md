---
title: "StyleGAN: Code architecture"
parent: "030 StyleGAN"
nav_order: 1
---
# Code Architecture — StyleGAN

## Overview

The notebook builds a miniature StyleGAN in PyTorch: a mapping network, a synthesis network that starts from a learned constant and is steered by AdaIN styles, per-pixel noise, style mixing, R1 regularization and a running average of the generator. It trains on 800 synthetic 32×32 images of colored circles and squares, so it shows how the parts work, not the paper's image quality.

## Section-by-Section Breakdown

### 1. Imports and Configuration
- Seeds 42; figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`
- `LATENT_DIM = 128`, `W_DIM = 128`, `IMAGE_SIZE = 32`, `CHANNELS = 32`, `BATCH_SIZE = 32`
- `EPOCHS = 120` (25 batches each, so 3,000 training steps), `LR = 1e-3`
- `MAPPING_LR_SCALE = 0.01`, `MIXING_PROB = 0.9`, `R1_GAMMA = 10.0`

### 2. Toy Dataset
- `make_synthetic_dataset(800, 32)`: each image has a random background color and one circle or square of random size, position and color, scaled to [-1, 1]
- No download; a `DataLoader` with batch size 32 and `drop_last=True`

### 3. Equalized Learning Rate (`EqualizedConv2d`, `EqualizedLinear`)
- Weights start from N(0, 1) and are scaled at runtime by `1/√fan_in`, as in Progressive GAN

### 4. `AdaIN`
- Normalizes each channel of each image to zero mean and unit variance, then applies a per-channel scale and bias from the style (Eq. 1 of the paper)

### 5. `NoiseInjection`
- Adds one channel of fresh Gaussian noise, scaled by a learned per-channel weight that starts at zero, on every call (training and sampling)

### 6. `StyledConvBlock`
- Two rounds of: 3×3 conv → noise → LeakyReLU(0.2) → AdaIN, each with its own affine map from w to the style

### 7. `MappingNetwork`
- 8 equalized linear layers with LeakyReLU, z (128) → w (128)

### 8. `SynthesisNetwork`
- Learned constant of shape 32×4×4, then four `StyledConvBlock`s at 4, 8, 16 and 32 pixels, with bilinear upsampling between them
- Style mixing: blocks from `crossover_idx` onward use a second w
- `to_rgb`: a 1×1 conv to 3 channels

### 9. `StyleGenerator`
- Mapping + synthesis; `generate()` mixes two latents with probability `mixing_prob`, at a random crossover block

### 10. `Discriminator`
- Three stride-2 4×4 convs (32 → 64 → 128 channels) with LeakyReLU, then a linear layer to one logit

### 11. Models and Optimizers
- Adam with betas (0.0, 0.99); the mapping network's learning rate is 0.01 × `LR`
- `ema_G`: a copy of the generator updated as a running average with decay 0.999, used for all sample images

### 12. Training Utilities
- Non-saturating losses: D uses `softplus(-real) + softplus(fake)`, G uses `softplus(-fake)`
- `r1_penalty`: mean squared gradient of D's output on real images, weighted by `R1_GAMMA / 2`

### 13. Training Loop
- Per batch: one D step with the R1 penalty, one G step, then the EMA update
- Prints mean losses for epoch 1 and every 20th epoch

### 14–18. Visualizations
- Loss curves (raw and 50-step mean), 16 EMA samples, style mixing at crossover blocks 1–3, eight samples from one z with different noise, and an 8-step interpolation in Z

### 19. Summary
- What the saved run shows, with its real numbers

## Data Flow / Shapes

```
z (B, 128) → MappingNetwork → w (B, 128)
const (1, 32, 4, 4) → expand to (B, 32, 4, 4)
  → StyledConvBlock(w) → 4×4 → upsample → 8×8 → 16×16 → 32×32
  → to_rgb → (B, 3, 32, 32)
Discriminator: (B, 3, 32, 32) → 16×16×32 → 8×8×64 → 4×4×128 → Linear → (B, 1)
```

## Deliberate Simplifications vs Full Paper

1. **Data and size:** 800 synthetic 32×32 shapes instead of FFHQ faces at 1024×1024.
2. **Width:** 32 feature maps and a 128-dimensional z and w, instead of 512.
3. **No progressive growing:** the paper's generator follows Progressive GAN's growing schedule; this one trains at 32×32 from the start.
4. **Shorter training:** 3,000 steps instead of the paper's millions of images.
5. **Simple discriminator:** three conv layers, without the minibatch standard deviation layer.
6. **No truncation trick and no quality metrics** (FID, perceptual path length, linear separability).
