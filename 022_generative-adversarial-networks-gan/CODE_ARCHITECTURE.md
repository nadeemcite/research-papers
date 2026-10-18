---
title: "GAN: Code architecture"
parent: "022 GAN"
nav_order: 1
---
# CODE_ARCHITECTURE.md — Generative Adversarial Networks (GAN)

## Overview

The notebook implements a from-scratch GAN using the original paper's framework: a generator MLP and a discriminator MLP trained in an adversarial minimax loop on MNIST for 100 epochs. It logs the losses and D's outputs, shows samples from a fixed set of noise vectors at several points in training, and saves an animated GIF of the progression.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, torchvision (for MNIST), matplotlib, numpy. Set random seeds for reproducibility. Configure device (CUDA if available).
- **Key detail**: Only pip-installable packages used (torch, torchvision, matplotlib, numpy). No external data downloads beyond torchvision's built-in MNIST loader.

### 2. Hyperparameters
- **Purpose**: Centralize all training configuration.
- **Key values**:
  - `latent_dim = 100` — dimension of noise vector z (standard Gaussian)
  - `img_dim = 784` — flattened MNIST image (28×28)
  - `hidden_dim = 256` — hidden layer size for both G and D
  - `lr = 0.0002` — learning rate (following common GAN practice)
  - `batch_size = 64` — minibatch size
  - `num_epochs = 100` — training epochs
  - `k_steps = 1` — discriminator updates per generator update (as in the paper)
  - `beta1 = 0.5` — Adam beta1
  - `sample_every = 5` — save fixed-noise samples after epoch 1 and every 5 epochs
- **Data flow**: These constants flow into model construction and the training loop.

### 3. Data Loading (MNIST)
- **Purpose**: Load MNIST via torchvision, flatten images to 784-dim vectors, scale pixels to [-1, 1].
- **Key functions**: `torchvision.datasets.MNIST`, `DataLoader`, `transforms.Normalize((0.5,), (0.5,))`, and a helper `to_img(x) = (x + 1) / 2` that maps back to [0, 1] for plotting.
- **Shapes**:
  - Raw MNIST: `(batch_size, 1, 28, 28)` → flattened to `(batch_size, 784)`
  - Pixel values in [-1, 1], to match the generator's Tanh output
- **Simplification vs paper**: The paper also tested on TFD and CIFAR-10. We use only MNIST for clarity and speed.

### 4. Generator Network (G)
- **Purpose**: Maps noise vector z to a 784-dim image vector.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 100)` — latent vector
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2)
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2)
  - Output: `(batch_size, 784)` — with Tanh (output in [-1, 1])
- **Key class**: `Generator(nn.Module)` with `forward(z)` method
- **Data flow**: `z ~ N(0, I)` → linear → LeakyReLU → linear → LeakyReLU → linear → Tanh → fake image (784-dim)
- **Why Tanh, not Sigmoid**: an earlier version used [0, 1] data with a Sigmoid output. On every seed we tried, it collapsed in the first epoch (one identical, almost pure black-and-white blob for every z), and the discriminator then won completely. Switching to [-1, 1] data and Tanh, with nothing else changed, fixed it.
- **Simplification vs paper**: The paper's generator used a mix of ReLU and sigmoid activations. We use LeakyReLU in the hidden layers, a common modern choice.

### 5. Discriminator Network (D)
- **Purpose**: Scores an input image as real or fake.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 784)` — image vector
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2) + Dropout(0.3)
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2) + Dropout(0.3)
  - Output: `(batch_size, 1)` — raw logit; `sigmoid(logit)` = probability the input is real
- **Key class**: `Discriminator(nn.Module)` with `forward(x)` method
- **Data flow**: image → linear → LeakyReLU → Dropout → linear → LeakyReLU → Dropout → linear → logit
- **Simplification vs paper**: The paper's discriminator used maxout activations with dropout. We use LeakyReLU + dropout.

### 6. Loss Functions & Optimizers
- **Purpose**: Define the adversarial loss (BCE) and optimizers for both networks.
- **Key details**:
  - **Loss**: `nn.BCEWithLogitsLoss()` — sigmoid + binary cross-entropy in one numerically stable op. With a separate `Sigmoid` + `BCELoss`, the gradient reaching G vanishes once D is extremely confident (about 1e-23 at G_loss ≈ 79), so G can never recover. With logits, the gradient on D's score stays near -1.
  - **Real label**: 1.0, **Fake label**: 0.0
  - **Optimizer**: Adam for both G and D (lr=0.0002, betas=(0.5, 0.999)) — beta1=0.5 is a common GAN setting that reduces momentum oscillation
- **Non-saturating generator loss**: Instead of minimizing log(1 - D(G(z))), we maximize log(D(G(z))) by using real labels (1.0) for fake samples — the practical trick from Section 3 of the paper that gives stronger gradients early in training.

### 7. Training Loop
- **Purpose**: Alternating D and G updates following Algorithm 1 from the paper.
- **Per-epoch flow**:
  ```
  for each minibatch:
    --- Discriminator step (k=1) ---
    1. Sample real batch x from MNIST
    2. Sample noise z, generate fake batch G(z)
    3. D_loss = -[mean(log D(x)) + mean(log(1 - D(G(z))))]
       (implemented as BCE on real + BCE on fake, with G(z) detached)
    4. Backprop D_loss, step D optimizer

    --- Generator step ---
    5. Sample new noise z, generate fake batch G(z)
    6. G_loss = -mean(log D(G(z)))  [non-saturating]
       (implemented as BCE with real labels on fake data)
    7. Backprop G_loss, step G optimizer
  ```
- **Tracking**: per-epoch averages of D_loss, G_loss, D(x) and D(G(z)) (D's outputs converted to probabilities with a sigmoid). A healthy run keeps D(G(z)) well above 0 and both losses small and bounded.
- **Key detail**: A fixed noise tensor `fixed_noise` (64 vectors) is passed through G after epoch 1 and every 5 epochs. The outputs go into `img_list`, and the matching epoch numbers into `sample_epochs`.

### 8. Loss Curves
- Plots D_loss and G_loss per epoch, with dashed reference lines at the theoretical equilibrium values (D loss 2·log 2 ≈ 1.39, G loss log 2 ≈ 0.69).

### 9. Generated Samples and Real vs Fake
- **Progression grid**: the first 8 fixed-noise samples at 5 saved epochs (first, ¼, ½, ¾, last), one column per noise vector.
- **Real vs Fake**: 8 real MNIST digits next to 8 samples from fresh random noise.

### 10. Animation
- Builds an 8×8 grid of all 64 fixed-noise samples for each saved epoch, and saves the frames as `gan_progress.gif` with matplotlib's `PillowWriter`. The GIF is displayed inline, which keeps the notebook much smaller than an embedded HTML animation.

### Evaluation
- **Method**: Visual inspection of generated digit grids, plus the logged D(x) / D(G(z)) values. No quantitative metric (like FID) is computed.
- **Simplification**: The original paper used a Parzen window-based log-likelihood estimate for quantitative evaluation. We skip this as it is known to be an unreliable metric and is not the key contribution.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `Generator` | nn.Module | (B, 100) | (B, 784) | 3-layer MLP, LeakyReLU, Tanh output |
| `Discriminator` | nn.Module | (B, 784) | (B, 1) | 3-layer MLP, LeakyReLU, Dropout, raw logit output |
| `BCEWithLogitsLoss` | Loss | (B, 1), (B, 1) | scalar | Sigmoid + binary cross-entropy for adversarial training |
| `to_img` | Function | any tensor in [-1, 1] | same shape in [0, 1] | Undo the normalization for plotting |
| Training loop | Notebook cell | — | — | Alternating D/G updates, logging, fixed-noise snapshots |
| `gan_progress.gif` | File | `img_list` | GIF | Animation of the 64 fixed-noise samples over training |

## Data Flow Diagram

```
Noise z ~ N(0, I)                            Real MNIST images x (in [-1, 1])
        |                                              |
        v                                              |
   Generator G (Tanh)                                  |
        |                                              |
        v                                              v
  Fake G(z) ──────────────────────────────────► Discriminator D (logit)
                                                   |        |
                                                   v        v
                                              D(real)    D(fake)
                                                   |        |
                                                   v        v
                                              BCE(D(real), 1) + BCE(D(fake), 0)  → D_loss
                                              BCE(D(fake), 1)                     → G_loss (non-saturating)
                                                   |                |
                                                   v                v
                                              Update D           Update G
```

## Deliberate Simplifications vs Full Paper

1. **Dataset**: MNIST only — paper also used TFD and CIFAR-10
2. **Architecture**: MLPs with LeakyReLU — paper used maxout activations for D and a ReLU/sigmoid mix for G
3. **Output range**: data in [-1, 1] with a Tanh generator output (see section 4 for why)
4. **No Parzen window evaluation**: Paper's quantitative metric is known to be unreliable; we rely on visual inspection
5. **Adam optimizer**: Paper used SGD with momentum; Adam with betas=(0.5, 0.999) is a common GAN choice today
6. **Noise prior**: standard Gaussian; the paper doesn't tie the framework to one prior
7. **Dropout in D only**: matches the paper, which applied dropout when training the discriminator
8. **k=1**: Matches the paper's choice (one D step per G step)
