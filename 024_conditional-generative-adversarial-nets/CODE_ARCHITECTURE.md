---
title: "Conditional GAN: Code architecture"
parent: "024 Conditional GAN"
nav_order: 1
---
# CODE_ARCHITECTURE.md — Conditional Generative Adversarial Nets (CGAN)

## Overview

The notebook implements a from-scratch Conditional GAN using PyTorch. Both the generator and discriminator are MLPs that accept concatenated noise+label and image+label respectively. The model is trained on MNIST to generate specific digit classes on demand. After training, a grid of generated digits (one row per class, 0-9) is displayed to demonstrate conditional generation.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, torchvision, matplotlib, numpy. Set random seeds (42). Configure device (CUDA if available).
- **Figures**: shown as JPEG at 80 dpi so the saved notebook stays small.
- **Outputs**: detects Kaggle from its `KAGGLE_KERNEL_RUN_TYPE` environment variable and sets `OUT_DIR` to the current folder on Kaggle (its output folder) or `./outputs` anywhere else; every figure is saved there.
- **Kaggle**: on Kaggle only, an `atexit` hook copies the executed notebook to `executed_solution.ipynb` in the output folder so its outputs can be downloaded.

### 2. Hyperparameters
- **Key values**:
  - `latent_dim = 100` — noise vector dimension (z drawn from a standard normal with `torch.randn`)
  - `num_classes = 10` — MNIST digit classes (0-9)
  - `img_dim = 784` — flattened 28×28 image
  - `hidden_g = 256` — first hidden layer of the generator
  - `hidden_d = 512` — first hidden layer of the discriminator
  - `lr = 0.0002`, `beta1 = 0.5` — Adam settings
  - `batch_size = 64` — minibatch size
  - `num_epochs = 100` — training epochs
  - `k_steps = 1` — discriminator updates per generator update (the loop does one of each)

### 3. Data Loading (MNIST)
- **Purpose**: Load MNIST via torchvision (download progress bars silenced). Images flattened to 784-dim and normalized to [-1, 1]. Labels converted to one-hot vectors.
- **Shapes**: 
  - Images: `(batch_size, 1, 28, 28)` → `(batch_size, 784)`
  - Labels: `(batch_size,)` → `(batch_size, 10)` one-hot
- **Custom Dataset**: `MNISTConditional` returns `(image_flat, one_hot_label, label)` so the DataLoader yields all three together.

### 4. Generator Network (G)
- **Purpose**: Maps noise + label to a 784-dim image.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 100+10=110)` — concatenated noise z and one-hot label y
  - Hidden: `(batch_size, 256)` — LeakyReLU(0.2)
  - Hidden: `(batch_size, 512)` — LeakyReLU(0.2)
  - Output: `(batch_size, 784)` — Tanh (output in [-1, 1], matching the normalized data)
- **Data flow**: `concat(z, y)` → linear → LeakyReLU → linear → LeakyReLU → linear → Tanh → fake image (784-dim)
- **Simplification vs paper**: the paper maps z to a ReLU layer of 200 units and y to a ReLU layer of 1000 units, joins them in a ReLU layer of 1200 units, and ends in a sigmoid layer of 784 units. We concatenate z and y at the input instead.

### 5. Discriminator Network (D)
- **Purpose**: Classifies image+label pair as real (1) or fake (0).
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 784+10=794)` — concatenated image and one-hot label
  - Hidden: `(batch_size, 512)` — LeakyReLU(0.2) + Dropout(0.3)
  - Hidden: `(batch_size, 256)` — LeakyReLU(0.2) + Dropout(0.3)
  - Output: `(batch_size, 1)` — Sigmoid (probability)
- **Key insight**: The discriminator sees the label, so it checks not just "is this real?" but "is this a real digit matching label y?"
- **Simplification vs paper**: Paper used maxout activations; we use LeakyReLU for standard practice.

### 6. Loss Functions & Optimizers
- **Loss**: `nn.BCELoss()` — Binary Cross Entropy
- **Optimizers**: Adam for both G and D (lr=0.0002, betas=(0.5, 0.999))
- **Non-saturating generator loss**: Generator maximizes log(D(G(z\|y))) using real labels (1.0) for fake samples — stronger gradients than minimizing log(1-D(G(z\|y))).

### 7. Training Loop
- **Per-epoch flow**:
  ```
  for each minibatch (x_real, y_real):
    --- Discriminator step ---
    1. Real: D(concat(x_real, y_real)) → BCE(D_real, 1)
    2. Fake: sample z, y_fake; G(concat(z, y_fake)) → x_fake
       D(concat(x_fake, y_fake)) → BCE(D_fake, 0)
    3. D_loss = BCE_real + BCE_fake
    4. Backprop D_loss, step D optimizer
    
    --- Generator step ---
    5. Sample new z, y_fake; G(concat(z, y_fake)) → x_fake
    6. D(concat(x_fake, y_fake)) → BCE(D_fake, 1)  [non-saturating]
    7. G_loss = BCE(D_fake, 1)
    8. Backprop G_loss, step G optimizer
  ```
- **Label sampling for G**: Random labels are sampled for the generator — the generator must learn to produce *any* requested class.
- **Tracking**: D_loss, G_loss logged per epoch. Fixed noise + fixed labels saved for visualizing progression.

### 8. Training Curves
- **Loss curves**: per-epoch average D_loss and G_loss.

### 9. Training Progression
- **Grids**: the fixed noise and fixed labels from the training loop, generated at epochs 1, 25, 50, 75 and 100 (rows = digit class 0-9).

### 10. Conditional Generation Grid
- **Class-conditioned grid**: a 10×10 grid — 10 rows (one per digit class 0-9), 10 columns (fresh noise per sample). This directly demonstrates conditional generation.

### 11. Same Noise, Different Classes
- **Method**: fix one noise vector z and vary the label y from 0 to 9, to show that the label sets the digit while the noise sets the style.

### 12. Summary
- **Stats cell**: prints the final D and G losses, epochs, and parameter counts of both networks.
- **What the run above shows**: the numbers and observations from the executed run, followed by the key takeaways.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `MNISTConditional` | Dataset | index | (784,), (10,), int | Flattened image, one-hot label, integer label |
| `Generator` | nn.Module | (B, 110) | (B, 784) | 3-layer MLP, concat(z,y), LeakyReLU, Tanh |
| `Discriminator` | nn.Module | (B, 794) | (B, 1) | 3-layer MLP, concat(x,y), LeakyReLU, Dropout, Sigmoid |
| `BCELoss` | Loss | (B, 1), (B, 1) | scalar | Binary cross-entropy |

## Data Flow Diagram

```
Noise z ~ N(0, I)          Label y (one-hot, dim 10)
        |                           |
        +-----------+---------------+
                    |
              Generator G
                    |
              Fake G(z|y) ──────────────────┐
                                              |
  Real MNIST x ──┐                            |
                 +------ concat(x, y) ───────+
  Label y ───────┘                            |
                                              v
                                    Discriminator D
                                         |        |
                                         v        v
                                    D(real|y)  D(fake|y)
                                         |        |
                                         v        v
                              BCE(D(real|y),1) + BCE(D(fake|y),0) → D_loss
                              BCE(D(fake|y),1)                    → G_loss
                                         |                |
                                         v                v
                                    Update D           Update G
```

## Deliberate Simplifications vs Full Paper

1. **Dataset**: MNIST only — paper also used MIR Flickr 25,000 for multi-modal image tagging
2. **Architecture**: MLPs with LeakyReLU — paper used maxout activations for D, ReLU for G
3. **Concatenation strategy**: We concatenate z+y upfront; paper used separate hidden layers before combining
4. **Optimizer**: Adam (betas=0.5, 0.999) — paper used SGD with momentum and exponential LR decay
5. **No Parzen window evaluation**: Paper's quantitative metric; we use visual inspection
6. **No multi-modal experiment**: Paper's Flickr image-tagging experiment requires pre-trained CNN features + word embeddings; we focus on the MNIST conditional generation demo
7. **Noise prior**: standard normal (`torch.randn`) — the paper draws z uniformly from the unit hypercube
8. **Output range**: images scaled to [-1, 1] with a Tanh output — the paper uses a sigmoid output layer
