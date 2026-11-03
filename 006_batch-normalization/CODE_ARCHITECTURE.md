---
title: "BatchNorm: Code architecture"
parent: "006 BatchNorm"
nav_order: 1
---
# Code Architecture — Batch Normalization Notebook

This document breaks down how `solution.ipynb` implements Batch Normalization (BN) from scratch in NumPy, plugs it into a small MLP, and visualizes activation distributions with and without BN across training.

## 1. Data loading and preprocessing

- **Dataset:** `make_moons` from scikit-learn (1,200 points, noise 0.25), generated inside the notebook.
- **Preprocessing:** features are standardized once at the start so that the *input* distribution is well-behaved; the network then learns from this fixed input. Labels are one-hot encoded for binary classification.
- **Train/validation split:** 80/20 split for monitoring generalization without needing large downloads.

## 2. NumPy BatchNorm layer

### 2.1 `BatchNorm1D` class

A fully vectorized forward/backward implementation for 2-D inputs of shape `(batch_size, features)`.

**Attributes**
- `gamma` — per-feature scale, shape `(1, features)`, initialized to 1.
- `beta` — per-feature shift, shape `(1, features)`, initialized to 0.
- `running_mean` — exponential moving average of batch means, shape `(1, features)`.
- `running_var` — exponential moving average of unbiased batch variances, shape `(1, features)`.
- `eps` — small constant for numerical stability, default `1e-5`.
- `momentum` — EMA momentum for running statistics, default `0.9` (training mode only).

**Key methods**
- `forward(x)` (the mode comes from `self.training`, set by `train()` and `eval()`)
  - Training: computes batch mean `mu_B` and variance `var_B`, normalizes `x_hat = (x - mu_B) / sqrt(var_B + eps)`, returns `y = gamma * x_hat + beta`. Stores intermediates for backward pass and updates `running_mean` / `running_var`.
  - Inference: normalizes with running statistics and applies the learned affine transform.
- `backward(dy)`
  - Computes gradients `dgamma`, `dbeta`, and `dx` using the closed-form chain-rule equations from Algorithm 1 of the paper. This is the core pedagogical component: the student can see *why* backprop through the mean and variance matters.

### 2.2 Shape and data flow

```
Input x                (batch_size, features)
  │
  ▼
BatchNorm forward      (batch_size, features)  → stores x, mu_B, var_B, x_hat, y
  │
  ▼
Activation (ReLU / tanh)   (batch_size, features)
  │
  ▼
Linear layer + softmax / logits   (batch_size, num_classes)
  │
  ▼
Cross-entropy loss     scalar
```

## 3. Small MLP built from scratch

### 3.1 `Linear` layer

- Weights initialized with small Gaussian random values (scale 0.1).
- Bias is **omitted on all hidden linear layers** when BN is used, because BN’s `beta` subsumes the bias term. The output layer and the plain network keep their biases.
- Forward: `out = x @ W + b`.
- Backward: computes `dW`, `db`, and passes `dx` upstream. The incoming `dout` is already averaged over the batch by the loss, so these are not divided by the batch size again.

### 3.2 `ReLU` and `Tanh` activations

- `ReLU`: element-wise `max(0, x)` with simple mask gradient.
- `Tanh`: chosen for one experiment to demonstrate that BN rescues saturating nonlinearities, mirroring the paper’s BN-x5-Sigmoid result on ImageNet.

### 3.3 `MLPClassifier` model

A small three-hidden-layer network:

```
input (2) → Linear(2, 64) → BN → ReLU
           → Linear(64, 64) → BN → ReLU
           → Linear(64, 64) → BN → ReLU
           → Linear(64, 2) → Softmax
```

A twin “plain” network is created with the same layer widths and initialization but **no BN**, so training curves can be compared directly.

## 4. Training loop

- **Optimizer:** vanilla SGD with a fixed learning rate: 0.05 for the plain network and 0.25 (5×) for the BN network. Because the rates differ, the comparison is not like-for-like.
- **Batch size:** 32.
- **Epochs:** 150.
- **Logging:** after epoch 1 and every 5 epochs the notebook records training loss, validation loss, and validation accuracy for both networks.
- **No PyTorch/TensorFlow autograd:** all gradients are hand-derived and implemented in NumPy, matching the paper’s algorithmic spirit.

## 5. Visualization

### 5.1 Training-curve comparison

A side-by-side plot of:
- train loss vs. epochs (BN vs. plain)
- validation loss vs. epochs (BN vs. plain)
- validation accuracy vs. epochs (BN vs. plain)

### 5.2 Activation distribution histograms

Two fresh networks are trained for 150 epochs, and every 30 epochs the notebook runs the validation set up to the first activation layer and overlays histograms of its inputs for the BN and plain networks. BN's inputs stay centred with a spread of about 1. The plain network's are narrower and also barely move, so this small network shows no visible drift.

### 5.3 Tanh comparison

The same plain vs BN (5× lr) comparison with Tanh activations, as a validation-accuracy plot. BN gets above 0.92 much sooner but is noisier, and both end near 0.925.

## 6. Inference sanity check

After training, the notebook switches both networks to evaluation mode and prints:
- final validation loss and accuracy (plain 0.1744 / 0.9208, BN 0.1651 / 0.9250),
- the learned `gamma` and `beta` values of the first BN layer,
- that layer's running mean and variance (printed only, not compared with population statistics).

## 7. Deliberate simplifications vs. the full paper

| Full paper | Notebook simplification |
|---|---|
| ImageNet / MNIST experiments | Toy 2-D classification for speed and clarity |
| Convolutional BN over spatial locations | 1-D BN over features only (`BatchNorm1D`) |
| Distributed training with momentum SGD | Single-machine vanilla SGD |
| Sigmoid nonlinearity demonstration | Tanh (another saturating nonlinearity) on one branch |
| Ensemble results and 4.9% ImageNet top-5 error | Not reproduced; focus is on the mechanics and qualitative speedup |
| `m/(m-1)` unbiased variance correction | Applied in running-statistic update, not in per-batch variance |

These simplifications keep the notebook small and quick to run while keeping the algorithm the paper proposes.
