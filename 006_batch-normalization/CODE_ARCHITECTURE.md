# Code Architecture — Batch Normalization Notebook

This document breaks down how `solution.ipynb` implements Batch Normalization (BN) from scratch in NumPy, plugs it into a small MLP, and visualizes activation distributions with and without BN across training.

## 1. Data loading and preprocessing

- **Dataset:** `make_moons` from scikit-learn, plus a small 2-D Gaussian-mixture toy dataset, generated inside the notebook.
- **Preprocessing:** features are standardized once at the start so that the *input* distribution is well-behaved; the network then learns from this fixed input. Labels are one-hot encoded for binary classification.
- **Train/validation split:** 80/20 split for monitoring generalization without needing large downloads.

## 2. NumPy BatchNorm layer

### 2.1 `BatchNorm1D` class

A fully vectorized forward/backward implementation for 2-D inputs of shape `(batch_size, features)`.

**Attributes**
- `gamma` — per-feature scale, shape `(features,)`, initialized to 1.
- `beta` — per-feature shift, shape `(features,)`, initialized to 0.
- `running_mean` — exponential moving average of batch means, shape `(features,)`.
- `running_var` — exponential moving average of unbiased batch variances, shape `(features,)`.
- `eps` — small constant for numerical stability, default `1e-5`.
- `momentum` — EMA momentum for running statistics, default `0.9` (training mode only).

**Key methods**
- `forward(x, training=True)`
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

- Weights initialized with small Gaussian random values.
- Bias is **optional and intentionally omitted** after the first linear layer when BN is used, because BN’s `beta` subsumes the bias term. A comparison branch keeps the bias for the non-BN network so the comparison is fair.
- Forward: `out = x @ W + b`.
- Backward: computes `dW`, `db`, and passes `dx` upstream.

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

- **Optimizer:** vanilla SGD with a fixed learning rate. A second run uses a 5× higher learning rate for the BN network to show that BN stabilizes large steps.
- **Batch size:** 32.
- **Epochs:** 100–200, enough to show the gap clearly on this toy problem.
- **Logging:** every few epochs the notebook records training loss, validation loss, and validation accuracy for both networks.
- **No PyTorch/TensorFlow autograd:** all gradients are hand-derived and implemented in NumPy, matching the paper’s algorithmic spirit.

## 5. Visualization

### 5.1 Training-curve comparison

A side-by-side plot of:
- train loss vs. epochs (BN vs. plain)
- validation loss vs. epochs (BN vs. plain)
- validation accuracy vs. epochs (BN vs. plain)

### 5.2 Activation distribution histograms

Every 20 epochs the notebook samples activations entering the second hidden layer and overlays histograms for the BN and plain networks. This directly reproduces the intuition in Figure 1(b,c) of the paper: BN activations stay centered and stable, while plain-network activations drift.

### 5.3 Layer-wise mean / variance tracking

A line plot showing the running mean and variance of one representative hidden unit across training for both networks. The BN line should stay flat near (0, 1), while the plain-network line wanders.

## 6. Inference sanity check

After training, the notebook switches both networks to evaluation mode and prints:
- final validation accuracy,
- the learned `gamma` and `beta` values for one BN layer,
- a check that the running statistics approximate the population statistics.

## 7. Deliberate simplifications vs. the full paper

| Full paper | Notebook simplification |
|---|---|
| ImageNet / MNIST experiments | Toy 2-D classification for speed and clarity |
| Convolutional BN over spatial locations | 1-D BN over features only (`BatchNorm1D`) |
| Distributed training with momentum SGD | Single-machine vanilla SGD |
| Sigmoid nonlinearity demonstration | Tanh (another saturating nonlinearity) on one branch |
| Ensemble results and 4.9% ImageNet top-5 error | Not reproduced; focus is on the mechanics and qualitative speedup |
| `m/(m-1)` unbiased variance correction | Applied in running-statistic update, not in per-batch variance |

These simplifications keep the notebook Colab-runnable in under a minute while preserving the exact algorithm the paper proposes.
