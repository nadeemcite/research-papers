---
title: "LayerNorm: Code architecture"
parent: "042 LayerNorm"
nav_order: 1
---
# Code Architecture — Layer Normalization Notebook

## Overview

The notebook implements Layer Normalization from scratch in NumPy (with a finite-difference gradient check) and in PyTorch (checked against `nn.LayerNorm`). It then compares an RNN with no normalization, with a from-scratch BatchNorm and with LayerNorm on a toy sequence classification task, trains a small pre-norm Transformer block with LayerNorm on the same task, and repeats the RNN comparison with batch size 1. Everything runs on the CPU.

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `numpy`, `matplotlib`; seeds 42; one CPU thread and `torch.use_deterministic_algorithms(True)` so reruns give the same numbers
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. LayerNorm from Scratch in NumPy
- `layer_norm_numpy(x, gain, bias, eps=1e-5)`: μ and σ² over the H elements of one sample, `(x − μ)/√(σ² + eps)`, then gain and bias
- Checks that the output has mean 0 and standard deviation 1 for H = 64
- `layer_norm_backward_numpy`: analytic gradients for x, gain and bias (loss = sum of outputs), compared with central finite differences; asserts relative error < 1e-5

### 3. LayerNorm as a PyTorch Module
- `LayerNorm(features, eps=1e-5)`: parameters `gamma` (ones) and `beta` (zeros); mean and biased variance over the last dimension
- Compared with `nn.LayerNorm` on (32, 64) and (16, 20, 64) inputs; asserts a maximum difference < 1e-6

### 4. BatchNorm for Comparison
- `BatchNorm1d(features, eps=1e-5, momentum=0.1)`: normalizes each feature over the batch (dim 0) in training and updates running mean and variance; uses the running statistics in evaluation

### 5. Toy Sequence Classification Task
- `generate_sequence_data(n_samples, seq_len=20, n_features=64, seed)`: class 0 adds a linear trend rising from 0 to 2 over the 20 steps to Gaussian noise, class 1 a falling trend
- 1,000 training sequences (seed 42) and 200 test sequences (seed 123); X shape (N, 20, 64)
- A figure shows 5 features of one sequence from each class

### 6. Simple RNN with Pluggable Normalization
- `RNNCellWithNorm(input_size, hidden_size, norm_type)`: `a_t = W_xh x_t + W_hh h_{t−1}` (no biases), then LayerNorm, BatchNorm, or a plain bias for `'none'`, then `h_t = tanh(a_t)`; Xavier init for `W_xh`, orthogonal for `W_hh`
- The BatchNorm variant shares one set of statistics across all time steps
- `SimpleRNN`: unrolls the cell over the 20 steps from `h_0 = 0` and classifies the final hidden state with `Linear(64, 2)`

### 7. Training Loop
- `train_and_evaluate(norm_type, ..., epochs=50, lr=0.001, batch_size=64, hidden_size=64)`: Adam, cross-entropy, gradient-norm clipping at 5; records the average loss and train/test accuracy each epoch and prints every 10 epochs
- Trains the three RNNs and prints their final test accuracies

### 8. Transformer Block with LayerNorm
- `SimpleSelfAttention`: single-head scaled dot-product attention
- `TransformerBlock(embed_dim, ff_dim)`: pre-norm, `x + attn(LN(x))`, then `x + FFN(LN(x))` with a GELU feed-forward network (64 → 128 → 64)
- `TransformerClassifier`: `Linear(64, 64)` input projection plus a learned positional embedding (1, 20, 64), one block, a final LayerNorm, mean pooling over time, `Linear(64, 2)`
- Trained for 50 epochs with the same optimizer settings as the RNNs

### 9. Visualization & Comparison
- Training loss and test accuracy curves for the three RNNs (plus the Transformer's test accuracy)
- `get_hidden_states`: average \|h_t\| over the 20 steps of one test sequence for the NoNorm and LayerNorm RNNs
- Prints the first and last hidden-state magnitudes, and for each RNN the first epoch with test accuracy ≥ 0.99 and the loss after epoch 1

### 10. Batch Size Sensitivity Test
- Retrains the three RNNs for 20 epochs with `batch_size=1` and plots their test accuracy

### 11. Summary
- A printed table of the final test accuracies of all experiments, then `### What the run above shows` with the saved run's real numbers

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `layer_norm_numpy`, `layer_norm_backward_numpy` | NumPy LayerNorm and its gradient check |
| `LayerNorm` | From-scratch PyTorch LayerNorm |
| `BatchNorm1d` | From-scratch BatchNorm for comparison |
| `generate_sequence_data` | Rising vs falling trend sequences |
| `RNNCellWithNorm`, `SimpleRNN` | RNN with pluggable normalization |
| `train_and_evaluate` | Trains one RNN and records its history |
| `SimpleSelfAttention`, `TransformerBlock`, `TransformerClassifier` | Pre-norm Transformer classifier with positional embedding |
| `get_hidden_states` | Hidden-state magnitude over time |

## Data Flow / Shapes

```
RNN:          X (B, 20, 64) → for t in 1..20: h_t = tanh(norm(W_xh x_t + W_hh h_{t−1}))  (B, 64)
              → Linear(64, 2) → logits (B, 2) → cross-entropy
Transformer:  X (B, 20, 64) → Linear + positional embedding (B, 20, 64)
              → x + Attn(LN(x)) → x + FFN(LN(x)) → LN → mean over time (B, 64) → Linear(64, 2)
```

## Deliberate Simplifications vs. the Paper

1. **Tasks:** one synthetic sequence task instead of the paper's experiments (image–sentence ranking, question answering, skip-thought vectors, DRAW on binarized MNIST, handwriting generation, permutation-invariant MNIST).
2. **Model:** a plain tanh RNN with 64 hidden units instead of LSTMs and GRUs.
3. **BatchNorm baseline:** one set of statistics shared across time steps; the paper's recurrent BatchNorm comparison follows Cooijmans et al., who keep separate statistics per time step.
4. **Transformer:** the Transformer came after this paper; the block here only shows where LayerNorm sits in it.
