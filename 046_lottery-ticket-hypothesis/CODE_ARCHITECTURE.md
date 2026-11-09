---
title: "Lottery Ticket: Code architecture"
parent: "046 Lottery Ticket"
nav_order: 1
---
# Code Architecture — Lottery Ticket Hypothesis Notebook

## Overview

The notebook implements **Iterative Magnitude Pruning (IMP)** from scratch on a small MLP (a fully-connected network) on MNIST. It then compares the winning ticket (original initialization + pruning mask) against a re-initialized subnetwork (new random weights + same mask) and against the unpruned network, and reports what that one run found.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision (MNIST), matplotlib, numpy
- Reproducibility: `SEED = 42` for the torch and NumPy random generators
- Figures are shown as JPEG at 80 dpi to keep the saved notebook small
- On Kaggle, an `atexit` hook copies the executed notebook to `executed_solution.ipynb` so its outputs can be downloaded
- The saved run used the CPU (`Device: cpu`)

### 2. Data Loading
- MNIST via torchvision, downloaded quietly to a temp folder (needs Kaggle internet on)
- `batch_size=128` for training, 256 for testing
- Train/test split as provided by torchvision (60,000 / 10,000)

### 3. Model Definition — `SimpleMLP`
- Architecture: `784 → 256 → 128 → 10` (fully-connected), ReLU between hidden layers
- 234,752 prunable weights (biases are not pruned)

### 4. `train_model(model, mask, train_loader, test_loader, epochs, lr)`
- Accepts an optional `mask` (dict of per-layer 0/1 tensors) that zeros pruned weights' gradients and re-zeros the weights after each step
- Cross-entropy loss, Adam optimizer (`lr=1e-3`)
- Returns the training history (loss and test accuracy per epoch)

### 5. `prune_by_magnitude(model, prune_fraction, current_mask)`
- Collects the absolute values of the **surviving** weights across all layers (global pruning)
- Finds the threshold at the `prune_fraction` quantile of those survivors
- Returns the new mask, which is the old mask minus the weights below the threshold, and the overall sparsity

Leaving the already-pruned weights out matters. They sit at exactly zero, so a quantile over all weights would return a threshold of 0 from round 2 onward and nothing more would be pruned.

### 6. `apply_mask_to_model(model, mask)`
- Zeroes out pruned weights in the model parameters

### 7. Iterative Magnitude Pruning Loop
```
init = copy of all parameters at the start
for round in 0..5:
    1. Train the model with the current mask (5 epochs)
    2. Log: sparsity and final test accuracy
    3. If round < 5: prune 20% of the remaining weights → new mask
    4. Reset every parameter to init; pruned weights stay at zero
```
- `num_iterations=5`, `prune_fraction=0.20`, `epochs_per_iter=5`
- Sparsity after each pruning: 20.0%, 36.0%, 48.8%, 59.0%, 67.2%

### 8. Winning Ticket Evaluation
- Takes the final mask (67.2% of weights removed) and trains three models for 15 epochs:
- **Winning Ticket:** surviving weights reset to their original values
- **Random Re-init:** same mask, new random weights from a different seed
- **Full Model:** no pruning, as a reference

### 9. Visualization
- **Plot 1:** Accuracy vs. sparsity across the IMP rounds
- **Plot 2:** Test accuracy and training loss per epoch for the three 15-epoch runs

### 10. Results and Summary
- A printed table of the numbers above, then a `### What the run above shows` cell written from them

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `SimpleMLP` | 3-layer MLP (784→256→128→10) |
| `train_model(...)` | Trains with an optional pruning mask |
| `evaluate(...)` | Test-set accuracy |
| `prune_by_magnitude(...)` | Global magnitude pruning of the surviving weights |
| `apply_mask_to_model(...)` | Zeroes pruned weights |
| `save_initial_weights(...)` | Copies every parameter's starting value |
| `reset_to_initial_weights(...)` | Restores starting values; pruned weights stay zero |
| `iterative_magnitude_pruning(...)` | Full IMP loop |
| `train_with_mask_and_init(...)` | Trains a fresh model from the original or a new initialization |

## Data Flow / Shapes

- Input: `(batch, 1, 28, 28)` → flattened to `(batch, 784)`
- FC1: `(batch, 784)` → `(batch, 256)`
- FC2: `(batch, 256)` → `(batch, 128)`
- FC3: `(batch, 128)` → `(batch, 10)` logits

## Deliberate Simplifications vs. Full Paper

1. **Smaller MLP, MNIST only:** the paper's MNIST model is Lenet-300-100, and it also runs convolutional networks, VGG-19 and Resnet-18 on CIFAR-10.
2. **5 pruning rounds, down to 67.2% sparsity:** the paper keeps pruning far past this, and its gap between winning tickets and random re-init shows most at heavier pruning.
3. **Global pruning at one rate:** for Lenet the paper prunes each layer separately and prunes the connections to the outputs at half the rate. It uses global pruning only for its deeper networks.
4. **One seed:** the paper averages five trials per point, with three random re-initializations of each ticket.
5. **Same optimizer family, different settings:** the paper's Lenet uses Adam at 1.2e-3; this notebook uses Adam at 1e-3.
6. **Fixed epochs, no early stopping:** the paper tracks the iteration of minimum validation loss. This notebook only reports test accuracy per epoch.
