---
title: "Adam: Code architecture"
parent: "003 Adam"
nav_order: 1
---
# CODE_ARCHITECTURE.md — Adam Optimizer From Scratch

## What the notebook builds

A small, self-contained demo that implements the SGD, RMSProp and Adam update rules in pure NumPy, runs all three on a 2-D tilted quadratic valley from the same starting point, and visualizes their paths as a contour plot plus an animated GIF of Adam's trajectory. Gradients are exact (no noise), so every run is deterministic and the whole notebook finishes in seconds on CPU.

## Notebook sections (in order)

### 1. Imports and reproducibility
- `numpy`, `matplotlib`, `PIL.Image` and `pathlib`. Seeds NumPy with 42 (not strictly needed, since nothing is random).
- Creates `/tmp/adam_frames` as scratch space for the plot and animation frames.
- On Kaggle only, registers an exit hook that copies the executed notebook (with outputs) to `/kaggle/working/executed_solution.ipynb`, so it can be downloaded with `kaggle kernels output`.

### 2. Toy 2-D loss surface and gradient
- Loss: `L(x, y) = (x − 1)² + 5·(y − 0.5x)²`, a valley tilted along the line `y = 0.5x`, with its minimum `L = 0` at `(1, 0.5)`. The curvature across the valley is much larger than along it.
- `loss_fn(theta)` returns the scalar loss; `grad_fn(theta)` returns the analytic gradient `[2(x − 1) − 5(y − 0.5x), 10(y − 0.5x)]`.
- A 200×200 grid over `[−3, 3]²` (`X`, `Y`, `Z`) is precomputed for the contour plots.

### 3. Optimizer implementations from scratch
The update equations are written out in LaTeX, then implemented as one function per optimizer. Each step function calls `grad_fn` itself and returns the new `theta`; RMSProp and Adam keep their running averages in a `state` dict.
- `sgd_step(theta, lr)`: `θ ← θ − lr·g`.
- `rmsprop_step(theta, lr, state, beta=0.9, eps=1e-8)`: `v ← β·v + (1 − β)·g²`, then `θ ← θ − lr·g / (√v + ε)`.
- `adam_step(theta, lr, state, t, beta1=0.9, beta2=0.999, eps=1e-8)`: updates `m` and `v`, bias-corrects them with `1 − β1^t` and `1 − β2^t`, then `θ ← θ − lr·m̂ / (√v̂ + ε)`.

### 4. Training loop
- `optimize(opt_name, n_iter=200, init=None)` starts at `[−2, 2]`, runs 200 steps of the chosen optimizer, and returns the trajectory `(201, 2)` and loss history `(201,)`.
- Learning rates: **SGD 0.02**, **RMSProp 0.05** (`β = 0.9`), **Adam 0.05** (`β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`).
- Final losses after 200 steps: SGD 9.1e-6, RMSProp 6.7e-3, Adam 3.7e-2.

### 5. Static contour plot of trajectories
- Filled log-spaced contours of the loss with the three paths overlaid (SGD red, RMSProp cyan, Adam yellow), the start point and the minimum. Saved as `trajectories.png` and shown inline.

### 6. Animation of the Adam trajectory
- Renders one Matplotlib frame per Adam step (201 PNGs), keeps every second frame (101 frames), and assembles them into `adam_trajectory.gif` with Pillow (`duration=80` ms, looping). The GIF is embedded inline with `IPython.display.Image`.

### 7. Summary
- Prints the start point, end point and final loss for each optimizer.
- A closing markdown cell, *What the run above shows*, tabulates the results and explains them: on this noise-free toy, tuned SGD wins, RMSProp bounces around the minimum, and Adam is still converging after 200 steps.

## Data flow and shapes

- `theta`, `g`, `m`, `v`, `m̂`, `v̂`: all shape `(2,)`.
- Trajectory: `(n_iter + 1, 2)` = `(201, 2)`; loss history: `(201,)`.
- Contour grid `X`, `Y`, `Z`: `(200, 200)`.

## Deliberate simplifications vs. the full paper

- **2-D toy loss** instead of the paper's logistic regression, multilayer networks and CNNs on MNIST, IMDB and CIFAR-10, so the paths are easy to see.
- **Exact, noise-free gradients** instead of stochastic mini-batch gradients. This removes the noisy, high-dimensional, badly scaled setting Adam was designed for, which is why tuned SGD wins here.
- **Hand-picked learning rates** (0.02 for SGD, 0.05 for RMSProp and Adam) instead of a hyperparameter sweep. Adam's `lr` is larger than the paper's default of 0.001 because the problem is tiny.
- **No AdaMax, no weight decay or AdamW, no learning-rate schedule.** These are discussed in the README and TALK but not implemented.
