# CODE_ARCHITECTURE.md — Adam Optimizer From Scratch

## What the notebook builds

A self-contained toy demo that implements the Adam update rule in pure NumPy, then compares it against vanilla SGD and RMSProp on a simple 2-D loss surface. The final cells animate or plot the optimization trajectories as contour plots.

## Notebook sections (in order)

### 1. Setup and imports
- `numpy`, `matplotlib`, `matplotlib.animation` (for the contour animation), and optional `IPython.display` for showing the animation inline.

### 2. Define the toy loss surface
- A 2-D quadratic bowl `f(x, y) = 0.5·(a·x² + b·y²)` with different curvatures along each axis (e.g. `a=1`, `b=20`).
- The asymmetric curvature makes global learning-rate tuning painful and highlights per-parameter scaling.
- Helper functions:
  - `loss_surface(X, Y)` — returns grid of loss values for contour plotting.
  - `gradient(params)` — returns the analytical gradient vector `∇f(params)`.

### 3. Vanilla SGD baseline
- Update rule: `θ ← θ − α·g`.
- Hyperparameters: `lr = 0.02` or similar; runs for a fixed number of steps.
- Records `history` array of shape `(steps, 2)` for plotting.

### 4. RMSProp baseline
- Maintains a moving average of squared gradients: `v_t = β2·v_{t-1} + (1−β2)·g_t²`.
- Update: `θ ← θ − α · g / (√v_t + ε)`.
- Uses the same `β2 = 0.999`, `ε = 1e-8` as Adam for fair comparison.

### 5. Adam optimizer from scratch
Implemented as a small class or a loop with explicit state:
- State variables:
  - `m` — biased first-moment estimate, shape `(2,)`.
  - `v` — biased second-moment estimate, shape `(2,)`.
  - `t` — step counter starting from 1.
- Update equations (elementwise):
  - `m = β1*m + (1−β1)*g`
  - `v = β2*v + (1−β2)*g*g`
  - `m_hat = m / (1 − β1**t)`
  - `v_hat = v / (1 − β2**t)`
  - `theta = theta − α * m_hat / (sqrt(v_hat) + ε)`
- Hyperparameters: `α=0.1` (larger than default because the problem is tiny and noise-free), `β1=0.9`, `β2=0.999`, `ε=1e-8`.

### 6. Run all three optimizers
- Each optimizer starts from the same initial point far from the optimum.
- Each stores its parameter trajectory and loss history.

### 7. Plotting: static contour + trajectories
- A single matplotlib figure showing the loss surface as filled contours.
- Overlay the three trajectories with different colors/markers.
- Print final loss and number of steps for each.

### 8. Animation (optional but recommended)
- `matplotlib.animation.FuncAnimation` that shows one frame per optimizer step.
- Draws the same contour background, then animates the moving point for one optimizer at a time.
- Saved as an HTML5 video or GIF and displayed inline.

## Key functions / classes

- `gradient(params)` → shape `(2,)`  
  Returns `[a*x, b*y]` for the chosen quadratic bowl.
- `run_sgd(theta0, lr, steps)` → `(history, losses)`  
  Pure NumPy loop recording the SGD trajectory.
- `run_rmsprop(theta0, lr, beta2, eps, steps)` → `(history, losses)`  
  RMSProp loop.
- `run_adam(theta0, lr, beta1, beta2, eps, steps)` → `(history, losses)`  
  Full Adam with bias correction.
- `plot_trajectories(...)` / `animate_optimizer(...)`  
  Visualization helpers.

## Data flow and shapes

- Input: initial point `theta0` shape `(2,)`.
- Per step:
  - `g` shape `(2,)` → `m`, `v` shape `(2,)`.
  - Bias-corrected `m_hat`, `v_hat` shape `(2,)`.
  - Update `theta` shape `(2,)`.
- History arrays: `(steps, 2)`.
- Loss history: `(steps,)`.

## Deliberate simplifications vs. the full paper

- **2-D toy loss** instead of high-dimensional ML objectives. This makes the contours and trajectories easy to visualize.
- **Noise-free, deterministic gradients** instead of stochastic mini-batch gradients. Keeps the animation stable and the comparison clean.
- **A single fixed hyperparameter set** for each optimizer. The paper sweeps many settings on real models; here we pick sensible defaults that show the qualitative differences.
- **No weight decay / decoupling**. AdamW is discussed in the README/TALK but not implemented; adding it would complicate the demo without changing the core idea.

The notebook therefore captures the *essence* of Adam — momentum + per-parameter adaptive scaling + bias correction — without reproducing the full convergence experiments on ImageNet or language models.
