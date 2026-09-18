# Code Architecture — Highway Networks (009)

## 1. Goal of the notebook
Build a fully-connected Highway layer from raw PyTorch operations, stack 20+ Highway layers, and compare training/gradient behavior against an equally deep plain MLP on MNIST. The notebook produces:
- side-by-side training-curve plots,
- per-layer gradient-norm plots,
- a visualization of transform-gate activity,
- a clean, Colab-runnable, pip-install-only implementation.

## 2. Notebook sections

| Section | What it does |
|---|---|
| **Setup** | Install/update `torch`, `torchvision`, `matplotlib`, `seaborn`, `tqdm` (CPU is enough; GPU optional). |
| **Dataset: MNIST** | `torchvision.datasets.MNIST`, normalize, create train/val dataloaders. |
| **HighwayBlock** | A single Highway layer: `H(x, W_H)` non-linear block + `T(x, W_T)` sigmoid gate; output `y = H * T + x * (1 - T)`. Bias `b_T` initialized to a negative value. |
| **HighwayMLP** | Input projection to hidden size (plain linear layer), stack N `HighwayBlock`s, then output linear + softmax. |
| **PlainMLP** | Matched architecture with the same number of layers/units but standard linear + activation blocks. |
| **Training loop** | Shared `train_epoch`/`evaluate` functions; tracks cross-entropy loss and accuracy; supports both models. |
| **Experiment: depth = 20** | Train both networks for a small number of epochs, plot train/val loss. |
| **Gradient-flow diagnostic** | After a forward/backward pass, record the L2 norm of gradients per layer and visualize them as a heatmap/line plot. |
| **Transform-gate inspection** | For a few test samples, print mean `T(x)` per layer to confirm that gates learn to be selective. |
| **Summary / take-aways** | Highlight that the Highway network trains stably where the plain deep MLP stalls. |

## 3. Key functions/classes

- `HighwayBlock(in_features, activation='relu', gate_bias=-2)`
  - `forward(x) → y`
  - Computes `H = act(x @ W_H^T + b_H)` and `T = σ(x @ W_T^T + b_T)`; returns `H * T + x * (1 - T)`.
- `HighwayMLP(input_size, hidden_size, num_highway_layers, num_classes)`
  - One plain input linear, N `HighwayBlock`s, one output linear.
- `PlainMLP(input_size, hidden_size, num_layers, num_classes)`
  - Standard `Linear → Activation` chain for comparison.
- `train_model(model, loader, optimizer, criterion) → (loss, accuracy)`
- `evaluate_model(model, loader, criterion) → (loss, accuracy)`
- `plot_curves(...)` and `plot_gradient_flow(...)` helpers.

## 4. Data flow and shapes

```
MNIST image [1, 28, 28]
  → flatten → [batch, 784]
  → Linear(784, hidden) + ReLU → [batch, hidden]
  → HighwayBlock x N:
       x: [batch, hidden]
       H: [batch, hidden]
       T: [batch, hidden]
       y = H*T + x*(1-T): [batch, hidden]
  → Linear(hidden, 10) → logits [batch, 10]
  → CrossEntropyLoss
```

## 5. Deliberate simplifications vs. the full paper

- The paper also experiments with convolutional highway layers on CIFAR-10/100; this notebook focuses on fully-connected layers on MNIST to keep the core gating mechanism visible and the runtime short.
- The paper uses up to 900 layers and random-search hyperparameter tuning; the notebook defaults to 20 layers (enough to show the effect) with a small number of epochs.
- CIFAR-100 experiments, maxout baselines, and FitNet comparisons are omitted; only the central optimization claim is reproduced.
- We do not implement the alternative `C(x, W_C)` carry gate separately; we follow the paper’s simplification `C = 1 − T`.
