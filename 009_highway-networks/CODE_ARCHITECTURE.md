---
title: "Highway Nets: Code architecture"
parent: "009 Highway Nets"
nav_order: 1
---
# Code Architecture — Highway Networks (009)

## 1. Goal of the notebook
Build a fully-connected Highway layer from raw PyTorch operations, stack 20 Highway layers, and compare training/gradient behavior against an equally deep plain MLP on MNIST (hidden width 64, 10 epochs, Adam with learning rate 1e-3, gate bias −2). The notebook produces:
- inline training-loss and validation-accuracy plots for both networks,
- a per-tensor gradient-norm line plot (log scale),
- a bar chart of mean transform-gate activity per layer.

## 2. Notebook sections

| Section | What it does |
|---|---|
| **Setup** | Imports; picks the GPU if available (CPU is enough). |
| **Dataset: MNIST** | `torchvision.datasets.MNIST`, normalize, create train/val dataloaders (the download progress bars are silenced). |
| **HighwayBlock** | A single Highway layer: `H(x, W_H)` non-linear block + `T(x, W_T)` sigmoid gate; output `y = H * T + x * (1 - T)`. Bias `b_T` initialized to a negative value. |
| **HighwayMLP** | Input projection to hidden size (plain linear layer), stack N `HighwayBlock`s, then an output linear layer that returns logits (the softmax lives inside `CrossEntropyLoss`). |
| **PlainMLP** | Matched architecture with the same number of layers/units but standard linear + activation blocks. |
| **Training loop** | Shared `train_epoch`/`evaluate` functions; tracks cross-entropy loss and accuracy; supports both models. |
| **Experiment: depth = 20** | Train both networks for 10 epochs, then plot training loss and validation accuracy inline. |
| **Gradient-flow diagnostic** | After a forward/backward pass, record the L2 norm of the gradient of every weight tensor and show them as a log-scale line plot. |
| **Transform-gate inspection** | For a few test samples, print mean `T(x)` per layer to confirm that gates learn to be selective. |
| **Summary / take-aways** | Highway reaches 97.2% validation accuracy vs 95.6% for the plain MLP; the plain MLP starts at chance (11.4%) and is still below 40% after epoch 2 before catching up. It did not stay stalled. |

## 3. Key functions/classes

- `HighwayBlock(in_features, activation='relu', gate_bias=-2)`
  - `forward(x) → y`
  - Computes `H = act(x @ W_H^T + b_H)` and `T = σ(x @ W_T^T + b_T)`; returns `H * T + x * (1 - T)`.
- `HighwayMLP(input_size, hidden_size, num_highway_layers, num_classes)`
  - One plain input linear, N `HighwayBlock`s, one output linear.
- `PlainMLP(input_size, hidden_size, num_layers, num_classes)`
  - Standard `Linear → Activation` chain for comparison.
- `train_epoch(model, loader, optimizer, criterion) → (loss, accuracy)`
- `evaluate(model, loader, criterion) → (loss, accuracy)`
- `run_experiment(depth, hidden, epochs, lr, gate_bias)` trains both models and returns them with their histories.
- `gradient_flow(model, loader, criterion, name)` and `collect_gate_activity(model, loader)` are the diagnostics. The plots are inline cells, not helper functions.

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
  → CrossEntropyLoss (applies log-softmax internally)
```

## 5. Deliberate simplifications vs. the full paper

- The paper also experiments with convolutional highway layers on CIFAR-10/100; this notebook focuses on fully-connected layers on MNIST to keep the core gating mechanism visible and the runtime short.
- The paper's experiments go up to 100 layers (plus a preliminary 900-layer run) with random-search hyperparameter tuning; the notebook uses 20 layers, hidden width 64, 10 epochs and Adam at 1e-3.
- The plain MLP is not parameter-matched to the Highway MLP (each Highway block has two weight matrices), so the comparison is illustrative.
- CIFAR-100 experiments, maxout baselines, and FitNet comparisons are omitted; only the central optimization claim is reproduced.
- We do not implement the alternative `C(x, W_C)` carry gate separately; we follow the paper’s simplification `C = 1 − T`.
