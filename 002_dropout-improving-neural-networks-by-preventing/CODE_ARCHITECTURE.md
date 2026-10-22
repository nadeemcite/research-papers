---
title: "Dropout: Code architecture"
parent: "002 Dropout"
nav_order: 1
---
# Code Architecture — Dropout (MLP with/without Dropout on Fashion-MNIST)

**Notebook:** `solution.ipynb`
**Task:** Build a small MLP in PyTorch, train it with and without Dropout on Fashion-MNIST, and plot train vs test loss curves side by side to show the overfitting gap closing.

---

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, torchvision, NumPy, and Matplotlib.
- Set a random seed for reproducibility.
- Detect GPU (`cuda`) if available; uses a GPU if available; the Kaggle version runs on CPU.

### 2. Dataset Loading (Fashion-MNIST)
- Use `torchvision.datasets.FashionMNIST` with `torchvision.transforms.ToTensor()`.
- Create train/test `DataLoader`s with moderate batch size (e.g., 128).
- **Data flow:** `images [B, 1, 28, 28]` → flatten to `[B, 784]` → input to MLP.

### 3. Model Definition
- Define a small MLP class with 2 hidden layers (e.g., 784 → 256 → 128 → 10).
- Use ReLU activations.
- Add an optional `dropout_p` argument that controls dropout probability.
- Apply `nn.Dropout(p)` after each ReLU in the trainable version.
- **Key function:** `class MLP(nn.Module)` with forward path `x → fc1 → relu → dropout → fc2 → relu → dropout → fc3`.

### 4. Train/Eval Helpers
- `train_epoch(model, loader, optimizer, criterion, device)` — one training epoch; sets `model.train()` so dropout is active.
- `evaluate(model, loader, criterion, device)` — computes average loss and accuracy; sets `model.eval()` so dropout is disabled.
- **Data flow per batch:**
  - Input: `[B, 784]` → `model` → logits `[B, 10]`
  - Loss: `CrossEntropyLoss(logits, labels)` scalar
  - Backward pass → optimizer step

### 5. Training Two Models
- Create two identical MLPs: `model_no_dropout` with `p=0.0` and `model_dropout` with `p=0.5`.
- Train both for the same number of epochs (25) using the same optimizer and learning rate.
- Record training loss and test loss per epoch for both models.

### 6. Side-by-Side Loss Plot
- Plot train loss and test loss curves for both models on the same axes.
- Include a legend: `No dropout train`, `No dropout test`, `Dropout train`, `Dropout test`.
- The no-dropout model should show a large train/test gap (overfitting), while the dropout model should show a smaller gap.

### 7. Accuracy Comparison
- Print final train/test accuracy for both models.
- There is no confusion matrix or per-class breakdown.

### 8. Dropout Mask Visualization (Optional)
- For a single forward pass in train mode, count the first-layer units that are zero (232 of 256 in the saved run). The notebook only prints this count and does not plot it.
- The count also includes units that ReLU had already set to zero, so it is higher than the 50% dropout rate.

---

## Key Functions/Classes

| Function/Class | Responsibility |
|---|---|
| `get_fashion_mnist_loaders(batch_size)` | Download and prepare train/test DataLoaders |
| `MLP(input_dim, hidden_dims, num_classes, dropout_p)` | Defines the MLP with optional dropout layers |
| `train_epoch(model, loader, optimizer, criterion, device)` | Runs one training epoch, returns average loss and accuracy |
| `evaluate(model, loader, criterion, device)` | Runs inference, returns avg loss and accuracy |
| `plot_losses(hist_no_dropout, hist_dropout)` | Plots train/test curves for both models |

---

## Deliberate Simplifications vs. the Full Paper

| Simplification | Original Paper | Notebook | Why |
|---|---|---|---|
| Dataset | MNIST, TIMIT, Reuters, CIFAR-10, ImageNet | Fashion-MNIST (10 classes, 28×28) | Small, quick, and well-known toy dataset |
| Model | Large conv nets + fully connected | Small 2-hidden-layer MLP | Demonstrates dropout concept without CNN complexity |
| Dropout rate | Keep probability 0.8 (drop 0.2) for inputs, drop 0.5 for hidden units | Drop 0.5 for hidden units, no input dropout | Input dropout less critical for grayscale images; hidden `0.5` is standard |
| Epochs | Trained to convergence | 25 epochs | Enough to show overfitting gap; keeps runtime reasonable |
| Weight scaling | Approximate averaging over the 2^H thinned networks with a "mean network": keep all units and halve their outgoing weights at test time | Inverted dropout: kept units are scaled by 1/(1−p) during training and nothing changes at test time | PyTorch `nn.Dropout` does this automatically; the result is equivalent |
| Tasks | Vision, speech, text | Single classification task | Keeps notebook self-contained and focused |
| Evaluation metric | Benchmark records | Train/test accuracy + loss curves | Sufficient to demonstrate regularization effect |