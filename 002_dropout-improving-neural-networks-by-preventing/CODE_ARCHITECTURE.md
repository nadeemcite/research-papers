# Code Architecture — Dropout (MLP with/without Dropout on Fashion-MNIST)

**Notebook:** `solution.ipynb`
**Task:** Build a small MLP in PyTorch, train it with and without Dropout on Fashion-MNIST, and plot train vs test loss curves side by side to show the overfitting gap closing.

---

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, torchvision, NumPy, and Matplotlib.
- Set a random seed for reproducibility.
- Detect GPU (`cuda`) if available; Kaggle kernels provide a GPU runtime for this notebook.

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
- Train both for the same number of epochs (e.g., 20–30) using the same optimizer and learning rate.
- Record training loss and test loss per epoch for both models.

### 6. Side-by-Side Loss Plot
- Plot train loss and test loss curves for both models on the same axes.
- Include a legend: `No dropout train`, `No dropout test`, `Dropout train`, `Dropout test`.
- The no-dropout model should show a large train/test gap (overfitting), while the dropout model should show a smaller gap.

### 7. Accuracy Comparison
- Print final train/test accuracy for both models.
- Optionally show a small confusion matrix or per-class accuracy for the dropout model.

### 8. Dropout Mask Visualization (Optional)
- For a single forward pass in train mode, capture and visualize which neurons were dropped.
- This is purely illustrative; it is not required by the code template but helps explain the method.

---

## Key Functions/Classes

| Function/Class | Responsibility |
|---|---|
| `get_fashion_mnist_loaders(batch_size)` | Download and prepare train/test DataLoaders |
| `MLP(input_dim, hidden_dims, num_classes, dropout_p)` | Defines the MLP with optional dropout layers |
| `train_epoch(model, loader, optimizer, criterion, device)` | Runs one training epoch, returns avg loss |
| `evaluate(model, loader, criterion, device)` | Runs inference, returns avg loss and accuracy |
| `plot_losses(history)` | Plots train/test curves for both models |

---

## Deliberate Simplifications vs. the Full Paper

| Simplification | Original Paper | Notebook | Why |
|---|---|---|---|
| Dataset | MNIST, CIFAR-10, ImageNet, speech | Fashion-MNIST (10 classes, 28×28) | Small, quick, and well-known toy dataset |
| Model | Large conv nets + fully connected | Small 2-hidden-layer MLP | Demonstrates dropout concept without CNN complexity |
| Dropout rate | Input `0.8`, hidden `0.5` tested | Hidden `0.5`, input `0.0` | Input dropout less critical for grayscale images; hidden `0.5` is standard |
| Epochs | Trained to convergence | 20–30 epochs | Enough to show overfitting gap; keeps runtime reasonable |
| Weight scaling | Exact averaging over 2^H networks | Multiply weights by `p` at test time | Standard PyTorch `nn.Dropout` already handles this automatically |
| Tasks | Vision, speech, bioinformatics | Single classification task | Keeps notebook self-contained and focused |
| Evaluation metric | Benchmark records | Train/test accuracy + loss curves | Sufficient to demonstrate regularization effect |