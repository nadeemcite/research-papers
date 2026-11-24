---
title: "GRU Enc-Dec: Code architecture"
parent: "013 GRU Enc-Dec"
nav_order: 1
---
# Code Architecture — Learning Phrase Representations using RNN Encoder–Decoder (GRU)

## Overview

The notebook implements the paper's gated hidden unit (the GRU) from scratch with raw `nn.Linear` layers (no `nn.GRU`) and compares it with a plain tanh RNN on a delayed-recall task: show 3 random vectors, wait D blank steps, then output the 3 vectors. Both networks are trained for delays of 5, 10, 20 and 40 steps, evaluated on new test sequences, and analysed through the gradient of the recall loss with respect to each hidden state and the GRU's gate activations. Everything runs on the CPU.

## Section-by-Section Breakdown

### Setup
- PyTorch, NumPy, matplotlib; nothing is installed
- Seeds 42; one CPU thread and `torch.use_deterministic_algorithms(True)` so reruns give the same numbers
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 1. GRU Cell from Scratch
- `GRUCellScratch(input_dim, hidden_dim)` with separate `W_r, U_r`, `W_z, U_z`, `W, U` layers
- Reset gate `r = σ(W_r x + U_r h_{t−1})` (Eq. 5), update gate `z = σ(W_z x + U_z h_{t−1})` (Eq. 6), candidate `h̃ = tanh(W x + U (r ⊙ h_{t−1}))` (Eq. 8), new state `h_t = z ⊙ h_{t−1} + (1 − z) ⊙ h̃` (Eq. 7)
- As in the paper, z close to 1 keeps the previous state

### 2. Plain RNN Cell from Scratch
- `VanillaRNNCell`: `h_t = tanh(W x_t + U h_{t−1})`

### 3. Full Recurrent Network Wrapper
- `RecurrentNetwork(cell, input_dim, hidden_dim, output_dim)`: unrolls the cell from `h_0 = 0`, applies `Linear(64, 8)` at every step, and returns all outputs and all hidden states (for the gradient analysis)

### 4. Delayed-Recall Task
- `K = 3`, `DATA_DIM = 8`, `INPUT_DIM = 9` (8 data channels + 1 recall cue), `HIDDEN_DIM = 64`, `OUTPUT_DIM = 8`
- `BATCH_SIZE = 128`, `NUM_EPOCHS = 300`, `LR = 0.002`, `N_TRAIN = 1000`, `N_TEST = 500`, `DELAYS = [5, 10, 20, 40]`
- `generate_recall_data(n, delay, generator)`: sequences of length T = 3 + delay + 3; the first 3 steps carry random N(0, 1) vectors, the last 3 steps have the recall cue set and the 3 vectors as targets; returns inputs, targets and a mask of the recall steps
- `recall_mse(outputs, y, mask)`: mean squared error on the recall steps only (about 1.0 for a model that outputs zeros)

### 5. Training Loop
- `build_model(cell_type)` and `train_model(cell_type, delay)`: Adam, gradient-norm clipping at 5, 300 epochs on 1,000 training sequences; prints the training recall MSE every 100 epochs and returns the model, the loss history and the recall MSE on 500 test sequences
- Trains a GRU and a plain RNN for each delay (8 models)

### 6. Training Loss Comparison
- Training recall MSE per epoch for both models at each delay (log scale)

### 7. Gradient Norm Analysis
- `compute_gradient_norms(model, delay)`: on 64 new sequences, the L2 norm of ∂L/∂h_t for every step t, where L is the recall loss; prints the norm at t = 1 and t = T and their ratio

### 8. Gradient Norm Plots
- ‖∂L/∂h_t‖ against t (log scale) for both models at each delay, with the recall steps shaded

### 9. Test Recall MSE by Delay
- Bar chart of the test recall MSE for both models at each delay

### 10. Summary Table
- Test MSE, gradient at t = 1, gradient at t = T and their ratio for all 8 models

### 11. GRU Gate Activations
- `extract_gate_activations(model, delay)`: reset and update gates of the delay-40 GRU over one test sequence; prints their mean over the show, blank and recall steps and plots them as heatmaps

### 12. Summary
- `### What the run above shows` with the saved run's real numbers

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `GRUCellScratch` | The paper's gated unit, Eqs. 5–8 |
| `VanillaRNNCell` | Plain tanh RNN baseline |
| `RecurrentNetwork` | Unrolls a cell and keeps all hidden states |
| `generate_recall_data`, `recall_mse` | Delayed-recall data and loss on recall steps only |
| `train_model` | Trains one model and returns its test recall MSE |
| `compute_gradient_norms` | ‖∂L/∂h_t‖ for every time step |
| `extract_gate_activations` | Reset and update gate values over a sequence |

## Data Flow / Shapes

```
x (batch, T, 9) with T = 3 + delay + 3: [3 data steps | delay blank steps | 3 recall-cue steps]
→ cell unrolled T times → h_t (batch, 64) for t = 1..T → Linear(64, 8) → outputs (batch, T, 8)
→ recall MSE on the last 3 steps vs the 3 data vectors
→ backprop: ∂L/∂h_t for every t (gradient analysis)
```

## Deliberate Simplifications vs. the Paper

1. **Task:** a synthetic delayed-recall task instead of English→French phrase scoring on WMT'14.
2. **Model:** a single recurrent network, not the paper's encoder–decoder; it isolates the gated hidden unit.
3. **Size:** 64 hidden units and 8-dimensional inputs instead of 1,000 hidden units and 100-dimensional word embeddings.
4. **Training:** Adam on 1,000 sequences for 300 epochs instead of Adadelta for about three days.
5. **Evaluation:** recall MSE and gradient norms instead of BLEU.
