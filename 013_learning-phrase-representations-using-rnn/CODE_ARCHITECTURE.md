# Code Architecture — Learning Phrase Representations using RNN Encoder–Decoder (GRU)

This notebook implements a **GRU cell from scratch** using raw matrix operations (no `nn.GRU`), builds both a GRU-based and a vanilla-RNN-based recurrent network, trains both on a sequence-memorisation task, and compares them by plotting gradient norms over time steps — directly demonstrating the GRU's advantage in preserving information over long sequences.

## Section 1 — Setup and imports

**What it does:** Installs and imports PyTorch, matplotlib, numpy. Seeds all RNGs for reproducibility.

**Key packages:** `torch`, `torch.nn`, `matplotlib`, `numpy`

## Section 2 — GRU cell from scratch

**What it does:** Implements the exact GRU equations from the paper (Eqs. 5–8) using raw `nn.Linear` layers and element-wise operations — no `nn.GRU` or `nn.GRUCell`.

**Class:** `GRUCellScratch(nn.Module)`
- **Reset gate:** `r = σ(W_r·x + U_r·h_{t-1})` — `nn.Linear` for input projection (`W_r`) and hidden projection (`U_r`), then sigmoid.
- **Update gate:** `z = σ(W_z·x + U_z·h_{t-1})` — same structure, separate parameters.
- **Candidate:** `h̃ = tanh(W·x + U·(r ⊙ h_{t-1}))` — the reset gate gates the previous hidden state before the candidate is computed.
- **New state:** `h = z·h_{t-1} + (1−z)·h̃` — linear interpolation.

**Shapes:**
- Input `x`: `(batch, input_dim)`
- Hidden `h`: `(batch, hidden_dim)`
- All weight matrices: `(hidden_dim, input_dim)` or `(hidden_dim, hidden_dim)`
- Output `h`: `(batch, hidden_dim)`

This is a faithful reproduction of Equations 5–8 from the paper, with the paper's convention that the update gate `z` keeps the old state (i.e. `h = z·h_old + (1−z)·h_new`). Note: some frameworks (e.g. PyTorch's `nn.GRU`) use the opposite convention; we follow the paper exactly.

## Section 3 — Vanilla RNN cell from scratch

**What it does:** Implements a plain tanh RNN cell for comparison.

**Class:** `VanillaRNNCell(nn.Module)`
- `h = tanh(W·x + U·h_{t-1} + b)` — the simplest possible recurrent unit with no gating.

## Section 4 — Full RNN wrapper

**What it does:** Wraps a cell (GRU or vanilla) into a full sequence-processing network that unrolls over time steps.

**Class:** `RecurrentNetwork(nn.Module)`
- Takes a cell and processes a sequence of length T, maintaining a running hidden state.
- Input: `(batch, seq_len, input_dim)` → Output: `(batch, seq_len, hidden_dim)`
- At each time step t: `h_t = cell(x_t, h_{t-1})`
- Stores all intermediate hidden states for gradient analysis.

## Section 5 — Sequence memorisation task

**What it does:** Creates a synthetic task where the model must reproduce a sequence of random values after reading it — a pure memorisation task that stresses long-range memory.

**Task design:**
- Input: a sequence of T random vectors (each a one-hot or continuous vector).
- Target: the same sequence (copy task) — the model must output x_t at position t after reading the full sequence.
- This task directly tests whether the recurrent unit can preserve information across T time steps.

**Key function:** `generate_memorisation_data(n_samples, seq_len, input_dim)`

## Section 6 — Training loop

**What it does:** Trains both the GRU-based and vanilla-RNN-based networks on the memorisation task with the same hyperparameters, using MSE loss and Adam optimiser.

**Key variables:**
- `SEQ_LEN`: sequence length (varied: 5, 10, 20, 40)
- `HIDDEN_DIM`: 64
- `INPUT_DIM`: 16
- `LR`: 0.01
- `NUM_EPOCHS`: 200

**Key function:** `train_model(cell_type, seq_len, epochs)`

## Section 7 — Gradient norm analysis

**What it does:** After training, runs a forward pass and backpropagates loss to the first time step. Measures the gradient norm of the loss with respect to the hidden state at each time step — this is the classic vanishing/exploding gradient diagnostic.

**Why this matters:** The paper's motivation for the GRU was precisely to combat the vanishing gradient problem in plain RNNs. By plotting ‖∂L/∂h_t‖ vs. time step t for both architectures, we see that:
- **Vanilla RNN:** gradients shrink exponentially with time step (vanishing gradient).
- **GRU:** gradients remain much more stable across time steps thanks to the update gate's linear interpolation (gradient highway).

**Key function:** `compute_gradient_norms(model, seq_len)`

## Section 8 — Comparison plots

### 8a — Loss convergence
Plots training loss vs. epoch for GRU and vanilla RNN at different sequence lengths, showing the GRU converges faster and to lower loss especially for longer sequences.

### 8b — Gradient norm vs. time step
Plots ‖∂L/∂h_t‖ for each time step t, on a log scale, for both architectures. This is the key figure demonstrating the GRU's resistance to vanishing gradients.

### 8c — Accuracy by sequence length
Bar chart comparing final memorisation accuracy for GRU vs. vanilla RNN at sequence lengths 5, 10, 20, 40.

## Data flow summary

```
random sequence (batch, seq_len, input_dim)
         ↓
  ┌──────┴──────┐
  │ GRU cell    │  vs.  │ Vanilla RNN cell │
  │ unrolled T  │       │ unrolled T        │
  │ times       │       │ times             │
  └──────┬──────┘       └───────┬───────────┘
         ↓                       ↓
  hidden states h_1...h_T   hidden states h_1...h_T
         ↓                       ↓
  output projection         output projection
         ↓                       ↓
  MSE loss vs. target       MSE loss vs. target
         ↓                       ↓
  backprop → gradient       backprop → gradient
  norms per time step       norms per time step
         ↓                       ↓
  ┌──────┴──────┐
  │  Plot:      │
  │  loss curve │
  │  grad norms │
  │  accuracy   │
  └─────────────┘
```

## Deliberate simplifications vs. full paper

| Paper | Notebook |
|---|---|
| English→French SMT, 348M words, 15k vocab | Synthetic sequence memorisation task |
| 1000 hidden units, rank-100 embeddings | 64 hidden units, 16-dim inputs |
| Encoder–Decoder for phrase-pair scoring | Standalone GRU vs. vanilla RNN comparison |
| Adadelta optimizer | Adam optimizer |
| BLEU evaluation on WMT'14 | MSE loss + gradient norm analysis |
| Trained for days on GPU | ~3 minutes on Kaggle GPU |

The notebook isolates the paper's core architectural contribution (the GRU cell with reset/update gates) and demonstrates its advantage over a vanilla RNN through gradient analysis — the exact motivation given in Section 2.3 of the paper. The full encoder–decoder and SMT integration are abstracted away to keep the notebook self-contained and fast.
