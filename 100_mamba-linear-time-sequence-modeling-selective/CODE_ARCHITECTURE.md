---
title: "Mamba: Code architecture"
parent: "100 Mamba"
nav_order: 1
---
# Code Architecture — Mamba Selective State-Space Model Notebook

## Overview

The notebook implements a simplified **selective state-space model (S6)** from scratch in PyTorch, trains it on toy sequence tasks, and benchmarks its speed/memory against a vanilla Transformer at increasing sequence lengths.

## Section-by-Section Breakdown

### 1. Imports and Setup
- `torch`, `torch.nn`, `torch.nn.functional`, `math`, `time`, `functools`
- Device selection (cuda if available, else cpu)
- Reproducibility: `torch.manual_seed(42)`

### 2. State Space Model (S4) — Baseline LTI
**Class: `S4Layer`**
- Parameters: `A_log` (logarithm of diagonal A, shape `(D, N)`), `B` (D, N), `C` (D, N), `D` (D,), `dt` (delta, shape (D,))
- `_init_A` uses HiPPO-inspired diagonal initialization: `A[n] = -(n+1)` for real case (S4D-Real)
- `forward(x)`: x has shape `(B, L, D)`
  - Discretize via ZOH: `A_bar = exp(dt * A)`, `B_bar = (dt * A)^{-1} (exp(dt*A) - I) * dt * B`
  - Since A is diagonal, this simplifies to element-wise operations
  - Compute via recurrence: `h_t = A_bar * h_{t-1} + B_bar * x_t`, `y_t = C * h_t`
  - LTI: B, C, dt are fixed (not input-dependent)
- Output shape: `(B, L, D)`

### 3. Selective State Space Model (S6) — Mamba's Core
**Class: `SelectiveSSM`**
- Parameters: `A_log` (D, N) — fixed, learnable, HiPPO-initialized
- Input-dependent projections: `x_proj` (D → 2*N + D) producing B_t, C_t, and delta_t for each timestep
- `forward(x)`: x has shape `(B, L, D)`
  - For each timestep t:
    - `B_t = x_proj_B(x[:, t])`  → shape (B, N)
    - `C_t = x_proj_C(x[:, t])`  → shape (B, N)
    - `delta_t = softplus(x_proj_delta(x[:, t]))`  → shape (B, D)
    - Discretize: `A_bar = exp(delta_t * A)`, `B_bar = delta_t * B_t` (simplified ZOH for diagonal A)
    - `h_t = A_bar * h_{t-1} + B_bar * x[:, t]`
    - `y_t = C_t * h_t`  → shape (B, D)
  - Returns y of shape (B, L, D)
- **Key difference from S4:** B, C, delta change at every timestep based on input → selection

**Class: `MambaBlock`**
- Input projection: `in_proj` (D → 2*D) for gated branch and main branch
- Conv1d (kernel_size=4) on the main branch
- SiLU activation on the gate branch
- `SelectiveSSM` on the main branch
- Multiplicative gating: `output = gate * ssm_output`
- Output projection: `out_proj` (D → D)
- Residual connection: `output + residual`

### 4. Full Mamba Model
**Class: `MambaLM`**
- Token embedding: `nn.Embedding(vocab_size, D)`
- Stack of `n_layers` MambaBlocks
- Output head: `nn.Linear(D, vocab_size)`
- `forward(x)`: x is token indices (B, L) → logits (B, L, vocab_size)
- Loss: cross-entropy

### 5. Baseline Transformer
**Class: `TransformerBlock`**
- Multi-head self-attention (4 heads)
- Feed-forward (D → 4D → D with GELU)
- LayerNorm pre-norm
- `TransformerLM`: embedding + n_layers blocks + output head

### 6. Toy Task: Selective Copy
- Generate sequences of length L with random tokens from vocab
- Place a few "marker" tokens at random positions
- Target: copy the markers to the end (or reconstruct the marked positions)
- This tests content-based selection — the model must identify which tokens are special

### 7. Toy Task: Induction Heads
- Generate sequences: [random tokens] + [anchor token] + [random tokens] + [anchor token] + [?]
- Target: the model must output the token that followed the first anchor
- This tests in-context retrieval — requires selective recall

### 8. Training Loop
- Adam optimizer, lr=1e-3
- Train both Mamba and Transformer on each toy task
- 50-100 epochs, batch size 32
- Track and plot training loss

### 9. Speed and Memory Benchmark
- For sequence lengths [256, 512, 1024, 2048, 4096]:
  - Measure forward+backward time (averaged over 10 runs)
  - Measure peak GPU memory with `torch.cuda.max_memory_allocated()`
  - Compare Mamba vs Transformer
- Plot: time vs sequence length, memory vs sequence length
- Expected: Mamba scales linearly, Transformer scales quadratically

### 10. Visualization and Conclusions
- Plot training loss curves (Mamba vs Transformer on each task)
- Plot speed benchmark (log-log scale)
- Plot memory benchmark
- Summary table of results

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `S4Layer` | Baseline LTI state-space model (fixed parameters) |
| `SelectiveSSM` | Core S6: input-dependent B, C, Δ with recurrence |
| `MambaBlock` | Full Mamba block: proj → conv → SiLU → SSM → gate → proj |
| `MambaLM` | Language model: embedding + Mamba blocks + output head |
| `TransformerBlock` | Baseline: multi-head attention + FFN |
| `TransformerLM` | Baseline language model |
| `selective_copy_task` | Generate toy data for selective copy |
| `induction_head_task` | Generate toy data for induction heads |
| `benchmark_model` | Measure time and memory at varying seq lengths |

## Data Flow and Shapes

```
Input tokens: (B, L) — integer indices
  → Embedding: (B, L, D)
  → MambaBlock:
      → in_proj: (B, L, 2D) → split into gate (B,L,D) and main (B,L,D)
      → Conv1d on main: (B, L, D)
      → SiLU on gate: (B, L, D)
      → SelectiveSSM on main: (B, L, D)  [internal state: (B, L, N)]
      → gate * ssm_out: (B, L, D)
      → out_proj: (B, L, D)
      → + residual: (B, L, D)
  → [repeat for n_layers]
  → Output head: (B, L, vocab_size)
```

## Deliberate Simplifications vs Full Paper

1. **Naive recurrence loop:** We implement the SSM recurrence with a Python for-loop over timesteps. The real Mamba uses a hardware-aware fused CUDA parallel scan. This is O(BLDN) and slower on GPU but demonstrates the algorithm clearly.
2. **Diagonal A only:** We use a diagonal A matrix (S4D-style) rather than the full HiPPO structured matrix. This is simpler and still effective.
3. **Toy-scale model:** D=64, N=16, 2 layers — vs the paper's D up to 4096, N=16-256, dozens of layers.
4. **No RMSNorm/weight decay tricks:** The paper uses RMSNorm in later versions; we use simple LayerNorm or no norm.
5. **Toy tasks only:** We train on synthetic selective-copy and induction-head tasks, not real language modeling (The Pile, etc.).
6. **Simplified ZOH:** For diagonal A, the ZOH B_bar simplifies; we use the simplified form rather than the full matrix inverse.
7. **No CUDA kernel fusion:** All operations are in pure PyTorch with standard ops — no custom CUDA kernels.
