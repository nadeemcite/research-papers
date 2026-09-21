# Code Architecture — Neural Turing Machines Notebook

## Overview

The notebook implements a simplified Neural Turing Machine (NTM) with a differentiable memory bank and content-based addressing, trains it on the copy task, and visualises the read/write attention weightings over memory slots across timesteps.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs and imports `torch`, `torch.nn`, `numpy`, `matplotlib`.
- Sets random seed for reproducibility.
- Detects CUDA (Kaggle provides Tesla P100/T4 GPUs).

### 2. NTM Addressing Module (`NTMAddressing`)
The core differentiable addressing mechanism, implemented as a standalone module so it can be reused by any head.

**Key functions:**
- `cosine_similarity(k, M)`: Computes batched cosine similarity between key vector k and each row of memory M. Returns similarity scores of shape (batch, N).
- `content_addressing(k, beta, M)`: Softmax over similarity scores scaled by key strength β. Returns weightings w_c of shape (batch, N).
- `location_addressing(w_prev, w_c, g, s, gamma)`: Combines content and location-based addressing:
  1. **Interpolation**: w_g = g · w_c + (1−g) · w_prev
  2. **Shift**: w̃ = circular_convolution(w_g, s) — convolution with shift weighting s
  3. **Sharpen**: w = w̃^γ / Σ w̃^γ
- `circular_convolution(w, s)`: Implements the shift as a circular convolution using a flattened approach. For shift range {-1, 0, +1}, s has 3 elements; the convolution rolls w by the appropriate amounts.

**Data flow:**
```
key k (batch, M)  ──→  cosine_similarity  ──→  scores (batch, N)
                                              ↓  × β, softmax
                                         w_c (batch, N)
w_prev (batch, N) ──→  interpolation(g)  ──→  w_g (batch, N)
                                              ↓  circular_conv(s)
                                         w̃ (batch, N)
                                              ↓  sharpen(γ)
                                         w (batch, N)  [final weighting]
```

### 3. NTM Head (`NTMHeadBase`, `NTMReadHead`, `NTMWriteHead`)
- `NTMHeadBase`: Abstract base class. Holds an LSTM controller (or linear layer for feedforward mode). Produces all addressing parameters via linear projections:
  - key k_t: (batch, M) — via `tanh` activation
  - key strength β_t: scalar — via `softplus` + 1 (ensures β ≥ 1)
  - interpolation gate g_t: scalar — via `sigmoid`
  - shift weighting s_t: (batch, 3) — via `softmax` (for shifts {-1, 0, +1})
  - sharpening γ_t: scalar — via `softplus` + 1 (ensures γ ≥ 1)
- `NTMReadHead`: Adds a linear layer to produce the read vector r_t = Σ w_t(i) · M_t(i).
- `NTMWriteHead`: Adds linear layers for erase vector e_t (batch, M, via `sigmoid`) and add vector a_t (batch, M, via `tanh`).

**Output shapes from controller:**
- Controller hidden: (batch, controller_size)
- All head parameters: derived via separate Linear(controller_size, param_dim) layers

### 4. NTM Model (`NeuralTuringMachine`)
- **Constructor**: Takes `input_size`, `output_size`, `controller_size`, `memory_n` (N), `memory_m` (M), `num_heads`, `num_shifts`.
- **Internal state**: `memory` (batch, N, M), `read_vectors` (batch, num_heads, M), `prev_weightings` (batch, num_heads, N).
- `init_state(batch_size)`: Initialises memory to small random values, read vectors to zeros, weightings to uniform.
- `forward(input)`: 
  1. Concatenate input with previous read vectors → controller input
  2. Run LSTM controller → hidden state
  3. For each head: compute addressing parameters → get weightings → read/write
  4. Update memory: erase then add (Equations 3, 4)
  5. Concatenate read vectors → output layer → output
  6. Return output and new state

**Memory update (per write head):**
```
M̃(i) = M(i) ⊙ (1 − w(i) · e)     # erase
M(i)  = M̃(i) + w(i) · a            # add
```
where ⊙ is element-wise, w is (N,), e and a are (M,).

### 5. Copy Task Data Generator (`copy_task_batch`)
- Generates random binary sequences of shape (batch, seq_len, input_size).
- Input format: [seq_len vectors of 8 bits] + [delimiter flag (extra dim)] + [T_pad zeros]
- Target format: [T_pad zeros] + [seq_len copies of input vectors]
- `input_size = 8` (8-bit vectors), plus 1 dimension for delimiter = 9 total input dims
- Sequence lengths randomised between 1 and 20 during training.

**Shapes:**
- Input tensor: (batch, seq_len + 1 + seq_len, input_size + 1) — sequence + delimiter + output period
- Target tensor: same shape, with targets in the output period

### 6. Training Loop
- **Loss**: Binary cross-entropy (BCE) per sequence — the paper reports bits-per-sequence.
- **Optimizer**: RMSProp (or Adam as a modern alternative), learning rate 1e-4.
- **Gradient clipping**: Elementwise clip to (-10, 10) per the paper.
- **Episode reset**: Memory, controller hidden state, and read vectors are reset at the start of each sequence.
- **Training loop**: 10,000 episodes (reduced from paper's ~100K for Kaggle runtime), batch size 1–4.
- **Reporting**: Loss printed every 1000 episodes; loss curve plotted at the end.

### 7. Evaluation & Generalisation Test
- After training, test on sequences of length 10 (in training range) and length 50 (out of training range).
- Compare NTM vs. a plain LSTM baseline (3-layer LSTM, 256 hidden units).
- Report bits-per-sequence for both.

### 8. Memory Visualisation
- Extract the read/write weightings (w_t) at each timestep during a copy task episode.
- **Read head attention heatmap**: Timesteps (y-axis) vs. memory locations (x-axis), colour = weighting intensity.
- **Write head attention heatmap**: Same layout for write weightings.
- **Memory snapshot**: Show the memory matrix M at key timesteps (start, mid-sequence, end).
- These visualisations reveal the learned algorithm: the write head sequentially fills memory slots, and the read head sequentially reads them back during the output phase.

## Key Classes Summary

| Class | Responsibility |
|-------|---------------|
| `NTMAddressing` | Static methods for content + location addressing math |
| `NTMHeadBase` | Base class producing addressing parameters from controller output |
| `NTMReadHead` | Read head: computes read vector from weightings + memory |
| `NTMWriteHead` | Write head: produces erase + add vectors for memory update |
| `NeuralTuringMachine` | Full NTM: controller + memory + heads, forward step, state init |
| `NTMController` | LSTM controller wrapper (or feedforward MLP) |

## Deliberate Simplifications vs. Full Paper

1. **Single read/write head**: The paper uses 1 head for copy, but we use 1 combined head for simplicity. The architecture supports multiple heads.
2. **Memory size**: 128×20 in the paper; we use 32×10 or 64×10 to reduce training time on Kaggle.
3. **Controller size**: 100 units in the paper (feedforward) or LSTM; we use 100 LSTM units.
4. **Training episodes**: ~10K instead of ~100K+ for Kaggle's 30-minute limit. The model still converges on shorter sequences.
5. **Location-based addressing**: Included but simplified — shift range is {-1, 0, +1} (3 shifts) as in the paper.
6. **No repeat-copy or priority-sort tasks**: Only the copy task is implemented as it's the canonical NTM demo.
7. **LSTM baseline**: A simple 1-layer LSTM rather than the paper's 3-layer stacked LSTM.
8. **Batch processing**: Sequences processed one at a time (batch_size=1) for clarity; the architecture supports batching.

## Data Flow Summary

```
Input x_t (batch, input_size)
    ↓
[concat with prev read vectors] → (batch, input_size + num_heads*M)
    ↓
LSTM Controller → hidden h_t (batch, controller_size)
    ↓
    ├─→ Read Head: k,β,g,s,γ → addressing → w_read → r_t = Σ w(i)·M(i)
    ├─→ Write Head: k,β,g,s,γ,e,a → addressing → w_write → M = erase(M,w,e) + add(M,w,a)
    └─→ Output Layer: [r_1, r_2, ...] → y_t (batch, output_size)
```
