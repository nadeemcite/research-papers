# Code Architecture — Pointer Networks Notebook

## Overview

The notebook implements a Pointer Network (Ptr-Net) from scratch in PyTorch, trains it to solve small convex-hull problems, and visualizes the pointer attention path over input points. The model uses an LSTM encoder-decoder architecture where the attention mechanism acts as a pointer — at each decoding step, the attention distribution over encoder hidden states directly produces the probability of selecting each input point as the next output. No fixed output vocabulary is used; the "vocabulary" is dynamically the set of input points.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, NumPy, Matplotlib
- Device detection (GPU if available)
- Reproducibility seeding

### 2. Synthetic Data Generation (`generate_convex_hull_example`)
Generates random 2D point sets and computes their convex hull using the monotone-chain algorithm. The output is the sequence of indices (into the input array) that form the hull boundary in clockwise/counterclockwise order.

**Key functions:**
- `generate_convex_hull_example(n_points, coord_range=10)`: Returns `(points, hull_indices)` where `points` is an `(n_points, 2)` array of random floats and `hull_indices` is a list of indices forming the convex hull.
- `generate_dataset(n_samples, min_points, max_points)`: Creates a dataset of varying-length point sets with their hull solutions.

**Data shapes:**
- Input points: `(n_points, 2)` — float coordinates
- Target output: `(hull_length,)` — integer indices into the input array
- Note: hull length varies per example (depends on point distribution)

### 3. Input Representation (`PointSequence` dataset)
- Each point (x, y) is represented as a 2D vector fed to the encoder LSTM
- Special start-of-sequence (SOS) and end-of-sequence (EOS) tokens are added to the decoder input/output
- Sequences are padded to the maximum length in each batch for efficient batching
- A separate index `n_points` is reserved as the EOS token position

**Key class:** `PointSequenceDataset(torch.utils.data.Dataset)`
- `__getitem__(i)`: returns `(points_tensor, hull_indices_tensor, n_points)`
- Padding masks are used to handle variable-length sequences

### 4. Pointer Network Model (`PointerNet`)

The core model: an LSTM encoder + LSTM decoder with pointer attention.

**Key class:** `PointerNet(nn.Module)`

**Components:**
- **Encoder:** `nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)` — processes the input point sequence. Outputs encoder hidden states `e_1, ..., e_n` of shape `(batch, n_points, hidden_dim)`.
- **Decoder:** `nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)` — generates output positions. At each step, receives the previously selected point (or SOS token for the first step).
- **Pointer attention:** Two linear projections `W_s` (decoder state) and `W_e` (encoder state) map to a shared attention space. The score for encoder state j at decoder step t is: `u_j = v^T tanh(W_s s_t + W_e e_j)`, normalized via softmax over input positions. A mask excludes padding positions.

**Key methods:**
- `forward(points, target_hull, n_points, teacher_forcing_ratio=1.0)`:
  1. Encode input points → encoder hidden states `(batch, max_n, hidden_dim)`
  2. Initialize decoder with SOS token
  3. For each decoder step (up to max output length):
     - Compute attention scores over all encoder positions
     - Apply padding mask
     - softmax → pointer probability distribution `(batch, max_n)`
     - If teacher forcing: next decoder input = points[batch, target[step]]
     - Else: next decoder input = points[batch, argmax(pointer_probs)]
  4. Return pointer_logits (list of `(batch, max_n)` tensors) and predicted indices

**Architecture parameters:**
- Hidden dimension: 128 (paper uses 512)
- Input dimension: 2 (x, y coordinates)
- LSTM layers: 1 (paper uses 1 for encoder, 1 for decoder)
- Maximum input length: 50 points (same as paper's training range)

### 5. Training Loop

- **Loss:** Cross-entropy loss over the pointer distribution at each decoder step. The target is the index of the correct hull point. EOS is handled with a special index.
- **Optimizer:** Adam, learning rate 1e-3 (paper uses Adam with lr scheduling)
- **Teacher forcing:** Always on during training (paper uses teacher forcing)
- **Batch size:** 128 (paper uses 128)
- **Epochs:** 50 (paper trains for ~1000 epochs; reduced for GPU runtime)
- **Curriculum:** Training data includes point sets of sizes 5–50 (paper uses 5–50)

**Training data shapes:**
- Batch of points: `(batch, max_n, 2)`
- Batch of hull indices: `(batch, max_hull_len)` — padded
- Lengths tensor: `(batch,)` — actual number of points per example

### 6. Evaluation
- Generate test point sets of various sizes (including sizes larger than training max)
- Measure accuracy: percentage of correctly predicted hull sequences (exact match)
- Measure point-level accuracy: percentage of individual hull indices correctly predicted
- Report generalization to unseen lengths (e.g., train on 5–30, test on 35–50)

### 7. Pointer Path Visualization
- For sample test cases, plot the 2D point cloud with the predicted hull and true hull overlaid
- Visualize the attention distribution at each decoder step as a heatmap over input points
- Show the pointer "path": which points were selected in order, with arrows connecting them
- Highlight mismatches between predicted and true hull

### 8. Sorting Task (Bonus)
- As a secondary task, apply the same Pointer Network to sort sequences of numbers
- Input: a sequence of random numbers; output: the indices that would sort them
- Demonstrates the generality of the pointer mechanism beyond geometric problems

## Data Flow

```
Input points (n, 2)  →  [LSTM Encoder]  →  Encoder hidden states (n, hidden_dim)
                                                        ↓
SOS token (1, 2)  →  [LSTM Decoder]  →  Decoder state s_t (1, hidden_dim)
                                                        ↓
                              [Pointer Attention: u_j = v^T tanh(W_s s_t + W_e e_j)]
                                                        ↓
                              softmax(u_j) → pointer distribution (n,)
                                                        ↓
                              argmax → predicted index → points[index] → next decoder input
                                                        ↓
                              Repeat until EOS predicted
                                                        ↓
                              Output: sequence of indices [i_1, i_2, ..., i_k]
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Tasks | Convex hull, Delaunay triangulation, TSP | Convex hull + sorting |
| Hidden dimension | 512 | 128 |
| Training range | 5–50 points | 5–50 points (same) |
| Test generalization | Up to 500–1000 points | Up to 80 points |
| LSTM layers | 1 (encoder + decoder) | 1 (same) |
| Epochs | ~1000 | 50 |
| Beam search | Beam size 5 at inference | Greedy decoding (argmax) |
| Attention mechanism | Bahdanau additive attention | Bahdanau additive (same) |
| Data size | 10M training examples | ~10K training examples |
| TSP | Full TSP with near-optimal comparison | Not included (focus on convex hull) |
| Delaunay | Triangle prediction with 3-index output | Not included |
| Curriculum learning | Not used | Not used (same) |
| Evaluation | Exact sequence match + point accuracy | Same metrics |

The simplifications keep the notebook self-contained and runnable within a Kaggle GPU session (~15 min) while preserving the core architecture: LSTM encoder-decoder with pointer attention, variable-length output dictionary, teacher-forced training, and generalization to longer sequences.
