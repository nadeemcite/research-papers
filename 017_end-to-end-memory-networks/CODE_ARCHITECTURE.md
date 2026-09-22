# Code Architecture — End-to-End Memory Networks Notebook

## Overview

The notebook implements a multi-hop End-to-End Memory Network (MemN2N) from scratch in PyTorch, trains it on a toy bAbI-style question answering task, and visualises how attention shifts across hops to reach the answer.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs and imports `torch`, `torch.nn`, `numpy`, `matplotlib`.
- Sets random seed for reproducibility.
- Detects CUDA (Kaggle provides Tesla P100/T4 GPUs).

### 2. bAbI-Style Task Data Generator
Generates synthetic stories in the style of Facebook's bAbI dataset: characters move between rooms, pick up and drop objects, and questions are asked about locations and object positions.

**Key function:**
- `generate_babi_story()`: Creates a random story with characters, objects, and rooms. Characters perform actions (go to room, pick up object, drop object). Generates questions like "Where is the milk?" and "Where is Joe?" with known answers derived from the story state.
- Returns: `(story_lines, questions, answers)` tuples.

**Data flow:**
```
Random seed → Generate story sentences (actions) → Track world state
→ Generate questions with ground-truth answers from world state
→ Return (story_lines, questions, answers)
```

**Deliberate simplification vs paper:**
- The paper uses the full bAbI dataset with 20 task types. We generate a subset of the "position reasoning" and "supporting facts" tasks programmatically.
- The paper also evaluates on Penn TreeBank and Text8 language modeling — we focus only on the QA task.

### 3. Vocabulary & Encoding
- `Vocabulary` class: Maps words to integer indices. Built from all stories in the training set.
- Two sentence representation modes (matching the paper):
  - **Bag-of-words (BoW):** Sum of word embedding vectors — `m_i = Σ_j A[x_j]`
  - **Position encoding (PE):** Element-wise multiplication with a position vector — `m_i = Σ_j l_j ⊙ A[x_j]`, where `l_j` encodes the position of word j within the sentence.

**Shapes:**
- Word index: scalar integer
- Sentence embedding: `(embed_dim,)` — computed from embedding matrix lookup + sum/PE
- Memory bank: `(max_memory, embed_dim)` — each row is a sentence embedding
- Query embedding: `(embed_dim,)`

### 4. MemN2N Model (`MemN2N`)
The core model implemented as a PyTorch `nn.Module`, following Section 2 of the paper.

**Key parameters:**
- `vocab_size`: Size of the vocabulary
- `embed_dim`: Embedding dimension (paper uses 20 for 1k tasks, 50 for 10k)
- `max_memory`: Maximum number of memory slots (paper uses 50)
- `num_hops`: Number of memory hops (paper uses 3)
- `encoding`: 'bow' or 'pe' (position encoding)
- `weight_tying`: 'adjacent' or 'layerwise'

**Architecture (single hop — Eq. 1-3):**
1. **Input memory representation:** Each sentence `x_i` is embedded via matrix `A` → memory vector `m_i`. Query `q` is embedded via matrix `B` (or same as A in layer-wise tying) → internal state `u`.
2. **Attention:** Match probabilities `p_i = softmax(u^T · m_i)` — inner product followed by softmax over all memories.
3. **Output memory:** Each sentence has a corresponding output vector `c_i` (via matrix `C`). Response vector `o = Σ_i p_i · c_i`.
4. **Output:** `â = softmax(W · (o + u))` — the sum of output and query embedding is passed through a final weight matrix and softmax.

**Multi-hop (Eq. 4):**
- `u^{k+1} = o^k + u^k` — the output of each hop is added to the input state and passed to the next hop.
- With adjacent weight tying: `C^k = A^{k+1}` (output embedding of layer k = input embedding of layer k+1), and `W = C^K` (final prediction matrix = last output embedding), and `B = A^1` (question embedding = first input embedding).
- With layer-wise weight tying: `A^1 = A^2 = ... = A^K` and `C^1 = C^2 = ... = C^K`, with a linear mapping `H` applied between hops: `u^{k+1} = H · u^k + o^k`.

**Temporal encoding:**
- Each memory `m_i` is augmented: `m_i = m_i + T_A[i]`, where `T_A` is a learned temporal matrix. Similarly for `c_i += T_C[i]`.
- Sentences are indexed in reverse order (most recent = highest index).

**Forward pass data flow:**
```
Story sentences → embed via A → memory bank M (N × d)
Query q → embed via B → state u (d,)
  Hop 1: p = softmax(u @ M.T) → o = p @ C → u = o + u
  Hop 2: p = softmax(u @ M.T) → o = p @ C → u = o + u
  Hop 3: p = softmax(u @ M.T) → o = p @ C → u = o + u
Final: â = softmax(u @ W.T) → predicted answer word
```

### 5. Training Loop
- **Loss:** Cross-entropy loss between predicted answer distribution and true answer (Section 4.2).
- **Optimizer:** SGD with learning rate 0.01, annealed by half every 25 epochs until 100 epochs (per paper).
- **Batch size:** 32 (per paper).
- **Gradient clipping:** Max L2 norm of 40 (per paper).
- **Linear start (LS):** Optionally commence training with softmax removed (model is linear except final softmax), then re-insert softmax when validation loss stops decreasing.
- **Random noise (RN):** Inject 10% empty memory slots during training to regularize temporal encoding.
- **Epochs:** 100 for independent task training; 60 for joint training (1k examples).

### 6. Training Curve Visualisation
- Plots training loss over epochs.
- Plots test accuracy over epochs.

### 7. Evaluation
- Tests the trained model on held-out stories.
- Computes accuracy: fraction of questions answered correctly.
- Compares MemN2N with different configurations: BoW vs PE, 1-hop vs 3-hop, with/without temporal encoding.

### 8. Attention Visualisation Across Hops
The key visualisation: for a sample question, show how attention over the memory bank shifts across each hop.

**Key visualisation:**
```python
# For each hop k, extract attention weights p_i over memory slots
# Plot heatmaps: rows = hops, columns = memory slots, color = attention weight
# Show the story sentences aligned with memory slot indices
```

This demonstrates the multi-hop reasoning pattern: hop 1 might attend to "Joe picked up the milk", hop 2 shifts to "Joe went to the office", hop 3 confirms "Joe left the milk" — progressively narrowing down the answer.

### 9. Multi-Hop Comparison: 1-hop vs 3-hop
- Trains a 1-hop model and a 3-hop model on the same data.
- Compares accuracy, demonstrating that multiple hops improve performance on questions requiring chained reasoning (replicating the paper's key finding).

### 10. Summary
- Prints final accuracy table comparing configurations.
- Shows example predictions with per-hop attention visualisations.

## Key Classes/Functions Summary

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| Data | `generate_babi_story()` | Generate bAbI-style stories with questions |
| Encoding | `Vocabulary`, `position_encoding()` | Word-to-index mapping, BoW / PE encoding |
| Model | `MemN2N(nn.Module)` | Full end-to-end memory network with multi-hop attention |
| Training | `train_memn2n()` | Training loop with cross-entropy, LS, RN |
| Evaluation | `evaluate_memn2n()` | Test accuracy |
| Visualisation | `plot_hop_attention()` | Heatmap of attention weights across hops |
| Baseline | `BaselineLSTM(nn.Module)` | Simple LSTM baseline for comparison |

## Deliberate Simplifications vs Full Paper

1. **Synthetic toy task only:** The paper uses the full bAbI dataset (20 task types from Facebook). We generate a simpler subset programmatically covering position reasoning and supporting facts.
2. **No Penn TreeBank / Text8 experiments:** The paper also evaluates on language modeling — we focus only on QA.
3. **Simplified vocabulary:** Our toy task has a small, controlled vocabulary (~30 words) vs the paper's larger dictionary.
4. **No joint training across all 20 bAbI tasks:** We train on a single task type for clarity.
5. **Reduced embedding dimension:** We use embed_dim=20 (matching the paper's 1k setting) but with fewer total parameters due to smaller vocabulary.
6. **No exhaustive hyperparameter search:** We use the paper's recommended settings directly.
7. **Fixed 3 hops:** The paper explores 1-3 hops; we train both 1-hop and 3-hop models for comparison but don't sweep all values.
