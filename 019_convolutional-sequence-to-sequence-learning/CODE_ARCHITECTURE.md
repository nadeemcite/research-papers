# Code Architecture — Convolutional Sequence to Sequence Learning Notebook

## Overview

The notebook implements a fully convolutional encoder-decoder model with gated linear units (GLUs) and multi-step attention for a toy sequence-to-sequence task (sorting random integer sequences), and compares its training speed against a traditional RNN (GRU) seq2seq baseline. Both models are trained on the same task with the same data, and we measure wall-clock training time per epoch and convergence speed. The implementation follows the architecture described in Gehring et al. (2017), adapted to a small-scale synthetic task for clarity and GPU runtime constraints.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, NumPy, Matplotlib
- Device detection (GPU if available)
- Reproducibility seeding
- Timing utilities

### 2. Toy Task: Sequence Sorting
We generate synthetic sequences of random integers from a small vocabulary. The task is to sort the input sequence in ascending order — a simple sequence-to-sequence problem where the output length equals the input length and the model must learn to rearrange tokens.

**Key functions:**
- `generate_sorting_example(seq_len, vocab_size)`: Returns `(input_seq, sorted_seq)` where both are lists of integers.
- `generate_dataset(n_samples, min_len, max_len, vocab_size)`: Creates a dataset of varying-length sorting examples.

**Data shapes:**
- Input sequence: `(seq_len,)` — integer tokens
- Target sequence: `(seq_len,)` — sorted integer tokens
- Special tokens: `<pad>=0`, `<sos>=vocab_size+1`, `<eos>=vocab_size+2`

### 3. Dataset & DataLoader
- `SortingDataset(torch.utils.data.Dataset)`: wraps generated examples, pads sequences to max length in batch.
- Collate function handles padding and creates padding masks.
- Batches are shaped `(batch, max_seq_len)` with corresponding masks `(batch, max_seq_len)`.

### 4. Position Embeddings
- Token embeddings: `nn.Embedding(vocab_size+3, embed_dim)` — maps each token to a dense vector.
- Position embeddings: `nn.Embedding(max_len+padding, embed_dim)` — learned absolute position representations.
- Combined: `e = token_emb(x) + pos_emb(positions)` — gives the model order awareness.

**Shapes:** `(batch, seq_len) → (batch, seq_len, embed_dim)`

### 5. Convolutional Block with GLU
The core building block of both encoder and decoder.

**Key class:** `ConvBlock(nn.Module)`

**Forward pass:**
1. Input: `(batch, seq_len, d)` 
2. 1D Convolution (kernel width k, maps d → 2d): `(batch, seq_len, 2d)`
3. Split into A, B each `(batch, seq_len, d)`
4. GLU: `A * sigmoid(B)` → `(batch, seq_len, d)`
5. Residual: `output + input` (with linear projection if dimensions change)
6. Scale by `sqrt(0.5)` to halve variance

**Parameters:**
- Kernel width k=3 (paper uses 3-5)
- Hidden dimension d=128 (paper uses 256-512)
- Number of layers: 4 (paper uses 6-15)

### 6. Convolutional Encoder
Stacks N ConvBlocks with padding on both sides to preserve sequence length.

**Key class:** `ConvEncoder(nn.Module)`

**Forward pass:**
1. Embed input: `token_emb + pos_emb → (batch, seq_len, embed_dim)`
2. Linear projection to hidden dim d
3. Apply N ConvBlocks (each preserves length via padding)
4. Output: `(batch, seq_len, d)` — encoder representations z

### 7. Convolutional Decoder with Multi-Step Attention
Causal convolutional decoder with a separate attention module at each layer.

**Key class:** `ConvDecoder(nn.Module)`

**Forward pass per layer l:**
1. Pad input on the LEFT by k-1 zeros (causal — no future information)
2. Apply ConvBlock → `h_i^l` of shape `(batch, seq_len, d)`
3. Compute attention:
   - `d_i^l = W_d * h_i^l + b_d + g_i` (decoder state + previous target embedding)
   - `a_ij = softmax(d_i^l · z_j^u)` (dot product with last encoder layer)
   - `c_i^l = Σ_j a_ij * (z_j^u + e_j)` (weighted sum of encoder output + source embeddings)
4. Add conditional input: `h_i^l = h_i^l + c_i^l`
5. Pass to next layer

**Final output:** Linear layer maps top decoder output to vocabulary distribution: `softmax(W_o * h^L + b_o)`

### 8. Full ConvS2S Model
**Key class:** `ConvS2S(nn.Module)`

- Combines encoder + decoder + embeddings
- `forward(src, tgt)`: encodes source, decodes with teacher forcing, returns logits `(batch, tgt_len, vocab_size)`
- Training: cross-entropy loss over target tokens

### 9. RNN Seq2Seq Baseline (GRU)
A standard recurrent encoder-decoder with attention for comparison.

**Key class:** `RNNSeq2Seq(nn.Module)`

- **Encoder:** Bidirectional GRU, `nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)`
- **Decoder:** Unidirectional GRU with Bahdanau-style additive attention over encoder outputs
- Same embedding dimensions and vocabulary as the ConvS2S model for fair comparison
- `forward(src, tgt)`: encodes source, decodes with teacher forcing, returns logits

### 10. Training Loop & Speed Comparison
- Train both models for the same number of epochs on the same data
- Measure wall-clock time per epoch using `time.time()`
- Track loss curves for both models
- Print training speed comparison (seconds per epoch, total training time)

**Training hyperparameters:**
- Batch size: 128
- Learning rate: 1e-3 (Adam)
- Epochs: 30
- Sequence lengths: 8-20 (varied)
- Vocabulary size: 50
- Training samples: 10,000

### 11. Evaluation & Visualization
- Compute exact-match accuracy on test sequences (full sequence correctly sorted)
- Compute token-level accuracy (percentage of individual positions correct)
- Plot training loss curves for ConvS2S vs RNN baseline
- Plot training time per epoch comparison
- Show sample predictions from both models
- Display attention weights from the ConvS2S decoder (heatmap for a sample)

## Data Flow

```
Source seq (batch, src_len)  →  [Token + Pos Embeddings]  →  (batch, src_len, embed_dim)
                                                                ↓ [Linear → d]
                                                                ↓ [ConvBlock × N (padded)]
                                                                ↓
                                              Encoder output z (batch, src_len, d)
                                                                ↓
Target seq (batch, tgt_len)  →  [Token + Pos Embeddings]  →  (batch, tgt_len, embed_dim)
                                                                ↓ [Linear → d]
                                                ↓ [ConvBlock (causal pad)]
                                                ↓               ↓
                                   h_i^l (batch, tgt_len, d)   z_j^u (batch, src_len, d)
                                                ↓               ↓
                                   [Multi-step Attention: dot product + weighted sum]
                                                ↓
                                   c_i^l + h_i^l  →  next ConvBlock layer
                                                ↓
                                                ↓ [Linear → vocab_size + softmax]
                                                ↓
                              Output logits (batch, tgt_len, vocab_size)
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Task | Machine translation (WMT'14 EN-DE, EN-FR, EN-RO) | Integer sequence sorting (toy task) |
| Vocabulary size | 32K-80K BPE subwords | 50 integers (+ special tokens) |
| Hidden dimension d | 256-512 | 128 |
| Number of layers | 6-15 (encoder) / 6-15 (decoder) | 4 (encoder) / 4 (decoder) |
| Kernel width k | 3-5 | 3 |
| Embedding dimension | 512 | 128 |
| Attention | Dot-product, multi-step per decoder layer | Same (dot-product, per-layer) |
| Normalization | Weight scaling + careful init | Simplified (no gradient scaling) |
| Beam search | Beam size 5-12 for inference | Greedy decoding (argmax) |
| Dropout | 0.1-0.2 on embeddings and conv inputs | 0.1 on conv inputs |
| Training data | 4.5M sentence pairs | 10K synthetic sequences |
| Training time | Days on multi-GPU | Minutes on single GPU |
| Optimization | NAG, cosine schedule, label smoothing | Adam, fixed lr |
| Evaluation metric | BLEU | Exact-match + token accuracy |
| Position embeddings | Learned absolute | Same (learned absolute) |
| GLU | Gated linear units | Same |

The simplifications keep the notebook self-contained and runnable within a Kaggle GPU session (~15 min) while preserving the core architecture: fully convolutional encoder-decoder with GLUs, position embeddings, residual connections, multi-step attention at every decoder layer, and a direct speed comparison against an RNN baseline — which is the central claim of the paper.
