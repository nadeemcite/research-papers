# Code Architecture — Attention Is All You Need (Transformer)

## Overview

The notebook implements a **mini Transformer** from scratch in PyTorch, following the original paper's architecture but scaled down for Colab GPU feasibility. The model is an encoder-decoder Transformer trained on a **toy sequence copy task** (and optionally a simple reverse/translation task), with full visualization of attention weights.

## Notebook Structure

### Section 1: Setup & Imports
- Installs/imports torch, numpy, matplotlib.
- Sets random seed for reproducibility.
- Device selection (`cuda` if available, else `cpu`).

### Section 2: Toy Dataset — Sequence Copy Task
- **Class:** `CopyTaskDataset(Dataset)` — generates random integer sequences of fixed length as source, with the target being the same sequence (copy) or reversed (reverse task).
- **Vocab:** Small synthetic vocabulary (e.g., integers 0–9 plus special tokens: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3).
- **Collate function:** Pads sequences to max length in batch, returns `(src, tgt)` tensors with padding masks.
- **Data flow:** Each sample is a 1D tensor of token IDs → batched into `(batch_size, seq_len)` tensors.
- **Shapes:** src: `(B, L_src)`, tgt: `(B, L_tgt)`.

### Section 3: Positional Encoding
- **Class:** `PositionalEncoding(nn.Module)`
- Computes sinusoidal positional encodings using the formula from the paper:
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Registered as a buffer (not a learnable parameter).
- Forward: adds PE to embedded input: `x = x + pe[:seq_len]`.
- **Shapes:** Input `(B, L, d_model)` → Output `(B, L, d_model)`.
- **Visualization:** Heatmap of the positional encoding values across positions and dimensions, showing the sinusoidal patterns at different frequencies.

### Section 4: Scaled Dot-Product Attention
- **Function:** `scaled_dot_product_attention(Q, K, V, mask=None)`
- Computes: `softmax(Q @ K^T / sqrt(d_k)) @ V`
- Optional mask: for causal masking in decoder self-attention, and padding masks.
- Returns both the attention output and the attention weights (for visualization).
- **Shapes:** Q `(B, H, L_q, d_k)`, K `(B, H, L_k, d_k)`, V `(B, H, L_k, d_v)` → output `(B, H, L_q, d_v)`, weights `(B, H, L_q, L_k)`.

### Section 5: Multi-Head Attention
- **Class:** `MultiHeadAttention(nn.Module)`
- Linear projections W_Q, W_K, W_V from d_model to d_model (then split into H heads of dimension d_k = d_model / H).
- Output projection W_O from d_model to d_model.
- Forward: splits Q/K/V into heads, applies scaled dot-product attention per head, concatenates, projects.
- **Shapes:** Input Q/K/V each `(B, L, d_model)` → reshaped to `(B, H, L, d_k)` → attention → concat to `(B, L, d_model)` → projected.

### Section 6: Position-wise Feed-Forward Network
- **Class:** `PositionwiseFeedForward(nn.Module)`
- Two linear layers with ReLU in between: `FFN(x) = Linear2(ReLU(Linear1(x)))`
- d_model → d_ff → d_model.
- **Shapes:** Input `(B, L, d_model)` → `(B, L, d_ff)` → `(B, L, d_model)`.

### Section 7: Encoder Layer
- **Class:** `EncoderLayer(nn.Module)`
- Two sub-layers: (1) multi-head self-attention, (2) position-wise FFN.
- Each sub-layer wrapped with residual connection + LayerNorm: `LayerNorm(x + Sublayer(x))`.
- Forward: applies self-attention (Q=K=V=x), then FFN.
- **Shapes:** Input `(B, L, d_model)` → Output `(B, L, d_model)`.

### Section 8: Decoder Layer
- **Class:** `DecoderLayer(nn.Module)`
- Three sub-layers: (1) masked multi-head self-attention, (2) cross-attention over encoder output, (3) position-wise FFN.
- Each with residual + LayerNorm.
- Causal mask prevents attending to future positions; padding mask handles variable lengths.
- **Shapes:** Input `(B, L_tgt, d_model)`, encoder output `(B, L_src, d_model)` → Output `(B, L_tgt, d_model)`.

### Section 9: Full Transformer Model
- **Class:** `Transformer(nn.Module)`
- Components: source embedding, target embedding, positional encoding, N encoder layers, N decoder layers, final linear projection to vocab size.
- Forward (training): embeds src and tgt, adds PE, passes through encoder and decoder stacks, applies final linear layer to get logits over vocabulary.
- Greedy decoding function for inference: generates output token-by-token using the causal mask.
- **Shapes:** src `(B, L_src)`, tgt `(B, L_tgt)` → logits `(B, L_tgt, vocab_size)`.

### Section 10: Training Loop
- **Loss:** CrossEntropyLoss with ignore_index for padding tokens.
- **Optimizer:** Adam with the paper's learning rate schedule (warmup + inverse sqrt decay).
- Trains for a fixed number of epochs on the copy task, printing loss every N steps.
- Plots training loss curve.

### Section 11: Evaluation & Inference
- Greedy decode function: feeds `<bos>`, generates one token at a time until `<eos>` or max length.
- Runs on test sequences and compares predicted output vs target.
- Computes accuracy on the copy/reverse task.

### Section 12: Attention Visualization
- Extracts attention weights from the trained model for a sample input.
- **Plot 1:** Encoder self-attention heatmap — shows which source tokens attend to which.
- **Plot 2:** Decoder-encoder cross-attention heatmap — shows alignment between decoder positions and encoder positions.
- **Plot 3:** Multi-head attention across different heads — shows different heads learn different patterns.
- Uses matplotlib heatmaps with labeled axes.

## Key Functions/Classes Summary

| Component | Class/Function | Purpose |
|---|---|---|
| Positional Encoding | `PositionalEncoding` | Injects positional info via sin/cos |
| Attention | `scaled_dot_product_attention` | Core attention computation |
| Multi-Head Attention | `MultiHeadAttention` | Parallel attention across subspaces |
| Feed-Forward | `PositionwiseFeedForward` | Per-position MLP |
| Encoder Layer | `EncoderLayer` | Self-attention + FFN with residuals |
| Decoder Layer | `DecoderLayer` | Masked self-attn + cross-attn + FFN |
| Full Model | `Transformer` | Complete encoder-decoder stack |
| Training | training loop | Adam + warmup schedule, cross-entropy loss |
| Inference | `greedy_decode` | Autoregressive generation |
| Visualization | attention plots | Heatmaps of attention weights |

## Deliberate Simplifications vs Full Paper

1. **Scale:** d_model=64 (vs 512), d_ff=128 (vs 2048), N=2 layers (vs 6), H=4 heads (vs 8). This keeps training under ~5 minutes on Colab GPU while preserving the architecture's essential structure.
2. **Task:** Toy copy/reverse task on synthetic integer sequences (vs WMT machine translation with 10K+ vocab and 40K+ sentence pairs). The copy task specifically tests whether the model can learn to attend to the correct source positions.
3. **Vocabulary:** ~14 tokens (vs ~37K for WMT En-De).
4. **No label smoothing:** Uses standard cross-entropy (the paper uses ε=0.1 label smoothing).
5. **No beam search:** Greedy decoding only (the paper uses beam search with width 4).
6. **No learned positional embeddings:** Uses the sinusoidal encoding as in the original paper (some later works use learned embeddings).
7. **Single run:** No extensive hyperparameter search or multi-GPU training.
