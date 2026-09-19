# Code Architecture — Neural Machine Translation by Jointly Learning to Align and Translate

This notebook builds a minimal but faithful Bahdanau attention seq2seq model from scratch in PyTorch, trains it on a toy date-translation task, and visualizes the attention alignment heatmap between source and target tokens.

## Section 1 — Data: a toy translation task

**What it does:** Generates synthetic date strings in two formats, e.g. `"March 5 2020"` → `"05/03/2020"` (European format). This is a genuine sequence-to-sequence task: the input and output lengths differ, tokens must be reordered, and the model must learn which input positions matter for each output token—exactly what attention solves.

**Why this task:** The paper trained on WMT'14 English→French (348M words, 30k vocab each). A date task lets us demonstrate the same attention mechanics with a tiny vocabulary and no external dataset download, so the notebook is self-contained and Kaggle-runnable.

**Key variables:**
- `src_seq`: human-readable date string as a list of characters.
- `tgt_seq`: European-format date string as a list of characters, terminated by `<EOS>`.

## Section 2 — Vocabulary and tokenization

**What it does:** Builds character-level vocabularies for source and target.

**Key mappings:**
- `char2idx`: maps every character to an integer index.
- `idx2char`: reverse mapping for decoding predictions.
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`.

## Section 3 — Bidirectional GRU Encoder

**What it does:** A bidirectional GRU reads the source sequence and produces annotations that capture both left and right context for each position.

**Class:** `BiGRUEncoder`
- Input shape: `(batch_size, src_len)` after embedding becomes `(batch_size, src_len, emb_dim)`.
- BiGRU output: `outputs` of shape `(batch_size, src_len, hidden_dim * 2)` — forward and backward hidden states concatenated per position.
- This matches the paper's bidirectional RNN encoder where `h_j = [h→_j ; h←_j]`.

## Section 4 — Bahdanau Attention

**What it does:** Computes additive attention scores between the decoder's previous hidden state and all encoder annotations.

**Class:** `BahdanauAttention`
- Alignment model: `e_ij = v^T tanh(W_s s_{i-1} + W_h h_j)` implemented as:
  - Linear projection of decoder hidden state: `W_s @ s_{i-1}`
  - Linear projection of encoder outputs: `W_h @ h_j`
  - Add, tanh, project through `v`
- Attention weights: `α_ij = softmax(e_ij)` over source positions.
- Context vector: `c_i = Σ_j α_ij h_j` — weighted sum of encoder annotations.
- Returns context vector and attention weights (for visualization).

## Section 5 — Attention Decoder

**What it does:** A GRU decoder that generates the target sequence one token at a time, conditioned on the previous token, its own hidden state, and the attention-derived context vector.

**Class:** `AttentionDecoder`
- At each step: compute attention → context vector → GRU cell (input = embedded previous token + context) → output projection to vocabulary logits.
- At training time: teacher forcing with the ground-truth previous token.
- At inference time: autoregressive generation from `<SOS>` until `<EOS>`.

## Section 6 — Full model wrapper

**Class:** `BahdanauSeq2Seq`
- Ties encoder, attention, and decoder.
- `forward(src, tgt, teacher_forcing_ratio)`: encodes source, then decodes target step by step, returning logits and attention weights.
- `translate(src, max_len)`: greedy inference, returns predicted tokens and attention matrix.

**Deliberate simplification vs. full paper:** The paper used 1000-unit GRUs, a bidirectional encoder with 1000 units each direction, maxout output layers, Adadelta optimizer, beam search, and 30k word vocabularies. We use smaller GRUs (256 hidden), character-level vocabularies, Adam optimizer, and greedy decoding. The architecture shape (bidirectional encoder → additive attention → conditional decoder) is identical; only the scale is reduced for fast Kaggle training.

## Section 7 — Training loop

**What it does:** Standard supervised training with Adam and cross-entropy loss.
- Shuffle paired examples each epoch.
- Pad sequences within a batch.
- Compute loss and backpropagate through encoder, attention, and decoder jointly.
- Track training loss and sample translations every few epochs.

**Key functions:**
- `train_step(model, batch, optimizer, criterion)`
- `evaluate(model, val_loader)`

## Section 8 — Attention visualization

**What it does:** Plots the attention alignment heatmap for sample translations.

### 8a — Attention heatmap
For a held-out source→target pair, we extract the attention weight matrix `α_ij` (shape: `tgt_len × src_len`) and plot it as a heatmap. Each row shows which source positions the decoder focused on when generating that target token. This directly reproduces Figure 3 from the paper.

### 8b — Comparison: with vs. without attention
We briefly compare the attention model's output against a plain encoder–decoder (from the seq2seq notebook) on long inputs, showing that attention maintains accuracy where the fixed-vector model degrades.

## Data flow summary

```
raw date string
      ↓
character tokenization
      ↓
BiGRU encoder  →  annotations h_1...h_Tx  (each: forward + backward context)
      ↓
for each target step i:
    decoder state s_{i-1}
         ↓
    Bahdanau attention: score s_{i-1} vs. each h_j  →  α_ij  →  context c_i
         ↓
    GRU decoder: s_i = GRU(s_{i-1}, [y_{i-1}, c_i])
         ↓
    logits → softmax → y_i
      ↓
generated date string + attention heatmap
```

## Simplifications explicitly retained

| Paper | Notebook |
|---|---|
| WMT'14 English→French (348M words) | Synthetic date normalization (20k pairs) |
| Word-level 30k/30k vocab | Character-level ~20 token vocab |
| 1000-unit BiGRU encoder, 1000-unit decoder | 256-unit BiGRU encoder, 256-unit decoder |
| Maxout output layer | Linear output projection |
| Adadelta optimizer | Adam optimizer |
| Beam search (B=12) | Greedy decoding |
| 5 days training | ~5 minutes on Kaggle GPU |

These reductions keep the code focused on the paper's architectural contribution (additive attention + bidirectional encoder) without requiring large downloads or long training.
