# Code Architecture — Sequence to Sequence Learning with Neural Networks

This notebook builds a minimal but faithful seq2seq LSTM system from scratch in PyTorch, trains it on a toy date-normalization task, and visualizes the encoder's hidden-state compression.

## Section 1 — Data: a toy sequence-to-sequence task

**What it does:** Generates synthetic date strings in two formats, e.g. "March 5 2020" → "2020-03-05". This is a classic toy task: the input and output lengths differ, characters/tokens must be reordered, and the model must learn a compact intermediate representation.

**Why this task:** The paper trained on WMT'14 English→French (12M sentence pairs, 160k source vocab, 80k target vocab). A date task lets us demonstrate the same encoder/decoder mechanics with a tiny vocabulary and no external dataset download, so the notebook is self-contained and Kaggle-runnable.

**Key variables:**
- `src_seq`: human-readable date string as a list of characters.
- `tgt_seq`: ISO-format date string as a list of characters, terminated by `<EOS>`.

## Section 2 — Vocabulary and tokenization

**What it does:** Builds character-level vocabularies for source and target. Characters are used instead of words because the date task has a small alphabet and a character model directly mirrors the token-by-token generation in the paper.

**Key mappings:**
- `char2idx`: maps every character to an integer index.
- `idx2char`: reverse mapping for decoding predictions.
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`.

## Section 3 — Encoder

**What it does:** A stack of LSTM layers reads the reversed source sequence one character at a time and returns the final hidden and cell states.

**Class:** `EncoderLSTM`
- Input shape: `(batch_size, src_len)` after embedding becomes `(batch_size, src_len, emb_dim)`.
- LSTM output: `outputs` of shape `(batch_size, src_len, hidden_dim)`; `hidden`, `cell` of shape `(num_layers, batch_size, hidden_dim)`.
- **Reversing:** the source string is reversed before encoding, matching the paper's famous trick. This brings the first source token closer to the first target token during decoding.

## Section 4 — Decoder

**What it does:** A separate stack of LSTM layers generates the target sequence one token at a time, conditioned on the encoder's final state.

**Class:** `DecoderLSTM`
- At training time: teacher forcing. The ground-truth previous token is fed as the next input; the input tensor is `(batch_size, tgt_len)` and the LSTM is unrolled in a single forward pass.
- At inference time: autoregressive generation. The decoder starts with `<SOS>` and loops until `<EOS>` is emitted.
- Output projection: a linear layer maps LSTM hidden state to logits over the target vocabulary.
- Loss: cross-entropy over target positions, ignoring `<PAD>`.

## Section 5 — Seq2Seq wrapper

**Class:** `Seq2Seq`
- Ties the encoder and decoder.
- `forward(src, tgt, teacher_forcing_ratio=0.5)`: encodes source, then decodes target using teacher forcing with the given probability.
- `translate(src, max_len)`: greedy inference, used for quick demos.

**Deliberate simplification vs. full paper:** The paper used 4-layer LSTMs, 1000-dimensional hidden states, an 80k target vocabulary, and beam search decoding. We use 2 layers, a small hidden dimension, a character-level target vocabulary, and greedy decoding. The architecture shape is identical; only the scale is reduced so the notebook trains in minutes on a Kaggle GPU.

## Section 6 — Training loop

**What it does:** Standard supervised training with Adam.
- Shuffle paired (source, target) examples each epoch.
- Pad sequences within a batch to the same length.
- Compute cross-entropy loss and back-propagate through both encoder and decoder.
- Track training loss and a few sample translations every few epochs.

**Key functions:**
- `train_step(model, batch)`
- `evaluate(model, val_loader)`

## Section 7 — Visualization

**What it does:** Visualizes how the encoder compresses the input.

### 7a — Encoder hidden-state trajectories
For a held-out source sequence we record the top two principal components of the encoder hidden states across time and plot the path, colored by position. The curve shows the model accumulating and re-encoding date information as it reads.

### 7b — Attention-style heatmap (optional extension)
The original paper did not use attention; that came in Bahdanau et al. 2015. We therefore keep the core visualization focused on the fixed "thought vector" and its PCA trajectory, which is the most direct empirical illustration of the paper's claim that a single vector can carry the whole input meaning.

## Data flow summary

```
raw date string
      ↓
character tokenization + reverse source
      ↓
encoder LSTM  →  final (hidden, cell) = thought vector
      ↓
decoder LSTM  →  per-step logits over target chars
      ↓
softmax + argmax  →  generated date string
      ↓
print / plot PCA trajectory
```

## Simplifications explicitly retained

| Paper | Notebook |
|---|---|
| WMT'14 English→French | Synthetic date normalization |
| Word-level 160k/80k vocab | Character-level ~20 token vocab |
| 4-layer LSTM, 1000 hidden | 2-layer LSTM, configurable hidden |
| Beam search (B=12 reported) | Greedy decoding |
| 12M sentence pairs | 20k synthetic pairs |
| Dropout, gradient clipping | Minimal (optional) |

These reductions keep the code focused on the paper's architectural contribution without requiring large downloads or long training.
