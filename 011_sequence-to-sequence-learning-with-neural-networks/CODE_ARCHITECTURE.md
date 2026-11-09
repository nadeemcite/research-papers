---
title: "Seq2Seq: Code architecture"
parent: "011 Seq2Seq"
nav_order: 1
---
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
- `src2idx`, `tgt2idx`: map every character to an integer index (40 source tokens and 14 target tokens, special tokens included).
- `idx2src`, `idx2tgt`: reverse mappings for decoding predictions.
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`.

## Section 3 — Encoder

**What it does:** A stack of LSTM layers reads the reversed source sequence one character at a time and returns the final hidden and cell states.

**Class:** `EncoderLSTM`
- Input shape: `(batch_size, src_len)` after embedding becomes `(batch_size, src_len, emb_dim)`.
- LSTM output: `outputs` of shape `(batch_size, src_len, hidden_dim)`; `hidden`, `cell` of shape `(num_layers, batch_size, hidden_dim)`.
- **Padding:** the encoder packs its input with `pack_padded_sequence`, so the final state comes from each date's last real character. Without packing, the same model scored only 77.35% on free-running decoding, because short dates were trained with trailing padding and tested without it.
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
- It has no `forward` method. The training loop calls `model.encoder` and `model.decoder` directly, always feeding the correct previous token (full teacher forcing).
- `translate(src, max_len)`: greedy inference on one source sequence.

**Deliberate simplification vs. full paper:** The paper used 4-layer LSTMs, 1000-dimensional hidden states, an 80k target vocabulary, and beam search decoding. We use 2 layers, 256 hidden units, 64-dimensional embeddings, a character-level vocabulary, and greedy decoding. The architecture shape is identical; only the scale is reduced so the notebook is quick to run on a Kaggle GPU.

## Section 6 — Training loop

**What it does:** Standard supervised training with Adam.
- Shuffle paired (source, target) examples each epoch.
- Pad sequences within a batch to the same length.
- Compute cross-entropy loss and back-propagate through both encoder and decoder.
- Train for 20 epochs, printing losses each epoch and five sample translations every five epochs.
- After training, measure exact-match accuracy with greedy decoding on all 2,000 validation dates.

**Key functions:**
- `train_epoch(model, loader, optimizer, criterion)`
- `evaluate(model, loader, criterion)`: reports loss and exact-match accuracy with teacher forcing, so its accuracy flatters the model. The free-running cell after the training loop is the fair measure.

## Section 7 — Visualization

**What it does:** Visualizes how the encoder compresses the input.

For one held-out source sequence we take the encoder's output at every source position, project those vectors onto their top two principal components, and plot the path colored by position. The plot shows the state moving as characters are read. It does not by itself prove what the final vector stores.

The original paper did not use attention (that came in Bahdanau et al., 2015), so the notebook has no attention heatmap.

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
| Word-level 160k/80k vocab | Character-level vocab (40 source, 14 target tokens) |
| 4-layer LSTM, 1000 hidden | 2-layer LSTM, 256 hidden |
| Beam search (sizes 1, 2 and 12 in Table 1) | Greedy decoding |
| 12M sentence pairs | 20k synthetic pairs |
| SGD at learning rate 0.7, 7.5 epochs, gradient norm clipped at 5 | Adam at 1e-3, 20 epochs, gradient norm clipped at 1, dropout 0.3 between LSTM layers |

These reductions keep the code focused on the paper's architectural contribution without requiring large downloads or long training.
