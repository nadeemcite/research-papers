---
title: "BERT: Code architecture"
parent: "074 BERT"
nav_order: 1
---
# Code Architecture — 074 BERT

## Overview

This notebook implements BERT from scratch in PyTorch, following the original paper's two-phase approach:

1. **Pre-training** — Masked Language Model (MLM) + Next Sentence Prediction (NSP) on a toy corpus
2. **Fine-tuning** — Binary text classification on a small sentiment task

The model is intentionally tiny (2 Transformer layers, 128 hidden dim, 4 attention heads, a 190-word vocabulary). It shows how BERT's parts connect, but it barely learns: see the notebook's Summary cell.

## Section-by-Section Breakdown

### 1. Imports & Configuration
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, `random`, `collections`
- Hyperparameters: `d_model=128`, `n_heads=4`, `n_layers=2`, `vocab_size` (built from corpus), `max_len=64`, `batch_size=32`, `lr=1e-4`, `mlm_epochs=15`, `finetune_epochs=10`

### 2. Toy Corpus Construction
- A small collection of sentence pairs drawn from simple English sentences
- Each entry is a pair (sentence A, sentence B) where B is either the actual next sentence or a random sentence
- This provides both MLM training data (masking tokens within sentences) and NSP data (is-B-next-A?)
- A simple word-level tokenizer (split on whitespace, lowercase) — deliberately simpler than WordPiece for clarity

### 3. Vocabulary Building
- Build a vocabulary from all words in the toy corpus
- Special tokens: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`, `[MASK]=4`
- Word-to-id and id-to-word mappings
- 190 tokens in the saved run

### 4. Data Preparation — Pre-training
- **MLM masking function:** Given a token sequence, select 15% of positions. Of those: 80% → `[MASK]`, 10% → random token, 10% → unchanged. Return masked input, labels (original tokens at masked positions, -100 elsewhere for loss masking).
- **NSP labels:** 1 if B follows A, 0 otherwise.
- **Input construction:** `[CLS] A [SEP] B [SEP]` → token IDs + segment IDs (0 for A, 1 for B) + position IDs
- **Dataset class:** Returns (input_ids, segment_ids, attention_mask, mlm_labels, nsp_label)

### 5. BERT Model Architecture

#### 5a. Multi-Head Self-Attention
- Standard scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
- Multi-head: split `d_model` into `n_heads` heads, attend independently, concat, project
- **Key BERT feature:** NO causal mask — attention is fully bidirectional (unlike GPT's decoder)

#### 5b. Position-wise Feed-Forward
- `FFN(x) = GELU(xW1 + b1)W2 + b2` with inner dimension `d_model * 4`

#### 5c. Transformer Encoder Layer
- `LayerNorm(x + MultiHeadAttn(x))`
- `LayerNorm(x + FFN(x))`
- Post-LN architecture (as in original BERT, not pre-LN)

#### 5d. BERT Model
- **Embedding layer:** `token_emb + segment_emb + position_emb` (all learned, all same dimension)
- **Encoder:** stack of `n_layers` Transformer encoder layers
- **MLM head:** Linear → GELU → LayerNorm → output projection over vocabulary
- **NSP head:** Linear(2) on the `[CLS]` token's output
- **Pooler:** Dense → Tanh applied to `[CLS]` output for NSP

#### 5e. GELU Activation
- `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` — the exact approximation used in BERT

### 6. Pre-training Loop
- For each epoch:
  - Forward pass: get MLM logits and NSP logits
  - MLM loss: CrossEntropy over masked positions only (ignore_index=-100)
  - NSP loss: CrossEntropy over binary label
  - Total loss = MLM loss + NSP loss
  - Backprop, Adam optimizer step
- Print loss every few epochs; plot loss curve at end

### 7. Fine-tuning Task — Sentiment Classification
- Small set of positive/negative sentences as the downstream task
- Input: `[CLS] sentence [SEP]` with segment ID all 0
- Classification head (`BERTForClassification`): take `[CLS]` output → the existing pooler (Dense → Tanh) → Dropout → Linear(2)
- Fine-tune ALL parameters (including pre-trained embeddings and encoder) for `finetune_epochs`
- Print training accuracy and plot fine-tuning loss

### 8. Evaluation & Visualisation
- Report final MLM accuracy (how often the model correctly predicts masked tokens)
- Report final NSP accuracy
- Report fine-tuning classification accuracy
- Loss curves for both pre-training and fine-tuning phases (matplotlib)
- Print sample masked-token predictions (in the saved run they barely change with the sentence)

## Key Functions/Classes

| Name | Type | Purpose |
|---|---|---|
| `GELU` | Module | Gaussian Error Linear Unit activation |
| `MultiHeadAttention` | Module | Bidirectional self-attention with n_heads |
| `TransformerEncoderLayer` | Module | One layer of BERT encoder (attn + FFN + residuals + LN) |
| `BERTEmbedding` | Module | Token + segment + position embeddings |
| `BERTModel` | Module | Full BERT: embeddings → encoder → MLM head + NSP head |
| `apply_mlm_mask` | Function | Applies 80/10/10 masking to a token sequence |
| `PretrainDataset` | Dataset | Yields (input_ids, segment_ids, mask, mlm_labels, nsp_label) |
| `FinetuneDataset` | Dataset | Yields (input_ids, segment_ids, mask, label) |
| `train_pretraining` | Function | Pre-training loop (MLM + NSP) |
| `train_finetuning` | Function | Fine-tuning loop for classification |

## Data Flow / Shapes

```
Input sentence pair → tokenize → [CLS] tok_A... [SEP] tok_B... [SEP]
→ input_ids: [batch, seq_len] (int)
→ segment_ids: [batch, seq_len] (0 or 1)
→ position_ids: [batch, seq_len] (0..seq_len-1)
→ embeddings: [batch, seq_len, 128]
→ encoder layers (x n_layers): [batch, seq_len, 128]
→ MLM head: [batch, seq_len, vocab_size] → CrossEntropy at masked positions
→ NSP head: [batch, 2] → CrossEntropy on [CLS] output
```

## Deliberate Simplifications vs Full Paper

| Simplification | Original BERT | Our Version | Rationale |
|---|---|---|---|
| Vocabulary | 30k WordPiece | 190 word-level | Simpler, no tokeniser dependency |
| Model size | 12-24 layers, 768-1024 hidden | 2 layers, 128 hidden | Small enough to run on a CPU |
| Pre-training data | BooksCorpus + Wikipedia | 400 toy sentence pairs | Demonstrate mechanism, not scale |
| Batch size | 256 | 32 | Smaller dataset needs smaller batch |
| Training steps | Far more than here | About 190 (15 epochs) | Too few to learn much, as the Summary cell reports |
| NSP task | Full binary classification | Same | Faithful to original |
| Token masking | WordPiece sub-tokens | Word-level tokens | Conceptually identical |
| Fine-tuning tasks | 11 tasks (GLUE, SQuAD, etc.) | One binary sentiment task | Demonstrates the fine-tuning paradigm |
| Pooler activation | Tanh | Tanh | Faithful |
| Optimizer | Adam at `lr=1e-4`, with learning-rate warmup | Adam at `lr=1e-4`, no warmup | Simpler |
| Position embeddings | Learned, max 512 | Learned, max 64 | Sufficient for short toy sequences |
