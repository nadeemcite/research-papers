---
title: "Word2Vec: Code architecture"
parent: "001 Word2Vec"
nav_order: 1
---
# Code Architecture — Word2Vec (Skip-gram + Negative Sampling)

**Notebook:** `solution.ipynb`
**Task:** Implement Skip-gram with Negative Sampling from scratch in NumPy, train on a small text corpus, visualize with t-SNE/PCA, and demonstrate word analogies.

---

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import NumPy, Matplotlib, and optionally scikit-learn (for t-SNE/PCA).
- No deep learning framework — everything is raw NumPy to show the mechanics clearly.

### 2. Corpus Loading & Preprocessing
- A small hand-written corpus of 14 sentences (a king, a queen, a prince and a dragon) is built in memory.
- Tokenize: lowercase and keep runs of letters (`re.findall`).
- Build vocabulary: map each word to an integer index, dropping words that appear fewer than `min_count=2` times (26 words and 112 tokens remain).
- **Function:** `preprocess_corpus(text, min_count)` → `tokens, counts, vocab, word2idx, idx2word, token_ids`.

### 3. Skip-gram Training Pairs Generation
- For each center word in the corpus, generate (center, context) pairs for all words within the window.
- **Function:** `generate_skip_gram_pairs(token_ids, window_size)` → list of (center_idx, context_idx) tuples.
- **Data flow:** `token_ids [N_tokens]` → sliding window → `pairs [N_pairs, 2]`

### 4. Negative Sampling Distribution
- Compute unigram frequency distribution from the corpus.
- Raise to the 0.75 power and normalize (this downweights frequent words, upweights rare ones).
- **Function:** `build_negative_sampling_table(counts, vocab, table_size=1_000_000)` → lookup table for O(1) negative sampling.
- **Data flow:** `counts [V]` → `p(w)^0.75` → normalized → sampling table

### 5. Model Parameters (Embedding Matrices)
- **W_in** (V × d): input embeddings (the final word vectors).
- **W_out** (V × d): output embeddings (used during training, typically discarded after).
- Initialized with small random values (uniform [−0.5/d, 0.5/d]).
- **Shapes:** W_in: `(vocab_size, embedding_dim)`, W_out: `(vocab_size, embedding_dim)`

### 6. Training Loop (Skip-gram + Negative Sampling)
- For each (center, context) pair:
  1. Get `v_c = W_in[center]` (d-dim vector).
  2. Sample k negative words from the sampling table.
  3. For the positive context word and all k negatives:
     - Compute dot product `v_c · v_w` (W_out row).
     - Apply sigmoid: `σ(v_c · v_w)`.
     - Compute loss: `−log σ(v_pos) − Σ log σ(−v_neg)`.
  4. Compute gradients for W_in[center] and W_out[positive/negative] rows.
  5. Update with a constant learning rate (no decay).
- **Key function:** `train_skipgram_neg(pairs, W_in, W_out, neg_table, epochs=80, lr=0.03, k=5)`
- **Data flow per step:**
  - Input: `(center_idx, context_idx)` → `v_c [d]` → dot products with `W_out[pos] [d]` and `W_out[neg_k] [k, d]` → sigmoid → loss scalar → gradients `[d]` → update

### 7. Training Loss Plot
- Plot the average loss per epoch to show convergence (4.1584 at epoch 1 to 2.3532 at epoch 80).

### 8. Evaluation: Nearest Neighbors
- Compute cosine similarity between a query word and all other words.
- **Function:** `most_similar(word, W_in, word2idx, idx2word, topn=10)` → list of (word, similarity).
- With only 112 tokens the neighbours are weak: the closest words to "king" are mostly common words.

### 9. Visualization: PCA + t-SNE
- Inline code (no helper function): reduce `W_in [V, d]` to 20 dimensions with PCA, then to 2D with t-SNE, and draw a labelled matplotlib scatter plot.

### 10. Word Analogies
- Vector arithmetic `v_a − v_b + v_c`, then the nearest word to the result, excluding the three input words.
- **Function:** `analogy(a, b, c, W_in, word2idx, idx2word, topn=5)`.
- Queries: `queen − king + prince` (hoping for "princess") and `prince − princess + queen` (hoping for "king"). Both fail on this tiny corpus: neither expected word reaches the top five.

---

## Key Functions/Classes

| Function | Responsibility |
|---|---|
| `preprocess_corpus(text)` | Tokenize, build vocabulary, return token IDs |
| `generate_skip_gram_pairs(tokens, window)` | Create (center, context) training pairs |
| `build_negative_sampling_table(counts, vocab, table_size)` | Precompute noise distribution for negative sampling |
| `train_skipgram_neg(...)` | Main training loop with gradient updates |
| `most_similar(word, ...)` | Cosine-similarity nearest neighbors |
| `analogy(a, b, c, ...)` | Vector arithmetic for word analogies |

---

## Deliberate Simplifications vs. the Original Paper

| Simplification | Original Paper | Notebook | Why |
|---|---|---|---|
| Corpus size | Up to 6 billion tokens (Google News); 1.6B-word subsets were used for some comparisons | 112 tokens (14 hand-written sentences) | Runnable in minutes on Colab; still demonstrates the mechanism |
| Vocabulary | The 1 million most frequent words | 26 words | Same — keeps training fast and visualization readable |
| Embedding dim | 50–1000 | 50 | Small vocab doesn't need high dimensions |
| Subsampling of frequent words | Not in this paper (introduced in the follow-up, arXiv:1310.4546) | Omitted | Adds complexity without changing the core algorithm |
| Negative samples (k) | Not in this paper; the follow-up suggests 5–20 for small data and 2–5 for large | 5 | Standard value, sufficient for small corpus |
| Output layer | Hierarchical softmax (the main method of this paper) | Negative sampling (from the follow-up paper) | Negative sampling is more commonly used and simpler to implement |
| Training epochs | 1–5 passes over billions of words | 80 epochs over small corpus | Small corpus needs more passes to converge |
| Evaluation | Google analogy test set (20k pairs) | Manual nearest-neighbor + analogy checks | Full test set requires large pretrained vectors |