# Code Architecture — Word2Vec (Skip-gram + Negative Sampling)

**Notebook:** `solution.ipynb`
**Task:** Implement Skip-gram with Negative Sampling from scratch in NumPy, train on a small text corpus, visualize with t-SNE/PCA, and demonstrate word analogies.

---

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import NumPy, Matplotlib, and optionally scikit-learn (for t-SNE/PCA).
- No deep learning framework — everything is raw NumPy to show the mechanics clearly.

### 2. Corpus Loading & Preprocessing
- Load a small text corpus (e.g., a few paragraphs from Project Gutenberg or a custom sentence collection).
- Tokenize: lowercase, split on whitespace, strip punctuation.
- Build vocabulary: map each unique word to an integer index, filter by minimum frequency.
- **Outputs:** `word2idx` dict, `idx2word` list, `vocab_size`, `token_ids` list.

### 3. Skip-gram Training Pairs Generation
- For each center word in the corpus, generate (center, context) pairs for all words within the window.
- **Function:** `generate_skip_gram_pairs(token_ids, window_size)` → list of (center_idx, context_idx) tuples.
- **Data flow:** `token_ids [N_tokens]` → sliding window → `pairs [N_pairs, 2]`

### 4. Negative Sampling Distribution
- Compute unigram frequency distribution from the corpus.
- Raise to the 0.75 power and normalize (this downweights frequent words, upweights rare ones).
- **Function:** `build_negative_sampling_table(freqs, table_size=1e8)` → lookup table for O(1) negative sampling.
- **Data flow:** `word_counts [V]` → `p(w)^0.75` → normalized → sampling table

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
  5. Update with learning rate.
- **Key function:** `train_skipgram_neg(pairs, W_in, W_out, epochs, lr, k_neg, neg_table)`
- **Data flow per step:**
  - Input: `(center_idx, context_idx)` → `v_c [d]` → dot products with `W_out[pos] [d]` and `W_out[neg_k] [k, d]` → sigmoid → loss scalar → gradients `[d]` → update

### 7. Evaluation: Nearest Neighbors
- Compute cosine similarity between a query word and all other words.
- **Function:** `most_similar(word, W_in, word2idx, idx2word, topn=10)` → list of (word, similarity).
- Show results for a few query words to verify the embeddings learned meaningful relationships.

### 8. Visualization: t-SNE / PCA
- Reduce the embedding matrix from d dimensions to 2D using PCA or t-SNE.
- Scatter plot with word labels for a subset of the vocabulary.
- **Data flow:** `W_in [V, d]` → `PCA/t-SNE` → `2D coords [V, 2]` → matplotlib scatter

### 9. Word Analogies
- Implement vector arithmetic: `king − man + woman`.
- Find the nearest word to the result vector (excluding the input words).
- **Function:** `analogy(word_a, word_b, word_c, W_in, ...)` → nearest word to `v_a − v_b + v_c`.
- Show several analogies (if the corpus is large enough to contain the relevant words).

### 10. Training Loss Plot
- Track and plot the average loss per epoch to show convergence.

---

## Key Functions/Classes

| Function | Responsibility |
|---|---|
| `preprocess_corpus(text)` | Tokenize, build vocabulary, return token IDs |
| `generate_skip_gram_pairs(tokens, window)` | Create (center, context) training pairs |
| `build_neg_sampling_table(freqs)` | Precompute noise distribution for negative sampling |
| `train_skipgram_neg(...)` | Main training loop with gradient updates |
| `most_similar(word, ...)` | Cosine-similarity nearest neighbors |
| `analogy(a, b, c, ...)` | Vector arithmetic for word analogies |
| `visualize_embeddings(W_in, ...)` | t-SNE/PCA 2D scatter plot |

---

## Deliberate Simplifications vs. the Original Paper

| Simplification | Original Paper | Notebook | Why |
|---|---|---|---|
| Corpus size | 1.6 billion words (Google News) | ~1,000–5,000 words (small text) | Runnable in minutes on Colab; still demonstrates the mechanism |
| Vocabulary | 3 million words | ~200–500 words | Same — keeps training fast and visualization readable |
| Embedding dim | 100–1000 | 50–100 | Small vocab doesn't need high dimensions |
| Subsampling of frequent words | Yes (improves quality) | Omitted | Adds complexity without changing the core algorithm |
| Negative samples (k) | 5–15 | 5 | Standard value, sufficient for small corpus |
| Hierarchical softmax | Also proposed in paper | Not implemented | Negative sampling is more commonly used and simpler to implement |
| Training epochs | 1–5 passes over billions of words | 50–100 epochs over small corpus | Small corpus needs more passes to converge |
| Evaluation | Google analogy test set (20k pairs) | Manual nearest-neighbor + analogy checks | Full test set requires large pretrained vectors |