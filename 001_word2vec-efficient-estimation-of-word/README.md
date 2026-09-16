# Word2Vec — Efficient Estimation of Word Representations in Vector Space

**Paper:** [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
**Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google Inc.)
**Year:** 2013

---

## Summary

Before Word2Vec, NLP systems treated words as atomic units — discrete indices in a vocabulary with no notion of similarity. "King" and "queen" were as different as "king" and "toaster." Neural language models existed but were too slow to train on billion-word datasets. Mikolov et al. proposed two lightweight architectures — **Continuous Bag-of-Words (CBOW)** and **Skip-gram** — that learn dense vector representations of words from raw text at unprecedented scale and speed. On a single machine, high-quality vectors could be trained from 1.6 billion words in under a day.

## The Core Idea

Strip the neural network language model down to almost nothing. No hidden layers, no nonlinearities, no complex preprocessing. Just a lookup table (embedding matrix) and a softmax classifier. **Skip-gram** predicts context words from a center word; **CBOW** predicts a center word from its context. By removing the hidden layer entirely, training becomes so cheap that you can throw billions of words at it and still finish in hours on a commodity machine. The resulting vectors capture semantic and syntactic relationships — most famously, vector arithmetic reveals analogies like `king − man + woman ≈ queen`.

## Key Method Details

### Skip-gram Architecture
- **Input:** A center word (one-hot vector of dimension V = vocab size).
- **Embedding:** Multiply one-hot by the input embedding matrix W (V×d) to get the word's d-dimensional vector.
- **Output:** Predict each context word within a window of size C using a softmax over the vocabulary: `p(w_c | w_t) = exp(v'_{w_c}^T v_{w_t}) / Σ exp(v'_w^T v_{w_t})`
- **Two matrices:** Input embeddings W and output embeddings W'. The final word vectors are typically W (the input embeddings).

### CBOW Architecture
- Reverse of Skip-gram: average the embeddings of context words, then predict the center word.

### Negative Sampling (from the follow-up paper, arXiv:1310.4546)
- Full softmax over the entire vocabulary is expensive (O(V) per training step).
- **Negative sampling** replaces the softmax with a binary classification task: for each (center, context) pair, sample k random "negative" words from a noise distribution (unigram^0.75). The loss becomes:
  `L = −log σ(v'_{w_c}^T v_{w_t}) − Σ_{k} log σ(−v'_{w_n}^T v_{w_t})`
- This reduces computation from O(V) to O(k+1) per step, enabling training on massive corpora.

### Training Setup (from the paper)
- 1.6 billion words (Google News dataset)
- Vocabulary: 3 million words
- Embedding dimensions: tested 100–1000
- Training time: less than a day on a single machine
- Evaluation: word similarity tasks and analogy tasks (semantic + syntactic)

## Why It Mattered

Word2Vec was a watershed moment for NLP. It demonstrated that meaningful semantic representations could be learned from raw text at scale, without labels. The word analogy discovery (`king − man + woman = queen`) became one of the most iconic results in machine learning, proving that vector spaces encode structured relationships. Word2Vec embeddings became the default input representation for NLP tasks (sentiment analysis, machine translation, named entity recognition) for years, until contextual embeddings (ELMo, BERT) superseded them. The Skip-gram + negative sampling recipe also directly inspired embedding methods in other domains — node2vec for graphs, item2vec for recommendations, and more.

## Influence Afterward

- **GloVe** (Pennington et al., 2014) — an alternative based on global co-occurrence matrices
- **FastText** (Mikolov et al., 2016) — extends Word2Vec to subword units
- **ELMo / BERT** — contextual embeddings that replaced static Word2Vec vectors
- **Item2Vec, node2vec, doc2vec** — the "2vec" pattern spread across domains
- The concept of learning dense representations from self-supervised objectives became foundational to modern deep learning