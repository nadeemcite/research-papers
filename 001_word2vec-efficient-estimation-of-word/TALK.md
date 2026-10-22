---
title: "Word2Vec: Talk & Q&A"
parent: "001 Word2Vec"
nav_order: 2
---
# TALK.md — Word2Vec: Context, Discussion, and Q&A

**Paper:** [Efficient Estimation of Word Representations in Vector Space (arXiv:1301.3781)](https://arxiv.org/abs/1301.3781)

---

## Notable Press Coverage, Blog Posts, and Announcements

1. **word2vec on Google Code (2013)** — Google open-sourced the word2vec tool in 2013. The original project page is kept in the [Google Code Archive](https://code.google.com/archive/p/word2vec/).

2. **"word2vec Explained" by Goldberg & Levy (2014)** — A widely-cited note that derives the negative-sampling objective step by step, clarifying the short original description: [arXiv:1402.3722](https://arxiv.org/abs/1402.3722)

3. **gensim Library** — The open-source gensim library made Word2Vec easy to use and became a common tool for training custom embeddings: [gensim word2vec docs](https://radimrehurek.com/gensim/models/word2vec.html)

---

## Talks and Articles with the Author

1. **Towards Data Science conversation with Tomas Mikolov (February 2023)** — [Uncovering the Pioneering Journey of Word2Vec and the State of AI Science](https://towardsdatascience.com/uncovering-the-pioneering-journey-of-word2vec-and-the-state-of-ai-science-an-in-depth-interview-fbca93d8f4ff/). Mikolov says: "I did not come up with Word2Vec when I worked at Google; I already did before then. The first thing I did, similar to Word2Vec, was during my master thesis in 2006." He also says the Word2Vec paper was rejected at the first ICLR (2013) and later accepted at a workshop, and that the follow-up paper with extensions was accepted to NIPS.

2. **IEEE Signal Processing Society, "Industry Leaders" (July 2021)** — [Industry Leaders in Signal Processing and Machine Learning: Tomas Mikolov](https://signalprocessingsociety.org/our-story/news-updates/2021/07/industry-leaders-signal-processing-and-machine-learning-tomas-mikolov). He notes that word2vec was not the original aim: "these models ended up as simplifications of more ambitious projects (which failed)."

---

## Q&A (answers from the paper)

The paper is [arXiv:1301.3781](https://arxiv.org/abs/1301.3781). Answers about negative sampling come from the follow-up paper, [arXiv:1310.4546](https://arxiv.org/abs/1310.4546) (Section 2.2).

**Q1: What is the difference between Skip-gram and CBOW?**
A (from the paper, Sections 3.1, 3.2 and 4.3): CBOW predicts the current word from the words around it (their vectors are averaged, so word order does not matter). Skip-gram does the opposite: it uses the current word to predict the words around it. In Table 3, CBOW does better on the syntactic analogy questions, while Skip-gram is slightly worse on syntax but much better on the semantic questions than all other models. CBOW also trains faster: on Google News, CBOW took about a day and Skip-gram about three days (Section 4.3, Table 5).

**Q2: Why does negative sampling work, and how is it different from hierarchical softmax?**
A (from the follow-up paper, Sections 2.1 and 2.2): Negative sampling turns the problem into binary classification: is this (word, context) pair real, or drawn from a noise distribution? Instead of normalising over the whole vocabulary, each step only uses the true context word and k noise words. The authors found k = 5–20 useful for small datasets and k = 2–5 enough for large ones. Hierarchical softmax, used in the original paper, instead puts the vocabulary in a binary (Huffman) tree, so each prediction costs about log2(V) steps.

**Q3: Why do word analogies like "king − man + woman ≈ queen" work?**
A (from the paper, Sections 1 and 5): The paper builds on earlier work showing that simple vector arithmetic captures relations between words, for example that vector("King") − vector("Man") + vector("Woman") is closest to vector("Queen"). Because the vectors are trained to predict context, words used in similar ways get similar vectors, and consistent relations show up as roughly consistent offsets. Section 5 (Table 8) lists more examples, such as country–capital (France–Paris) and adjective–comparative (big–bigger) pairs.

**Q4: What are the limitations of Word2Vec compared to modern contextual embeddings like BERT?**
A (general background, not from the paper): Word2Vec gives each word one fixed vector, so "bank" has the same vector for a river bank and a money bank. Contextual models like ELMo and BERT give a different vector for each use of a word. Word2Vec also has a fixed vocabulary and cannot handle unseen words (FastText later added subword pieces for this).

**Q5: Why is the noise distribution raised to the 3/4 power?**
A (from the follow-up paper, Section 2.2): The authors tried several noise distributions and found that the unigram distribution raised to the 3/4 power clearly beat both the plain unigram and the uniform distribution. Raising to 3/4 makes very common words ("the", "and") a bit less likely to be picked as negatives and rare words a bit more likely.

---

## Common Misconceptions

1. **"Word2Vec is a neural network with hidden layers."** — No. The key innovation was removing the hidden layer entirely. Skip-gram and CBOW are log-linear models: just an embedding lookup followed by a (softmax or sigmoid) output. There are no hidden layers or nonlinear activations.

2. **"Word2Vec and GloVe are the same thing."** — They produce similar outputs (static word vectors) but use fundamentally different approaches. Word2Vec learns by predicting context (a local, sliding-window objective). GloVe factorizes the global word co-occurrence matrix (a count-based objective). Word2Vec is trained online via stochastic gradient descent; GloVe is trained on precomputed co-occurrence statistics.

3. **"Negative sampling is just random sampling."** — It's not uniform random. It uses a specific noise distribution (unigram^0.75) that balances frequent and rare words. The choice of distribution significantly affects embedding quality.

4. **"More negative samples is always better."** — No. More negatives make each step slower. The follow-up paper (Section 2.2) found k = 5–20 useful for small datasets, while for large datasets k can be as small as 2–5.

5. **"The analogy king − man + woman = queen proves the model 'understands' gender."** — It demonstrates that the embedding space has linear structure, but the model has no concept of gender. The offset emerges from statistical patterns in text, not from any explicit encoding of semantic features.