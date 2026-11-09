---
title: "073 ELMo"
nav_order: 74
permalink: "/073_deep-contextualized-word-representations-elmo/"
---
# Deep Contextualized Word Representations (ELMo)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deep-contextualized-word-representations-elmo)

- **Paper:** [Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365)
- **Authors:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
- **Lab, year:** Allen Institute for AI and University of Washington, 2018 (NAACL 2018)
- **Video:** YouTube Short coming on 8 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

ELMo (Embeddings from Language Models) gives each word a vector that depends on the sentence around it. Word2Vec and GloVe assign one fixed vector to each word, so "bank" looks the same in every sentence. ELMo instead uses the internal states of a pretrained bidirectional language model: LSTMs (recurrent networks with memory) that read the sentence in both directions.

The final representation is a weighted sum of the model's layers, and the weights are learned separately for each task. The paper finds that the top layer is better at word sense disambiguation and the first layer is better at part-of-speech tagging, so a task can pick the mix it needs.

Adding ELMo set a new state of the art on all six benchmark tasks in the paper, with relative error reductions of 6 to 20% in the text and 5.8% to 24.9% in Table 1.

## What Problem Does It Solve

Imagine you have a dictionary where the word "bank" has only one definition. But in real life, "bank" can mean the side of a river ("I sat by the bank of the river") or a place that keeps money ("I deposited cash at the bank"). Old word-embedding systems like Word2Vec gave "bank" the same vector no matter which meaning you intended — it was like a dictionary with only one entry per word.

ELMo fixes this by looking at the whole sentence before deciding what "bank" means. It's like a smart friend who knows that if you mention "river" and "sitting," you probably mean the riverbank, and if you mention "deposit" and "cash," you mean the money bank. ELMo reads the entire sentence, understands the context, and gives "bank" a different vector in each sentence. This means computers can finally tell apart words that are spelled the same but mean different things — just like humans do.

## Core Idea

Train a deep bidirectional language model on a large unlabeled corpus, freeze it, and use all of its layers as word features for other tasks. Each task learns only how to mix the layers:

**ELMo = γ · Σ_j s_j · h_j**

Here h_j is layer j of the language model, s_j are weights normalized to sum to 1 (a softmax), and γ scales the result. The same word in two sentences gets two different vectors because the language model reads the whole sentence.

## Key Method Details

- **Bidirectional language model (biLM):** a forward LSTM predicts the next token and a backward LSTM predicts the previous one. They share a character-level CNN token encoder. The notebook uses a simpler word-level embedding.
- **Layers:** at each LSTM layer the forward and backward states are concatenated. The input token embedding counts as layer 0.
- **Task-specific weights:** the biLM's weights are frozen. The task model learns s_j and γ, and the ELMo vector is concatenated with the task model's input (Section 3.3).
- **Paper scale:** 2 biLSTM layers of 4096 units with 512-dimension projections, trained on the 1B Word Benchmark.

### What the Notebook Result Shows

The notebook trains a 2-layer biLM with 256 hidden units on 30 sentences for 80 epochs. The loss falls from **9.55 to 0.24**, and with no held-out data the model is memorizing the corpus.

Vectors do change with context: average cosine similarity between uses of "bank" is **1.00, 0.74 and 0.55** at layers 0, 1 and 2, and "play" and "right" fall the same way. Word senses are not separated, though. At layer 2, "right" is **0.24** for same-sense pairs and **0.41** for different-sense pairs, and "bank" and "play" also score higher for different senses.

On a toy "bank" sense task, the mean learned weights over 5 seeds are **0.23, 0.37 and 0.40** for layers 0, 1 and 2. Every seed reaches 8/8 on its own 8 training sentences, with no test set, so this does not show that different tasks prefer different layers.

## Influence

ELMo was one of the pretrained language models, with ULMFiT and the OpenAI transformer, that moved NLP from static word vectors towards pretrained representations ([The Gradient, 2018](https://thegradient.pub/nlp-imagenet/)). BERT followed later in 2018. The paper was one of NAACL 2018's four [outstanding papers](https://naacl2018.wordpress.com/2018/04/11/outstanding-papers/), and OpenAlex lists [13,289 citations](https://openalex.org/works/W2787560479) (checked 9 Nov 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/073_deep-contextualized-word-representations-elmo/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, Q&A (answers from the paper), citations, and common misconceptions about the paper |
