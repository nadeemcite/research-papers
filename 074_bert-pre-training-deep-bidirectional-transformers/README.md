---
title: "074 BERT"
nav_order: 75
permalink: "/074_bert-pre-training-deep-bidirectional-transformers/"
---
# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/bert-pre-training-deep-bidirectional-transformers)

- **Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Lab, year:** Google AI Language, 2018 (NAACL 2019)

## Summary

BERT (Bidirectional Encoder Representations from Transformers) is a way to pre-train a language model on unlabelled text so that it reads every word with context from both sides. A Transformer encoder (a stack of attention layers) lets each token look at all the tokens to its left and right in every layer.

Earlier pre-training was limited to one direction. OpenAI GPT reads left to right, and ELMo joins two separately trained one-directional models. BERT gets around this with a **masked language model**: it hides some input words and predicts them from the surrounding context. A second task, **next sentence prediction**, trains it on pairs of sentences.

After pre-training, the same model is fine-tuned on a labelled task by adding a small output layer. The paper reports state-of-the-art results on eleven natural language processing tasks.

## What Problem Does It Solve

Imagine a fill-in-the-blank test: "The cat sat on the ___ because it was tired." To choose the missing word, you read the words before and after the blank. Older pre-trained models were like reading with your hand covering everything to the right of your finger.

BERT reads the whole sentence at once. During training, it covers up random words and guesses them from everything around them. That practice builds a general understanding of language on text nobody had to label.

The model can then be taught a specific job, such as judging whether a review is positive or finding an answer in a passage. That needs only a small extra layer and a much smaller set of labelled examples.

## Core Idea

- **Deep bidirectional encoding:** every Transformer layer attends to both left and right context, with no mask hiding later tokens.
- **Masked language model:** 15% of token positions are chosen at random. Each is replaced by `[MASK]` 80% of the time, by a random token 10% of the time, and left unchanged 10% of the time. The model predicts the original token.
- **Next sentence prediction:** the model decides whether sentence B really followed sentence A, using the output for the `[CLS]` token. Half the pairs are real and half are random.
- **Fine-tuning:** a task-specific output layer is added, and the whole model is trained end to end on labelled data.

## Key Method Details

- **Input:** `[CLS] sentence A [SEP] sentence B [SEP]`. Token, segment and position embeddings are added together.
- **Model sizes:** BERT-Base has 12 layers, hidden size 768 and 12 attention heads (110M parameters). BERT-Large has 24 layers, hidden size 1024 and 16 heads.
- **Ablation:** in Table 5, removing next sentence prediction hurts performance significantly on QNLI, MNLI and SQuAD 1.1. A left-to-right model does worse on all the tasks tested.

### Simplifications in This Notebook

The notebook builds a 2-layer model with hidden size 128 and 4 heads (463,424 parameters). It pre-trains on 400 toy sentence pairs with a 190-word vocabulary, for 15 epochs. It then fine-tunes for 10 epochs on 32 labelled sentences and tests on 8.

### What the Notebook Result Shows

The code runs end to end, but this tiny model does not learn much. It shows how the parts connect, not the paper's results.

Masked-word accuracy ended at **14.2%**. For all five demo sentences the top guess was "the", at about 4% probability. Next sentence accuracy sat at **53.2%**, exactly the share of real pairs (213 of 400), so the model was not telling real pairs from random ones.

Fine-tuning ended at **53.1%** train accuracy, and test accuracy stayed at **37.5%** (3 of 8) in every epoch, which is worse than guessing. The notebook has no ablation, so it does not test what bidirectionality or next sentence prediction add.

## Influence

BERT won the [Best Long Paper award at NAACL 2019](https://naacl2019.org/blog/best-papers/), and OpenAlex lists about 54,000 citations for the conference record (checked November 2026). Google [released the code and pre-trained models](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/) in November 2018, and in October 2019 said BERT would help [Search understand one in 10 English searches in the U.S.](https://blog.google/products/search/search-language-understanding-bert/) Follow-ups include [RoBERTa](https://arxiv.org/abs/1907.11692), [ALBERT](https://arxiv.org/abs/1909.11942) and [DistilBERT](https://arxiv.org/abs/1910.01108).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/074_bert-pre-training-deep-bidirectional-transformers/solution.ipynb) | From-scratch PyTorch implementation of a tiny BERT, with pre-training, fine-tuning and plots |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
