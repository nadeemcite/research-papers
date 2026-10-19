---
title: "098 RAG"
nav_order: 99
permalink: "/098_retrieval-augmented-generation-rag/"
---
# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/retrieval-augmented-generation-rag)

- **Paper:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
- **Lab, year:** Facebook AI Research, University College London and New York University, 2020 (NeurIPS 2020)

## Summary

The RAG paper introduced Retrieval-Augmented Generation. It combines a pre-trained **parametric** memory (a seq2seq generator, BART) with a **non-parametric** memory (a dense vector index of Wikipedia, searched with a pre-trained neural retriever, DPR). Instead of storing all facts inside the model's weights, RAG looks up relevant passages at answer time and gives both the question and the passages to the generator.

The paper proposes two versions. **RAG-Sequence** uses the same retrieved document for the whole answer. **RAG-Token** can use a different document for each output token. Both treat the retrieved document as a hidden (latent) variable and sum over the top-k documents.

RAG set a new state of the art on four open-domain QA tasks (Natural Questions, TriviaQA, WebQuestions and CuratedTrec); for TriviaQA this holds only on the T5-comparable test split. It also wrote more specific, diverse and factual text than a BART-only baseline. The name "RAG" is now used widely for the general idea of adding search to language models.

## What Problem Does It Solve

Imagine you're taking an open-book exam. You could try to memorize every fact in the textbook beforehand (that's how regular AI models work — they stuff everything into their "brain", the weights). But that's impractical: you can't memorize everything, facts change over time, and you can't show your teacher where you got each answer.

Instead, imagine being allowed to bring a searchable encyclopedia to the exam. When you get a question, you look up the relevant pages, read them, and then write your answer based on both what you already know AND what you just looked up. That's what RAG does.

The model has a general understanding of language (like your brain), but when it needs a specific fact, it looks up passages from a huge searchable database (the encyclopedia) and writes its answer using them. If a fact changes, like who leads a country, you can update the encyclopedia instead of retraining the whole model. And you can point to the passage you used, so people can check the answer.

## Core Idea

Traditional language models store knowledge in their parameters (weights). This has problems: knowledge is hard to update, models can "hallucinate" (make things up), and they cannot show where an answer came from. RAG splits knowledge into two parts:

1. **Parametric memory** — a pre-trained seq2seq model (the generator, BART) that handles language understanding and generation.
2. **Non-parametric memory** — a dense vector index of Wikipedia passages, searched through a pre-trained neural retriever (DPR, Dense Passage Retriever).

At answer time, given a query **x**:
- The retriever selects the top-k most relevant passages **z**
- The generator produces an output **y** conditioned on both **x** and each retrieved **z**
- The output probability is summed over the retrieved documents: **p(y\|x) ≈ Σ p(z\|x) · p(y\|x,z)**

**RAG-Sequence** uses the same document for the whole output sequence: p(y\|x,z_i) is computed for each document and then summed. **RAG-Token** lets different documents condition different tokens: the sum over documents happens at every generation step.

The retriever's query encoder and the generator are fine-tuned together, end-to-end. No one tells the retriever which document is correct; it learns which passages help the generator. The document encoder and the index stay fixed during training, which keeps the method practical (Section 2.4).

## Key Method Details

- **Retriever:** DPR — two BERT-base encoders, one for questions and one for passages. The paper uses the DPR retriever trained on TriviaQA and Natural Questions as its starting point.
- **Generator:** BART-large, a pre-trained seq2seq transformer with 400M parameters. The input and the retrieved passage are simply joined together.
- **Index:** the December 2018 Wikipedia dump, split into 100-word chunks: 21M documents. Search is Maximum Inner Product Search (MIPS) with FAISS.
- **RAG-Sequence:** p(y\|x) ≈ Σ_i p(z_i\|x) · p(y\|x, z_i) — one document conditions the whole sequence.
- **RAG-Token:** p(y\|x) ≈ Π_j Σ_i p(z_i\|x) · p(y_j\|x, z_i, y_{<j}) — different documents can condition different tokens.
- **Top-k retrieval:** k = 5 or 10 documents during training; k at test time is chosen on dev data.
- **Training:** the query encoder and BART are fine-tuned jointly (Adam), minimizing the negative marginal log-likelihood. Tasks: open-domain QA (Natural Questions, TriviaQA, WebQuestions, CuratedTrec), abstractive QA (MS MARCO), Jeopardy question generation and FEVER fact verification.
- **Knowledge updating:** because the memory is an index, it can be swapped. With a 2016 Wikipedia index, RAG answered 70% of questions about 2016 world leaders correctly; with the 2018 index, 68% for 2018 leaders. Mismatched indexes gave only 12% and 4% (Section 4.5).
- **Results (Table 1, RAG-Sequence, exact match):** Natural Questions 44.5, TriviaQA 56.8 (68.0 on the TQA-Wiki split), WebQuestions 45.2, CuratedTrec 52.2. These are state of the art for NQ, WebQuestions and CuratedTrec. On TriviaQA, only the 68.0 on the TQA-Wiki split is state of the art; on the standard test set, DPR is higher (57.9 vs 56.8). RAG-Token is best on WebQuestions (45.5).

### Simplifications in This Notebook

- A toy index of **40 short passages** instead of 21M Wikipedia chunks
- Retriever: one pre-trained `all-MiniLM-L6-v2` sentence encoder (384-dim) instead of DPR's two BERT encoders
- Generator: `google/flan-t5-base` (247.6M parameters) instead of BART-large (400M)
- **No training at all:** off-the-shelf models, no end-to-end fine-tuning
- RAG-Sequence picks the single best candidate by "retrieval probability + log-probability" instead of summing over documents; RAG-Token uses greedy decoding instead of beam search

### What the Notebook Result Shows

On 5 test questions, both RAG versions found the right answer every time (for example "Leonardo da Vinci", "330 meters", "299,792 kilometers per second"). The parametric-only baseline was wrong on all 5 (for example "franz heidegger" for the Mona Lisa and "84 m" for the Eiffel Tower). So retrieval clearly helped this small generator. But the questions match the toy passages almost word for word, so this is an easy test.

Two parts did not work as described. For the Einstein question, RAG-Sequence answered "Nobel Prize" but picked a passage about Marie Curie as its source. And in the knowledge-update demo, the answer was **"New York Harbor" both before and after** adding the Atlantis passage, because the Statue of Liberty passage still ranked first. The cell's printed message says the answer changed, but in the saved run it did not.

## Influence

The RAG paper gave its name to an approach that is now standard: give a language model a search step so it can use outside documents and point to its sources. Its key idea — separating parametric knowledge (model weights) from non-parametric knowledge (an external index) — shaped much later work. Follow-ups include RETRO (DeepMind, retrieval from trillions of tokens) and Atlas (retrieval-augmented few-shot learning), and open-source frameworks such as LlamaIndex, LangChain and Haystack build "retrieve, then generate" pipelines. Many production systems skip the paper's end-to-end fine-tuning, but the basic recipe is the same.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/098_retrieval-augmented-generation-rag/solution.ipynb) | Compact RAG pipeline (dense retriever + FAISS + Flan-T5 generator) with RAG-Sequence, RAG-Token, a parametric-only baseline and a knowledge-update demo |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Blog posts, Q&A (answers from the paper), citations, and common misconceptions about the paper |
