---
title: "011 Seq2Seq"
nav_order: 12
permalink: "/011_sequence-to-sequence-learning-with-neural-networks/"
---
# Sequence to Sequence Learning with Neural Networks
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/sequence-to-sequence-learning-with-neural-networks)

- **Paper:** [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- **Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- **Lab, year:** Google, 2014 (NIPS 2014, now NeurIPS)
- **Video:** [Watch the YouTube Short](https://youtube.com/shorts/jcLW_NY0aOs) on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

This paper introduced the sequence-to-sequence (seq2seq) recipe for learning to turn one variable-length sequence into another. One deep LSTM, a recurrent network with gated memory, reads the input and compresses it into a single fixed-size vector. A second deep LSTM starts from that vector and generates the output one token at a time.

On English to French translation from the WMT'14 benchmark, an ensemble of 5 deep LSTMs reached a BLEU score of 34.81, against 33.30 for a phrase-based system on the same data. BLEU measures overlap with human translations. The paper also reported a simple and surprising trick: feeding the source sentence in reverse improved results markedly.

## What Problem Does It Solve

Imagine you want to turn "the cat sat on the mat" into French. Older neural networks were like a form with a fixed number of boxes: you could not give them six words today and eight tomorrow, and the output length had to be known in advance too.

This paper treats the sentence as a stream. One network reads the whole English stream and sums it up in a vector, and a second network unrolls that vector into French words until it decides it is done. It is like listening to a whole sentence, holding the meaning in your head, and only then speaking it in another language.

## Core Idea

Most deep learning models of 2014 needed fixed-size inputs and outputs. Sutskever, Vinyals and Le sidestep that limit by chaining two recurrent models: an **encoder** that consumes the input and produces a compact representation, and a **decoder** that turns that representation back into a sequence.

Nothing in the template is specific to translation. The same encoder/decoder shape later powered summarization, image captioning and speech recognition.

## Key Method Details

- **Encoder:** a multilayer LSTM reads the tokens `x1, x2, …, xT` and returns its final hidden state `v`.
- **Decoder:** a second multilayer LSTM with separate parameters starts from `v` and emits `y1, y2, …, yT'`, with a softmax over the target vocabulary at each step.
- **Objective:** maximize the average log probability of the correct translation T given the source S, written p(T \| S), over the training pairs.
- **Inference:** left-to-right beam search over the decoder. The authors found a beam of size 1 already worked well and size 2 gave most of the benefit.
- **Reversing trick:** the source is read backwards while the target stays in order. On the paper's data, BLEU rose from 25.9 to 30.6.
- **Deep LSTMs:** four layers with 1000 cells each, in both encoder and decoder.
- **End-of-sequence token:** a special `<EOS>` token lets the model decide when to stop, so the output length is learned.

### What the Notebook Result Shows

The notebook trains a 2-layer, 256-unit LSTM encoder and decoder on a synthetic task: converting dates such as "April 17 1949" into "1949-04-17", character by character. After 20 epochs on 18,000 examples, greedy decoding got all 2,000 held-out dates exactly right (100.00%).

That confirms the encoder/decoder mechanics work, but the task is easy: dates follow a fixed rule and the vocabulary is tiny. It says nothing about real translation quality. The notebook always reverses the source, so it does not test whether reversing helps. It also uses greedy decoding instead of beam search.

## Influence

Seq2seq became a dominant neural approach to machine translation before attention and Transformers took over. Its descendants include Bahdanau attention (ICLR 2015), the neural system Google deployed in Google Translate in 2016, and encoder/decoder Transformers such as T5 and BART. [OpenAlex](https://openalex.org/works/W2130942839) records 18,335 citations as of 9 Nov 2026.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/011_sequence-to-sequence-learning-with-neural-networks/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, Q&A (answers from the paper), citations, and common misconceptions about the paper |
