# Sequence to Sequence Learning with Neural Networks

**Paper:** Sequence to Sequence Learning with Neural Networks  
**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le  
**Venue:** Advances in Neural Information Processing Systems (NeurIPS), 2014  
**arXiv:** https://arxiv.org/abs/1409.3215
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/sequence-to-sequence-learning-with-neural-networks)

## One-paragraph summary

This paper introduced the now-standard "sequence-to-sequence" (seq2seq) recipe for end-to-end learning of variable-length input–output mappings. The idea is disarmingly simple: use one deep LSTM to read the input sequence left-to-right and compress it into a single fixed-size vector (the last hidden state), then use a second deep LSTM as a language model whose initial hidden state is that vector, generating the output sequence one token at a time. The whole system is trained with ordinary maximum-likelihood on paired sequences. On WMT'14 English→French translation the model reached a BLEU score of 34.8 (phrase-based SMT scored 33.3 on the same data), and it handled long sentences far better than expected. The paper also reported a famous practical trick: reversing the source sentence improved performance markedly, because it put related source and target words closer together in time and created easier short-term dependencies.

## Core idea

Most deep learning models of 2014 needed fixed-size inputs and outputs. Sutskever, Vinyals and Le showed that you can sidestep this limitation by chaining two recurrent models: an **encoder** that consumes the input and produces a compact representation, and a **decoder** that turns that representation back into a sequence. Nothing in the architecture is specific to translation—the same encoder/decoder template later powered summarization, image captioning, speech recognition and conversational agents.

## Key method details relevant to the code

- **Encoder:** a multilayer LSTM reads the token sequence `x1, x2, …, xT` and returns its final hidden state `v`.
- **Decoder:** another multilayer LSTM (different parameters) starts from `v` and emits `y1, y2, …, yT'` with a softmax over the target vocabulary at every step.
- **Objective:** maximize the log probability of the correct target given the source, i.e. maximize `Σ log p(T | S)` over the training pairs.
- **Inference:** left-to-right beam search over the decoder. A beam size of 1 already worked well; size 2 captured most of the gain.
- **Reversing trick:** the source sentence is read in reverse order while the target stays normal. This places the first source word close to the first target word and makes gradient flow easier.
- **Deep LSTMs:** the paper used four-layer LSTMs in both encoder and decoder; deep stacks helped quality.
- **End-of-sequence token:** a special `<EOS>` token lets the model decide when to stop generating, so output length is learned, not fixed.

## Influence

The seq2seq architecture became the dominant neural approach to machine translation before attention and Transformers took over. Its direct descendants include Bahdanau attention (ICLR 2015), Google's Neural Machine Translation system deployed in Google Translate in 2016, and modern encoder/decoder Transformers such as T5 and BART. OpenAlex records more than 13,000 citations for the paper as of 2026, and the tf-seq2seq / PyTorch "NLP from Scratch" tutorials still teach the same pattern.

## What problem does it solve

Imagine you want to turn the English sentence "the cat sat on the mat" into French. Older neural networks were like a calculator that only accepts numbers of a fixed length: you could not feed them a sentence of, say, six words today and eight words tomorrow. This paper solves that mismatch by treating the sentence as a stream of tokens: one LSTM reads the whole English stream and summarizes it as a "thought vector"; a second LSTM unrolls that thought vector into French words, one at a time, until it decides it is done. A real-life everyday analogy is dictating a voice message to a friend in your language and getting a typed translation in theirs—the model has to understand a whole variable-length utterance and produce a different variable-length utterance, not just classify a fixed form.
