# Learning Phrase Representations using RNN Encoder–Decoder (GRU)

**Authors:** Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio  
**Published:** EMNLP 2014 (arXiv: 3 Jun 2014, last revised 3 Sep 2014)  
**arXiv:** https://arxiv.org/abs/1406.1078

## Summary

This paper introduces the **RNN Encoder–Decoder**, a pair of recurrent neural networks that jointly learn to map a variable-length source sequence into a fixed-length vector (encoder) and then decode that vector back into a variable-length target sequence (decoder). The two networks are trained jointly to maximise the conditional probability *p(target | source)*. The authors propose a new **Gated Recurrent Unit (GRU)** as the hidden unit inside both networks — a unit with a *reset gate* and an *update gate* that adaptively decides what to forget and what to remember at each time step, giving the recurrent network better memory capacity and easier training than a plain tanh RNN, without the complexity of a full LSTM.

The model was evaluated as an additional phrase-pair scoring feature inside a phrase-based statistical machine translation (SMT) system for English→French on WMT'14. Adding the RNN Encoder–Decoder score to the Moses log-linear model improved BLEU from 33.30 → 33.87 on the test set, and together with a continuous-space language model (CSLM) reached 34.64 BLEU. Qualitatively, the learned phrase representations preserved both semantic and syntactic structure, as shown by t-SNE visualisations of word/phrase embeddings.

### Core idea

The central architectural contribution is the **GRU cell**:

1. **Reset gate** r = σ(W_r·x + U_r·h_{t−1}) — controls how much of the previous hidden state to forget when computing the candidate.
2. **Update gate** z = σ(W_z·x + U_z·h_{t−1}) — controls the interpolation between the old hidden state and the new candidate.
3. **Candidate** h̃ = tanh(W·x + U·(r ⊙ h_{t−1})) — a proposed new hidden state, conditioned on the reset-gated previous state.
4. **New hidden state** h_t = z·h_{t−1} + (1−z)·h̃ — linear interpolation governed by the update gate.

The **encoder** reads the source sequence token by token with a GRU and produces a fixed-length context vector *c* (the final hidden state). The **decoder** is another GRU that, starting from *c*, generates the target sequence one token at a time, conditioning each step on its own previous hidden state and the previously emitted token.

### Key method details relevant to the code

- The GRU's two-gate design (reset + update) is the heart of this paper and what the notebook implements from scratch with raw matrix operations.
- The encoder–decoder conditional probability formulation p(y_1,…,y_T' | x_1,…,x_T) = ∏ p(y_t | y_{<t}, c) is the training objective.
- Both networks trained jointly end-to-end; in the notebook we train on a sequence-memorisation task and compare GRU vs. vanilla RNN.
- The paper uses 1000 hidden units, rank-100 input/output approximations (100-dim embeddings), tanh for the candidate, and adadelta. The notebook uses smaller dimensions for speed but the same gate equations.

### Influence

The GRU introduced here became one of the two standard gated recurrent cells in deep learning (alongside LSTM). It was adopted in Google's neural machine translation system, in speech recognition models, and in countless sequence-modeling tutorials. The RNN Encoder–Decoder framework from this paper directly inspired the attention-based model of Bahdanau et al. (2014), which in turn led to the Transformer. As of 2026, the paper has been cited over 25,000 times on Google Scholar.

## What problem does it solve?

Imagine you're trying to memorise a phone number someone just told you. If it's short — like 5 digits — you can hold it in your head easily. But if someone reads out a 15-digit number, by the time they say the last few digits, you've probably forgotten the first ones. That's exactly the problem a regular RNN has: as it reads a long sequence one step at a time, the early information fades away and gets overwritten by new stuff.

The GRU fixes this by giving the network two little "decision-makers" called **gates**. One gate (the *update gate*) acts like a bouncer at a club — it decides "should I keep the old memory or replace it with something new?" The other gate (the *reset gate*) acts like an eraser — it decides "when I'm thinking about what comes next, should I ignore the old stuff and start fresh?" With these two gates, the GRU can hold onto important information for a long time and forget unimportant stuff quickly — just like how *you* might repeat the first few digits of that phone number to yourself while listening to the rest.

This paper also paired the GRU with an **encoder–decoder** setup: one network "reads" a sentence in English and turns it into a compact summary, and another network "writes" the French translation from that summary. This was a big deal in 2014 because it showed neural networks could help translate languages better than traditional systems that relied on hand-crafted rules.
